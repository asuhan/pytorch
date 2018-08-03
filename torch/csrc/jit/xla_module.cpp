#include "torch/csrc/jit/xla_module.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/decompose_addmm.h"
#include "torch/csrc/jit/passes/specialize_undef.h"
#include "torch/csrc/jit/passes/unwrap_buffered_functions.h"
#include "torch/csrc/jit/passes/constant_folding.h"

namespace torch {
namespace jit {

  XlaModule::XlaModule(script::Module& module,
		       std::vector<autograd::Variable>& inputs,
		       bool backward) : backward_graph_initialized(false) {

  const auto forward = module.find_method("forward");
  assert(forward);

  // get forward graph
  auto fgraph = forward->graph();

  // run forward passes
  DecomposeAddmm(fgraph);
  UnwrapBufferedFunctions(fgraph);
  ConstantFold(fgraph);
  EliminateDeadCode(fgraph);
  
  // convert model parameters to vector of XLATensors
  for (auto p : forward->params()) {
    params_.push_back(std::make_shared<XLATensor>(autograd::as_variable_ref(*p)));
  }

  // if backward is true, differentiate graph
  std::vector<bool> inputs_require_grad;
  for (auto p : inputs) {
    inputs_require_grad.push_back(p.requires_grad());
  }
  for (auto p : params_) {
    inputs_require_grad.push_back(p.get()->requires_grad);
  }

  // convert forward and backward graphs to XLAOp
  auto fgraph_copy = fgraph->copy();
  Gradient gradient = differentiate(fgraph_copy, inputs_require_grad);

  // run forward passes
  DecomposeAddmm(gradient.f);
  UnwrapBufferedFunctions(gradient.f);
  ConstantFold(gradient.f);
  EliminateDeadCode(gradient.f);
  // run backward passes
  std::vector<bool> defined;
  for (auto i : gradient.df->inputs()) {
    defined.push_back(true);
  }
  specializeUndef(*(gradient.df.get()), defined);
  ConstantFold(gradient.df);
  EliminateDeadCode(gradient.df);

  // record some graph information
  f_real_outputs = gradient.f_real_outputs;
  df_input_captured_inputs = gradient.df_input_captured_inputs;
  df_input_captured_outputs = gradient.df_input_captured_outputs;
  

  // Now convert the forward and backward graphs to XlaOp
  std::vector<xla::Shape> forward_shapes;
  for (auto p : inputs) {
    forward_shapes.push_back(XLATensor(autograd::as_variable_ref(p)).shape);
  }
  for (auto p : params_) {
    forward_shapes.push_back(p.get()->shape);
  }
  
  XlaCodeImpl xla_fwd_impl(gradient.f);
  auto maybe_computation = xla_fwd_impl.buildXlaComputation(forward_shapes);
  if (!maybe_computation) {
    std::runtime_error("Failed to build XlaComputation");
  }
  forward_graph_ = std::move(*maybe_computation);

  df_ = gradient.df;
}

  std::vector<std::shared_ptr<XLATensor> > XlaModule::forward(const std::vector<std::shared_ptr<XLATensor> >& inputs) {

    // clear the previous forward's captured vectors.
    // This is needed in case backward is not yet run, but two forward calls were made
  inputs_.clear();
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();

  inputs_ = inputs; // needed so that in backward, we can set .grad attributes correctly
  
  auto client_ = XlaGetClient();
  std::vector<std::shared_ptr<XLATensor> > inputs_params_buffers;
  for (auto p : inputs) {
    inputs_params_buffers.push_back(p);
  }
  for (auto p : params_) {
    inputs_params_buffers.push_back(p);
  }
  
  std::vector<xla::GlobalData*> inputs_params_buffers_data;
  for (auto p : inputs_params_buffers) {
    inputs_params_buffers_data.push_back(p.get()->data_.get());
  }
  // TODO: move to ExecuteComputation (avoid transfer)
  // for that, one needs to know how to construct XLATensor from xla::GlobalData*
  auto result_literal = client_->ExecuteComputationAndTransfer(
	       forward_graph_, inputs_params_buffers_data);
  std::vector<std::shared_ptr<XLATensor> > raw_outputs;

  // if return value is a tuple, 
  if (xla::ShapeUtil::IsTuple(result_literal->shape())) {
    const std::vector<xla::Literal> tuple_elements = result_literal->DecomposeTuple();
    for (const auto& tuple_element : tuple_elements) {
      raw_outputs.push_back(std::make_shared<XLATensor>(tuple_element));
    }
  } else {
    raw_outputs.push_back(std::make_shared<XLATensor>(*result_literal));
  }

  // filter out real outputs from backward-captured outputs
  std::vector<std::shared_ptr<XLATensor> > outputs;
  for (uint64_t i=0; i < f_real_outputs; i++) {
    outputs.push_back(raw_outputs[i]);
  }

  // set backward-captured outputs on Module (for now, TODO: later move to a handle on output Tensor)
  for (uint64_t i=f_real_outputs; i < raw_outputs.size(); i++) {
    captured_outputs_.push_back(raw_outputs[i]);
  }

  for (auto i : df_input_captured_inputs) {
    captured_inputs_outputs_.push_back(inputs_params_buffers[i]);
  }
  for (auto i : df_input_captured_outputs) {
    captured_inputs_outputs_.push_back(raw_outputs[i]);
  }

  return outputs;
}

void XlaModule::backward(const std::vector<std::shared_ptr<XLATensor> >& grad_outputs) {
    std::vector<std::shared_ptr<XLATensor> > raw_grad_outputs;

    for (auto p : grad_outputs) {
      raw_grad_outputs.push_back(p);
    }

    for (auto p : captured_outputs_) {
      // dummy all zeros grad outputs for captured_outputs
      auto shape = p->shape;
      std::vector<int64_t> dims;
      for (const auto d : shape.dimensions()) {
    	dims.push_back(d);
      }
      raw_grad_outputs.push_back(std::make_shared<XLATensor>(autograd::make_variable(at::zeros(dims))));
    }

    for (auto p : captured_inputs_outputs_) {
      raw_grad_outputs.push_back(p);
    }

    // if backward graph is not compiled, compile it
    if (!backward_graph_initialized) {
      std::vector<xla::Shape> backward_shapes;
      for (auto p : raw_grad_outputs) {
    	backward_shapes.push_back(p.get()->shape);
      }

      XlaCodeImpl xla_bwd_impl(df_);
      auto maybe_computation = xla_bwd_impl.buildXlaComputation(backward_shapes);
      if (!maybe_computation) {
    	std::runtime_error("Failed to build backward XlaComputation");
      }
      backward_graph_ = std::move(*maybe_computation);
    }


    std::vector<xla::GlobalData* > raw_grad_outputs_data;
    for (auto p : raw_grad_outputs) {
      xla::GlobalData* ptr = p.get()->data_.get();
      raw_grad_outputs_data.push_back(ptr);
    }

    
    auto client_ = XlaGetClient();
    auto result_literal = client_->ExecuteComputationAndTransfer(
    	       backward_graph_, raw_grad_outputs_data);

    // convert tuple literals into vector of XLATensor
    std::vector<std::shared_ptr<XLATensor> > grad_inputs;
    if (xla::ShapeUtil::IsTuple(result_literal->shape())) {
      const std::vector<xla::Literal> tuple_elements = result_literal->DecomposeTuple();
      for (const auto& tuple_element : tuple_elements) {
    	grad_inputs.push_back(std::make_shared<XLATensor>(tuple_element));
      }
    } else {
      grad_inputs.push_back(std::make_shared<XLATensor>(*result_literal));
    }

    // now set .grad attributes of the input and param tensors
    for (int i = 0; i < inputs_.size(); i++) {
      inputs_[i]->grad = grad_inputs[i];
    }

    for (int i = 0; i < params_.size(); i++) {
      params_[i]->grad = grad_inputs[i + inputs_.size()];
    }

    // release handles to saved / captured inputs and outputs
    inputs_.clear();
    captured_outputs_.clear();
    captured_inputs_outputs_.clear();
  }

} // namespace jit
} // namespace torch
