#include "torch/csrc/jit/xla_module.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_folding.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/decompose_addmm.h"
#include "torch/csrc/jit/passes/specialize_undef.h"
#include "torch/csrc/jit/passes/unwrap_buffered_functions.h"
#include "torch/csrc/jit/passes/xla_remove_unused.h"

namespace {
using namespace torch::jit;
static void gatherParameters(
    std::vector<at::Tensor*>& values,
    std::vector<bool>& is_buffer,
    const script::Module& m) {
  for (auto& param : m.get_parameters()) {
    values.push_back(param->slot());
    is_buffer.push_back(param->is_buffer);
  }
  for (const auto& sub : m.get_modules()) {
    gatherParameters(values, is_buffer, *sub->module);
  }
}

} // namespace

namespace torch {
namespace jit {

XlaModule::XlaModule(
    script::Module& module,
    std::vector<autograd::Variable>& inputs,
    bool backward)
    : forward_graph_initialized(false), backward_graph_initialized(false) {
  const auto forward = module.find_method("forward");
  JIT_ASSERT(forward);

  // get forward graph
  auto fgraph = forward->graph();

  // run forward passes
  DecomposeAddmm(fgraph);
  UnwrapBufferedFunctions(fgraph);
  ConstantFold(fgraph);
  EliminateDeadCode(fgraph);

  std::vector<at::Tensor*> params_buffers_regather;
  gatherParameters(params_buffers_regather, is_buffer_, module);

  // convert model parameters to vector of XLATensors
  auto forward_params = forward->params();
  JIT_ASSERT(params_buffers_regather.size() == forward_params.size());

  for (uint32_t i = 0; i < forward_params.size(); i++) {
    JIT_ASSERT(forward_params[i] == params_buffers_regather[i]);
    auto p = forward_params[i];
    params_buffers_.push_back(
        std::make_shared<XLATensor>(autograd::as_variable_ref(*p)));
  }

  // if backward is true, differentiate graph
  std::vector<bool> inputs_require_grad;
  for (auto p : inputs) {
    inputs_require_grad.push_back(p.requires_grad());
  }

  for (uint32_t i = 0; i < params_buffers_.size(); i++) {
    if (!is_buffer_[i]) {
      params_.push_back(params_buffers_[i]);
    }
  }

  for (auto buf : is_buffer_) {
    inputs_require_grad.push_back(!buf);
  }

  // symbolically differentiate graph to get backward graph
  auto fgraph_copy = fgraph->copy();
  Gradient gradient = differentiate(fgraph_copy, inputs_require_grad);

  // run forward passes
  DecomposeAddmm(gradient.f);
  UnwrapBufferedFunctions(gradient.f);
  ConstantFold(gradient.f);
  EliminateDeadCode(gradient.f);
  // run backward passes
  std::vector<bool> defined(gradient.df->inputs().size(), true);
  specializeUndef(*(gradient.df.get()), defined);
  ConstantFold(gradient.df);
  EliminateDeadCode(gradient.df);

  // run pass on forward and backward graphs that drops outputs that XLA doesn't
  // need
  XlaRemoveUnused(gradient);

  // record some graph information
  f_real_outputs = gradient.f_real_outputs;
  df_input_captured_inputs = gradient.df_input_captured_inputs;
  df_input_captured_outputs = gradient.df_input_captured_outputs;

  f_ = gradient.f;
  df_ = gradient.df;
}

std::vector<std::shared_ptr<XLATensor>> XlaModule::forward(
    const std::vector<std::shared_ptr<XLATensor>>& inputs) {
  XLATensor::applyOpsMulti(inputs);
  XLATensor::applyOpsMulti(params_);
  // clear the previous forward's captured vectors.
  // This is needed in case backward is not yet run, but two forward calls were
  // made
  inputs_.clear();
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();

  inputs_ = inputs; // needed so that in backward, we can set .grad attributes
                    // correctly

  auto client_ = XlaGetClient();
  std::vector<std::shared_ptr<XLATensor>> inputs_params_buffers;
  for (auto p : inputs) {
    inputs_params_buffers.push_back(p);
  }
  for (auto p : params_buffers_) {
    inputs_params_buffers.push_back(p);
  }

  // Lazy-convert forward graph to XlaComputation
  if (!forward_graph_initialized) {
    std::vector<xla::Shape> forward_shapes;
    for (auto p : inputs_params_buffers) {
      forward_shapes.push_back(p.get()->shape());
    }

    XlaCodeImpl xla_fwd_impl(f_);
    auto maybe_computation = xla_fwd_impl.buildXlaComputation(forward_shapes);
    if (!maybe_computation) {
      AT_ERROR("Failed to build XlaComputation");
    }
    forward_graph_ = std::move(*maybe_computation);
  }

  std::vector<xla::GlobalData*> inputs_params_buffers_data;
  for (auto p : inputs_params_buffers) {
    inputs_params_buffers_data.push_back(p.get()->xlaData());
  }
  // TODO: move to ExecuteComputation (avoid transfer)
  // for that, one needs to know how to construct XLATensor from
  // xla::GlobalData*
  auto result_literal = client_->ExecuteComputationAndTransfer(
      forward_graph_, inputs_params_buffers_data);
  std::vector<std::shared_ptr<XLATensor>> raw_outputs;

  // if return value is a tuple,
  if (xla::ShapeUtil::IsTuple(result_literal->shape())) {
    const std::vector<xla::Literal> tuple_elements =
        result_literal->DecomposeTuple();
    for (const auto& tuple_element : tuple_elements) {
      raw_outputs.push_back(std::make_shared<XLATensor>(tuple_element));
    }
  } else {
    raw_outputs.push_back(std::make_shared<XLATensor>(*result_literal));
  }

  // filter out real outputs from backward-captured outputs
  std::vector<std::shared_ptr<XLATensor>> outputs;
  for (uint64_t i = 0; i < f_real_outputs; i++) {
    outputs.push_back(raw_outputs[i]);
  }

  // set backward-captured outputs on Module (for now, TODO: later move to a
  // handle on output Tensor)
  for (uint64_t i = f_real_outputs; i < raw_outputs.size(); i++) {
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

void XlaModule::backward(
    const std::vector<std::shared_ptr<XLATensor>>& grad_outputs) {
  XLATensor::applyOpsMulti(grad_outputs);
  XLATensor::applyOpsMulti(params_);
  std::vector<std::shared_ptr<XLATensor>> raw_grad_outputs;

  for (auto p : grad_outputs) {
    raw_grad_outputs.push_back(p);
  }

  for (auto p : captured_outputs_) {
    // dummy all zeros grad outputs for captured_outputs
    auto shape = p->shape();
    std::vector<int64_t> dims;
    for (const auto d : shape.dimensions()) {
      dims.push_back(d);
    }
    raw_grad_outputs.push_back(
        std::make_shared<XLATensor>(autograd::make_variable(at::zeros(dims))));
  }

  for (auto p : captured_inputs_outputs_) {
    raw_grad_outputs.push_back(p);
  }

  // if backward graph is not compiled, compile it
  if (!backward_graph_initialized) {
    std::vector<xla::Shape> backward_shapes;
    for (auto p : raw_grad_outputs) {
      backward_shapes.push_back(p.get()->shape());
    }

    XlaCodeImpl xla_bwd_impl(df_);
    auto maybe_computation = xla_bwd_impl.buildXlaComputation(backward_shapes);
    if (!maybe_computation) {
      AT_ERROR("Failed to build backward XlaComputation");
    }
    backward_graph_ = std::move(*maybe_computation);
  }

  std::vector<xla::GlobalData*> raw_grad_outputs_data;
  for (auto p : raw_grad_outputs) {
    xla::GlobalData* ptr = p.get()->xlaData();
    raw_grad_outputs_data.push_back(ptr);
  }

  auto client_ = XlaGetClient();
  auto result_literal = client_->ExecuteComputationAndTransfer(
      backward_graph_, raw_grad_outputs_data);

  // convert tuple literals into vector of XLATensor
  std::vector<std::shared_ptr<XLATensor>> grad_inputs;
  if (xla::ShapeUtil::IsTuple(result_literal->shape())) {
    const std::vector<xla::Literal> tuple_elements =
        result_literal->DecomposeTuple();
    for (const auto& tuple_element : tuple_elements) {
      grad_inputs.push_back(std::make_shared<XLATensor>(tuple_element));
    }
  } else {
    grad_inputs.push_back(std::make_shared<XLATensor>(*result_literal));
  }

  JIT_ASSERT((inputs_.size() + params_.size()) == grad_inputs.size());

  // now set .grad attributes of the input and param tensors
  for (size_t i = 0; i < inputs_.size(); i++) {
    inputs_[i]->setGrad(grad_inputs[i]);
  }

  for (size_t i = 0; i < params_.size(); i++) {
    auto t = grad_inputs[i + inputs_.size()];
    params_[i]->setGrad(t);
  }

  // release handles to saved / captured inputs and outputs
  inputs_.clear();
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();
}

std::vector<std::shared_ptr<XLATensor>> XlaModule::parameters() {
  return params_;
}

std::vector<std::shared_ptr<XLATensor>> XlaModule::parameters_buffers() {
  return params_buffers_;
}

} // namespace jit
} // namespace torch
