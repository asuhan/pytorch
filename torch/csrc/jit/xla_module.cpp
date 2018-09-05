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
    : forward_graph_initialized_(false), backward_graph_initialized_(false) {
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

namespace {

std::vector<std::vector<int64>> get_logical_shapes(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  std::vector<std::vector<int64>> logical_shapes;
  for (const auto& tensor : tensors) {
    if (tensor) {
      logical_shapes.push_back(tensor->logicalShape());
    } else {
      logical_shapes.emplace_back();
    }
  }
  return logical_shapes;
}

} // namespace

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
  if (!forward_graph_initialized_) {
    std::vector<xla::Shape> forward_shapes;
    for (auto p : inputs_params_buffers) {
      forward_shapes.push_back(p->shape());
    }

    XlaCodeImpl xla_fwd_impl(f_);
    auto maybe_computation = xla_fwd_impl.buildXlaComputation(
        forward_shapes, get_logical_shapes(inputs_params_buffers));
    if (!maybe_computation) {
      AT_ERROR("Failed to build XlaComputation");
    }
    forward_graph_ = std::move(maybe_computation->computation);
    forward_ret_logical_shapes_ = maybe_computation->ret_logical_shapes;
    const auto program_shape = forward_graph_.GetProgramShape().ValueOrDie();
    const auto result_shape = program_shape.result();
    if (xla::ShapeUtil::IsTuple(result_shape)) {
      for (const auto& element_shape : result_shape.tuple_shapes()) {
        forward_ret_shape_cache_.push_back(element_shape);
      }
    } else {
      forward_ret_shape_cache_.push_back(result_shape);
    }
    forward_graph_initialized_ = true;
  }

  std::vector<tensorflow::Output*> inputs_params_buffers_data;
  for (auto p : inputs_params_buffers) {
    inputs_params_buffers_data.push_back(p->xlaData());
  }
  auto result_dh =
      client_->ExecuteComputation(forward_graph_, inputs_params_buffers_data);
  std::vector<std::shared_ptr<XLATensor>> raw_outputs;

  std::vector<xla::Shape> forward_ret_shape;
  // if return value is a tuple,
  CHECK_EQ(forward_ret_logical_shapes_.size(), forward_ret_shape_cache_.size());
  if (forward_ret_shape_cache_.size() > 1) {
    auto tuple_elements = client_->DeconstructTuple(*result_dh).ValueOrDie();
    CHECK_EQ(forward_ret_shape_cache_.size(), tuple_elements.size());
    for (size_t i = 0; i < tuple_elements.size(); ++i) {
      auto& tuple_element = tuple_elements[i];
      auto raw_output = std::make_shared<XLATensor>(
          std::move(tuple_element),
          forward_ret_shape_cache_[i],
          forward_ret_logical_shapes_[i]);
      raw_outputs.push_back(raw_output);
    }
  } else {
    CHECK_EQ(forward_ret_shape_cache_.size(), size_t(1));
    auto raw_output = std::make_shared<XLATensor>(
        std::move(result_dh),
        forward_ret_shape_cache_[0],
        forward_ret_logical_shapes_[0]);
    raw_outputs.push_back(raw_output);
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
    raw_grad_outputs.push_back(nullptr);
  }

  for (auto p : captured_inputs_outputs_) {
    raw_grad_outputs.push_back(p);
  }

  auto client_ = XlaGetClient();

  // if backward graph is not compiled, compile it
  if (!backward_graph_initialized_) {
    std::vector<xla::Shape> backward_shapes;
    for (auto p : raw_grad_outputs) {
      if (p) {
        backward_shapes.push_back(p->shape());
      } else {
        backward_shapes.emplace_back();
      }
    }

    XlaCodeImpl xla_bwd_impl(df_);
    auto maybe_computation = xla_bwd_impl.buildXlaComputation(
        backward_shapes, get_logical_shapes(raw_grad_outputs));
    if (!maybe_computation) {
      AT_ERROR("Failed to build backward XlaComputation");
    }
    backward_graph_ = std::move(maybe_computation->computation);
    backward_ret_logical_shapes_ = maybe_computation->ret_logical_shapes;
    const auto program_shape = backward_graph_.GetProgramShape().ValueOrDie();
    const auto result_shape = program_shape.result();
    if (xla::ShapeUtil::IsTuple(result_shape)) {
      for (const auto& element_shape : result_shape.tuple_shapes()) {
        backward_ret_shape_cache_.push_back(element_shape);
      }
    } else {
      backward_ret_shape_cache_.push_back(result_shape);
    }
    backward_graph_initialized_ = true;
  }

  std::vector<tensorflow::Output*> raw_grad_outputs_data;
  for (auto p : raw_grad_outputs) {
    if (!p) {
      continue;
    }
    auto ptr = p->xlaData();
    raw_grad_outputs_data.push_back(ptr);
  }

  auto result_dh =
      client_->ExecuteComputation(backward_graph_, raw_grad_outputs_data);

  std::vector<std::shared_ptr<XLATensor>> grad_inputs;
  std::vector<xla::Shape> backward_ret_shape;
  // convert tuples into vector of XLATensor
  CHECK_EQ(
      backward_ret_logical_shapes_.size(), backward_ret_shape_cache_.size());
  if (backward_ret_shape_cache_.size() > 1) {
    auto tuple_elements = client_->DeconstructTuple(*result_dh).ValueOrDie();
    CHECK_EQ(backward_ret_shape_cache_.size(), tuple_elements.size());
    for (size_t i = 0; i < tuple_elements.size(); ++i) {
      auto& tuple_element = tuple_elements[i];
      auto grad_input = std::make_shared<XLATensor>(
          std::move(tuple_element),
          backward_ret_shape_cache_[i],
          backward_ret_logical_shapes_[i]);
      grad_inputs.push_back(grad_input);
    }
  } else {
    CHECK_EQ(backward_ret_shape_cache_.size(), size_t(1));
    auto grad_input = std::make_shared<XLATensor>(
        std::move(result_dh),
        backward_ret_shape_cache_[0],
        backward_ret_logical_shapes_[0]);
    grad_inputs.push_back(grad_input);
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
