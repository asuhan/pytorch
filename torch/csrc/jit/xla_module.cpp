#include "torch/csrc/jit/xla_module.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/decompose_addmm.h"
#include "torch/csrc/jit/passes/specialize_undef.h"
#include "torch/csrc/jit/passes/unwrap_buffered_functions.h"
#include "torch/csrc/jit/passes/constant_folding.h"

namespace torch {
namespace jit {

  // std::vector<XLATensor> XlaModule::run(const std::vector<XLATensor>& inputs) {
  // std::vector<XLATensor> inputs_and_parameters(inputs.begin(), inputs.end());
  // inputs_and_parameters.insert(
  //     inputs_and_parameters.end(),
  //     model_parameters_.begin(),
  //     model_parameters_.end());
  // const auto maybe_result = xla_code_.run(inputs_and_parameters);
  // if (!maybe_result.has_value()) {
  //   throw std::runtime_error("Failed to run XLA module");
  // }
  // return *maybe_result;
  // }

  XlaModule::XlaModule(script::Module& module,
		       std::vector<autograd::Variable>& inputs,
		       bool backward) {

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
  Gradient gradient = differentiate(fgraph, inputs_require_grad);

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

  // Now convert the forward and backward graphs to XlaOp
  
  // in ::run, do the logic, and then set the parameter gradients to `.grad` correctly.

  
}

} // namespace jit
} // namespace torch
