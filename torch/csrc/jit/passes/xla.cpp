#include "torch/csrc/jit/passes/xla.h"

namespace torch {
namespace jit {

namespace {

static void gather_parameters(
    std::vector<at::Tensor>& model_parameters,
    const script::Module& module) {
  for (auto& param : module.get_parameters()) {
    model_parameters.push_back(*param->slot());
  }
  for (const auto& sub : module.get_modules()) {
    gather_parameters(model_parameters, *sub->module);
  }
}

} // namespace

std::shared_ptr<XlaModule> ToXLA(script::Module& module) {
  const auto method = module.find_method("forward");
  assert(method);
  std::vector<at::Tensor> model_parameters;
  gather_parameters(model_parameters, module);
  return std::make_shared<XlaModule>(method, model_parameters);
}

} // namespace jit
} // namespace torch
