#include "torch/csrc/jit/passes/xla.h"

namespace torch { namespace jit {

std::shared_ptr<XlaModule> ToXLA(script::Module& module) {
  const auto method = module.find_method("forward");
  assert(method);
  std::vector<at::Tensor> model_parameters;
  for (const auto& model_parameter_it : module.get_parameters()) {
    model_parameters.push_back(*(model_parameter_it->slot()));
  }
  return std::make_shared<XlaModule>(method, model_parameters);
}

}}
