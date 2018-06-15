#include "torch/csrc/jit/xla_module.h"

namespace torch { namespace jit {

at::Tensor XlaModule::run(const std::vector<at::Tensor>& inputs) {
  std::vector<at::Tensor> inputs_and_parameters(inputs.begin(), inputs.end());
  inputs_and_parameters.insert(inputs_and_parameters.end(),
                               model_parameters_.begin(),
                               model_parameters_.end());
  const auto maybe_tensor = xla_code_.run(inputs_and_parameters);
  if (!maybe_tensor.has_value()) {
    throw std::runtime_error("Failed to run XLA module");
  }
  return *maybe_tensor;
}

}}
