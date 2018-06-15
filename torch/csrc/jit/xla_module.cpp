#include "torch/csrc/jit/xla_module.h"

namespace torch { namespace jit {

at::Tensor XlaModule::run(const std::vector<at::Tensor>& inputs) {
  const auto maybe_tensor = xla_code_.run(inputs);
  if (!maybe_tensor.has_value()) {
    throw std::runtime_error("Failed to run XLA module");
  }
  return *maybe_tensor;
}

}}
