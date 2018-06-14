#include "torch/csrc/jit/xla_module.h"

namespace torch { namespace jit {

at::Tensor XlaModule::run(const std::vector<at::Tensor>& inputs) {
  // TODO(asuhan)
  return inputs.front();
}

}}
