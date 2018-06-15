#include "torch/csrc/jit/passes/xla.h"

namespace torch { namespace jit {

std::shared_ptr<XlaModule> ToXLA(script::Module& module) {
  const auto method = module.find_method("forward");
  assert(method);
  return std::make_shared<XlaModule>(method);
}

}}
