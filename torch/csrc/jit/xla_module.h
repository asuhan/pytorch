#pragma once
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/xla_code_impl.h"
#include "torch/csrc/utils/disallow_copy.h"

#include <memory>

namespace torch { namespace jit {

struct XlaModule : public std::enable_shared_from_this<XlaModule> {
  TH_DISALLOW_COPY_AND_ASSIGN(XlaModule);

  XlaModule(const script::Method* method) : xla_code_(method->graph()) {}

  at::Tensor run(const std::vector<at::Tensor>& inputs);

 private:
  XlaCodeImpl xla_code_;
};

}}
