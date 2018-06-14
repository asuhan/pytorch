#pragma once
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/jit/script/module.h"

#include <memory>

namespace torch { namespace jit {

struct XlaModule : public std::enable_shared_from_this<XlaModule> {
  TH_DISALLOW_COPY_AND_ASSIGN(XlaModule);

  XlaModule(const script::Method* method) : method_(method) {}

  at::Tensor run(const std::vector<at::Tensor>& inputs);

private:
  const script::Method* method_;
};

}}
