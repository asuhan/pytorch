#pragma once
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/xla_code_impl.h"
#include "torch/csrc/utils/disallow_copy.h"

#include <memory>

namespace torch {
namespace jit {

struct XlaModule : public std::enable_shared_from_this<XlaModule> {
  TH_DISALLOW_COPY_AND_ASSIGN(XlaModule);

  XlaModule(
      const script::Method* method,
      const std::vector<at::Tensor>& model_parameters)
      : xla_code_(method->graph()), model_parameters_(model_parameters) {}

  at::Tensor run(const std::vector<at::Tensor>& inputs);

 private:
  XlaCodeImpl xla_code_;
  std::vector<at::Tensor> model_parameters_;
};

} // namespace jit
} // namespace torch
