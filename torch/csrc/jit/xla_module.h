#pragma once
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/xla_code_impl.h"
#include "torch/csrc/jit/xla_tensor.h"
#include "torch/csrc/utils/disallow_copy.h"

#include <memory>

namespace torch {
namespace jit {

struct XlaModule : public std::enable_shared_from_this<XlaModule> {
  TH_DISALLOW_COPY_AND_ASSIGN(XlaModule);

  XlaModule(script::Module& module,
	    std::vector<autograd::Variable>& inputs,
	    bool backward=true);

  // std::vector<XLATensor> run(const std::vector<XLATensor>& inputs);

 private:
  std::vector<std::shared_ptr<XLATensor>> params_;
};

} // namespace jit
} // namespace torch
