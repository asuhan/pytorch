#pragma once

#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/xla_module.h"

namespace torch {
namespace jit {

std::shared_ptr<XlaModule> ToXLA(script::Module& module);

std::shared_ptr<XlaModule> ToXLAGrad(std::shared_ptr<Graph> graph);

} // namespace jit
} // namespace torch
