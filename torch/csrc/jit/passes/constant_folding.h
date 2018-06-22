#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

void ConstantFold(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
