#pragma once

#include "torch/csrc/jit/autodiff.h"

namespace torch {
namespace jit {

void XlaRemoveUnused(Gradient& gradient);
}
} // namespace torch
