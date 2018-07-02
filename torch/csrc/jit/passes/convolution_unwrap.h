#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

void ConvolutionUnwrap(const std::shared_ptr<Graph>& graph);

}}
