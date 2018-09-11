#pragma once

#ifdef WITH_XLA

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

class XlaCodeImpl {
 public:
  XlaCodeImpl(const std::shared_ptr<Graph>& graph);

  at::optional<xla::XlaComputation> buildXlaComputation(
      const std::vector<xla::Shape>& parameter_shapes) const;

  std::shared_ptr<Graph> graph_;
};

xla::XlaComputationClient* XlaGetClient();

} // namespace jit
} // namespace torch

#endif // WITH_XLA
