#pragma once

#ifdef WITH_XLA

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

using int64 = long long;

struct XlaComputationResult {
  xla::XlaComputation computation;
  std::vector<std::vector<int64>> ret_logical_shapes;
};

class XlaCodeImpl {
 public:
  XlaCodeImpl(const std::shared_ptr<Graph>& graph);

  at::optional<std::vector<xla::Shape>> captureInputShapes(
      const std::vector<at::Tensor>& inputs) const;

  at::optional<XlaComputationResult> buildXlaComputation(
      const std::vector<xla::Shape>& parameter_shapes,
      const std::vector<std::vector<int64>>& logical_parameter_shapes) const;

  std::shared_ptr<Graph> graph_;
};

xla::XlaComputationClient* XlaGetClient();

} // namespace jit
} // namespace torch

#endif // WITH_XLA
