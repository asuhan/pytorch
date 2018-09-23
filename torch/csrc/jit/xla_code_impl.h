#pragma once

#ifdef WITH_XLA

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

struct XlaComputationInOut {
  std::vector<xla::XlaOp> inputs;
  std::vector<xla::XlaOp> outputs;
};

class XlaCodeImpl {
 public:
  XlaCodeImpl(const std::shared_ptr<Graph>& graph);

  at::optional<xla::XlaComputation> buildXlaComputation(
      const std::vector<xla::Shape>& parameter_shapes,
      size_t param_to_return_count) const;

  at::optional<XlaComputationInOut> buildInlinedXlaComputation(
      const std::vector<xla::Shape>& parameter_shapes,
      xla::XlaBuilder* b) const;

  static std::pair<xla::XlaOp, xla::XlaOp> CrossEntropyWithLogits(
      const xla::XlaOp& logits,
      const xla::XlaOp& sparse_labels,
      xla::XlaBuilder* b);

  std::shared_ptr<Graph> graph_;
};

xla::XlaComputationClient* XlaGetClient();

} // namespace jit
} // namespace torch

#endif // WITH_XLA
