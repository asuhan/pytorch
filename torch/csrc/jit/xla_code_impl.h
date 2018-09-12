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

  static at::optional<xla::XlaComputation> buildXlaComputation(
      const Block* block,
      const std::vector<xla::Shape>& parameter_shapes,
      const std::map<size_t, xla::XlaOp>& init_node_xla_ops,
      const std::vector<xla::XlaOp>& captured_inputs,
      const at::optional<xla::XlaOp> iteration_counter,
      xla::XlaBuilder* b);

  std::shared_ptr<Graph> graph_;
};

xla::XlaComputationClient* XlaGetClient();

} // namespace jit
} // namespace torch

#endif // WITH_XLA
