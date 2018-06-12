#pragma once

#ifdef WITH_XLA

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "torch/csrc/jit/ir.h"
#include "ATen/optional.h"

namespace torch {
namespace jit {

class XlaCodeImpl {
 public:
  XlaCodeImpl(const std::shared_ptr<Graph>& graph);

  bool runComputation(Stack& stack, size_t current_stage) const;

 private:
  void initStageEnd();

  size_t startNodeIndex(size_t current_stage) const;

  size_t endNodeIndex(size_t current_stage) const;

  // Retrieves the tensor shapes from the stack, nullopt if not supported.
  at::optional<std::vector<xla::Shape>> captureInputShapes(
      Stack& stack, size_t current_stage) const;

  // Builds the XLA computation for the given fusion group, nullopt if not
  // supported.
  static at::optional<xla::XlaComputation> buildFusionGroupXlaComputation(
      Graph* fusion_subgraph);

  // Retrieves the tensor shapes from the fusion group inputs, nullopt if not
  // supported.
  static at::optional<std::vector<xla::Shape>> captureInputShapes(
      Graph* fusion_subgraph);

  // Creats a map from fusion group input parameter indices to XlaParameter's.
  static std::unordered_map<size_t, xla::XlaOp> bindInputs(
      Graph* fusion_subgraph, const std::vector<xla::Shape>& parameter_shapes,
      xla::XlaBuilder* b);

  // Helper for XLA binary operators.
  static xla::XlaOp buildBinaryXlaOp(const NodeKind kind, const xla::XlaOp& lhs,
                                     const xla::XlaOp& rhs, xla::XlaBuilder* b);

  std::vector<Node*> nodes_;       // sequence of nodes in the graph
  std::vector<size_t> stage_end_;  // store the end of each stage of computation
  std::shared_ptr<Graph> graph_;
};

}  // namespace jit
}  // namespace torch

#endif  // WITH_XLA
