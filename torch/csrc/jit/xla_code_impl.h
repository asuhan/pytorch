#pragma once

#ifdef WITH_XLA

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "torch/csrc/jit/aten_dispatch.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

struct XlaExecutionStatus {
  size_t pc_delta;
  bool ok;
};

class XlaCodeImpl {
 public:
  XlaCodeImpl(const std::shared_ptr<Graph>& graph);

  at::optional<at::Tensor> run(const std::vector<at::Tensor>& inputs) const;

 private:
  at::optional<std::vector<xla::Shape>> captureInputShapes(
      const std::vector<at::Tensor>& inputs) const;

  at::optional<xla::XlaComputation> buildXlaComputation(
      const std::vector<xla::Shape>& parameter_shapes) const;

  xla::XlaOp buildBinaryXlaOp(const NodeKind kind, const xla::XlaOp& lhs,
                              const xla::XlaOp& rhs, xla::XlaBuilder* b) const;

  std::shared_ptr<Graph> graph_;
};

}  // namespace jit
}  // namespace torch

#endif  // WITH_XLA
