#pragma once

#ifdef WITH_XLA

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
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

  at::optional<std::vector<at::Tensor>> run(
      const std::vector<at::Tensor>& inputs) const;

 private:
  at::optional<std::vector<xla::Shape>> captureInputShapes(
      const std::vector<at::Tensor>& inputs) const;

  at::optional<xla::XlaComputation> buildXlaComputation(
      const std::vector<xla::Shape>& parameter_shapes) const;

  std::shared_ptr<Graph> graph_;
  xla::XlaComputationClient client_;
};

} // namespace jit
} // namespace torch

#endif // WITH_XLA
