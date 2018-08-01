#ifndef XLA_TENSOR_H
#define XLA_TENSOR_H

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "torch/csrc/jit/aten_dispatch.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

struct XLATensor {
  XLATensor(const at::Tensor);

  std::unique_ptr<xla::GlobalData> data_;
  at::IntList sizes; // TODO: remove this and just use shape
  xla::Shape shape;
  xla::PrimitiveType dtype; // naming dtype for consistency with at::Tensor
  /* std::shared_ptr<xla::XlaComputation> grad_fn; */
  /* std::shared_ptr<XLATensor> grad; */

  at::Tensor toTensor();
};

} // namespace jit
} // namespace torch

#endif
