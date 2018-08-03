#ifndef XLA_TENSOR_H
#define XLA_TENSOR_H

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "torch/csrc/jit/aten_dispatch.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

struct XLATensor : public std::enable_shared_from_this<XLATensor> {
  TH_DISALLOW_COPY_AND_ASSIGN(XLATensor);
  XLATensor(const autograd::Variable&);
  XLATensor(const xla::Literal&);

  std::unique_ptr<xla::GlobalData> data_;
  xla::Shape shape;
  xla::PrimitiveType dtype; // naming dtype for consistency with at::Tensor
  bool requires_grad;
  /* std::shared_ptr<xla::XlaComputation> grad_fn; */
  std::shared_ptr<XLATensor> grad;

  at::Tensor toTensor();
};

} // namespace jit
} // namespace torch

#endif
