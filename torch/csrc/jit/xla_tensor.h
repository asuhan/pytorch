#ifndef XLA_TENSOR_H
#define XLA_TENSOR_H

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "torch/csrc/jit/aten_dispatch.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

class XLATensor : public std::enable_shared_from_this<XLATensor> {
 public:
  TH_DISALLOW_COPY_AND_ASSIGN(XLATensor);
  XLATensor(const autograd::Variable&);
  XLATensor(const xla::Literal&);

  at::Tensor toTensor();

  std::shared_ptr<XLATensor> grad() const;
  void setGrad(std::shared_ptr<XLATensor> grad);

  xla::Shape shape() const;
  xla::GlobalData* data() const;

  void add_(const XLATensor& other, const at::Scalar& alpha);
  void mul_(const XLATensor& other);

 private:
  std::unique_ptr<xla::GlobalData> data_;
  xla::Shape shape_;
  xla::PrimitiveType dtype_; // naming dtype for consistency with at::Tensor
  bool requires_grad_;
  /* std::shared_ptr<xla::XlaComputation> grad_fn; */
  std::shared_ptr<XLATensor> grad_;
};

} // namespace jit
} // namespace torch

#endif
