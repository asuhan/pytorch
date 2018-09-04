#ifndef XLA_TENSOR_H
#define XLA_TENSOR_H

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

using int64 = long long;

class XLATensorData;

class XLATensor : public std::enable_shared_from_this<XLATensor> {
 public:
  TH_DISALLOW_COPY_AND_ASSIGN(XLATensor);
  XLATensor(const autograd::Variable&);
  XLATensor(
      std::unique_ptr<xla::GlobalData>,
      const xla::Shape& shape,
      const std::vector<int64>& logical_shape);

  virtual at::Tensor toTensor();

  virtual std::shared_ptr<XLATensor> grad() const;

  std::shared_ptr<XLATensorData> data() const;

  virtual void setGrad(std::shared_ptr<XLATensor> grad);

  virtual const std::vector<int64>& logicalShape() const;

  virtual xla::Shape shape() const;

  virtual std::vector<int64_t> size() const;

  virtual xla::GlobalData* xlaData() const;

  // Basic tensor operations used by the optimizers.
  virtual void add_(XLATensor& other, const at::Scalar& alpha);

  virtual void mul_(XLATensor& other);

  virtual void mul_(const at::Scalar& other);

  virtual void zero_();

  virtual void detach_();

  // Applies the queue of operations for a list of tensors.
  static void applyOpsMulti(
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  // In place scale and add for multiple tensors.
  static void mulAddMulti(
      const double scale_dest,
      const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
      const double alpha,
      const std::vector<std::shared_ptr<XLATensor>>& source_tuple);

  static void zeroMulti(
      const std::vector<std::shared_ptr<XLATensor>>& dest_tuple);

 private:
  XLATensor() : requires_grad_(false) {}

  // Applies the queue of operations in preparation for using the data.
  virtual void applyOps();

  static void setMultiFromResult(
      const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
      std::vector<std::unique_ptr<xla::GlobalData>>& new_dest_elements);

  std::shared_ptr<XLATensorData> data_;
  bool requires_grad_;

  friend class XLATensorData;
};

class XLATensorData : public XLATensor {
 public:
  TH_DISALLOW_COPY_AND_ASSIGN(XLATensorData);
  XLATensorData(const autograd::Variable&);
  XLATensorData(
      std::unique_ptr<xla::GlobalData>,
      const xla::Shape& shape,
      const std::vector<int64>& logical_shape);

  at::Tensor toTensor() override;

  std::shared_ptr<XLATensor> grad() const override;
  void setGrad(std::shared_ptr<XLATensor> grad) override;
  const std::vector<int64>& logicalShape() const override;
  xla::Shape shape() const override;
  std::vector<int64_t> size() const override;
  xla::GlobalData* xlaData() const override;

  // Basic tensor operations used by the optimizers.
  void add_(XLATensor& other, const at::Scalar& alpha) override;
  void mul_(XLATensor& other) override;
  void mul_(const at::Scalar& other) override;
  void zero_() override;

  // Applies the queue of operations in preparation for using the data.
  void applyOps() override;

 private:
  void resetOperationsState();

  std::unique_ptr<xla::GlobalData> xla_data_;
  xla::Shape shape_;
  std::vector<int64> logical_shape_;
  xla::PrimitiveType dtype_; // naming dtype for consistency with at::Tensor
  /* std::shared_ptr<xla::XlaComputation> grad_fn; */
  std::shared_ptr<XLATensor> grad_;
  // Keeps track of operations applied so far.
  at::optional<xla::XlaOp> operations_;
  xla::XlaBuilder b_;
  std::vector<xla::GlobalData*> operations_params_;

  friend class XLATensor;
};

} // namespace jit
} // namespace torch

#endif
