#include "xla_tensor.h"
#include "torch/csrc/autograd/variable.h"
#include "xla_code_impl.h"

namespace {

using int64 = long long;

std::vector<int64> xla_i64_list(const at::IntList& input) {
  std::vector<int64> output(input.size());
  std::copy(input.begin(), input.end(), output.begin());
  return output;
}

xla::Shape make_xla_shape(
    const at::IntList& tensor_dimensions,
    const xla::PrimitiveType type) {
  const auto dimensions = xla_i64_list(tensor_dimensions);
  std::vector<int64> layout(dimensions.size());
  // XLA uses minor-to-major.
  std::iota(layout.rbegin(), layout.rend(), 0);
  return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
}

at::optional<xla::PrimitiveType> make_xla_primitive_type(
    const at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Float:
      return xla::PrimitiveType::F32;
    case at::ScalarType::Long:
      return xla::PrimitiveType::S64;
    default:
      LOG(INFO) << "Type not supported: " << scalar_type;
      return at::nullopt;
  }
}

template <class NativeT>
void linearize_tensor(
    const at::Tensor& t,
    const size_t total_elements,
    void* literal_buffer);

template <>
void linearize_tensor<float>(
    const at::Tensor& t,
    const size_t total_elements,
    void* literal_buffer) {
  JIT_ASSERT(
      t.is_contiguous()); // the logic below works only for contiguous Tensors
  std::copy(
      t.data<float>(),
      t.data<float>() + total_elements,
      static_cast<float*>(literal_buffer));
}

template <>
void linearize_tensor<int64>(
    const at::Tensor& t,
    const size_t total_elements,
    void* literal_buffer) {
  JIT_ASSERT(
      t.is_contiguous()); // the logic below works only for contiguous Tensors
  std::copy(
      t.data<int64_t>(),
      t.data<int64_t>() + total_elements,
      static_cast<int64_t*>(literal_buffer));
}

template <class NativeT>
std::unique_ptr<xla::GlobalData> tensor_to_xla_impl(
    const at::Tensor& param_tensor,
    const xla::Shape& param_shape,
    const xla::XlaComputationClient* client) {
  size_t total_elements = 1;
  for (const auto dimension_size : param_tensor.sizes()) {
    total_elements *= dimension_size;
  }
  xla::Literal literal(param_shape);
  linearize_tensor<NativeT>(
      param_tensor, total_elements, literal.data<NativeT>().data());
  return client->TransferParameterToServer(literal);
}

std::unique_ptr<xla::GlobalData> tensor_to_xla(
    const at::Tensor& param_tensor,
    const xla::Shape& param_shape,
    const xla::XlaComputationClient* client) {
  switch (param_tensor.type().scalarType()) {
    case at::ScalarType::Float:
      return tensor_to_xla_impl<float>(param_tensor, param_shape, client);
    case at::ScalarType::Long:
      return tensor_to_xla_impl<int64>(param_tensor, param_shape, client);
    default:
      LOG(FATAL) << "Tensor type not supported";
  }
}

at::Tensor make_tensor_from_xla_literal(const xla::Literal& literal) {
  const auto& result_shape = literal.shape();
  std::vector<int64_t> dimensions;
  for (const auto result_dimension : result_shape.dimensions()) {
    dimensions.push_back(result_dimension);
  }
  auto literal_type = result_shape.element_type();
  switch (literal_type) {
    case xla::PrimitiveType::F32: {
      const auto result_slice = literal.data<float>();
      at::Tensor result_tensor = at::empty(at::CPU(at::kFloat), dimensions);
      std::copy(
          result_slice.begin(),
          result_slice.end(),
          result_tensor.data<float>());
      return result_tensor;
    }
    case xla::PrimitiveType::S64: {
      const auto result_slice = literal.data<int64>();
      at::Tensor result_tensor = at::empty(at::CPU(at::kLong), dimensions);
      std::copy(
          result_slice.begin(),
          result_slice.end(),
          result_tensor.data<int64_t>());
      return result_tensor;
    }
    default:
      AT_ERROR("Unsupported literal type");
  }
}

} // namespace

using namespace torch::jit;

XLATensor::XLATensor(const autograd::Variable& tensor)
    : data_(new XLATensorData(tensor)),
      requires_grad_(tensor.requires_grad()) {}

XLATensor::XLATensor(std::unique_ptr<xla::GlobalData> xla_data)
    : data_(new XLATensorData(std::move(xla_data))), requires_grad_(false) {}

XLATensor::XLATensor(
    std::unique_ptr<xla::GlobalData> xla_data,
    const xla::Shape& shape)
    : data_(new XLATensorData(std::move(xla_data), shape)),
      requires_grad_(false) {}

at::Tensor XLATensor::toTensor() {
  const auto t = data_->toTensor();
  const auto v = static_cast<const autograd::Variable&>(t);
  return autograd::make_variable(v.data(), requires_grad_);
}

std::shared_ptr<XLATensor> XLATensor::grad() const {
  return data_->grad();
}

std::shared_ptr<XLATensorData> XLATensor::data() const {
  return data_;
}

void XLATensor::setGrad(std::shared_ptr<XLATensor> grad) {
  data_->setGrad(grad);
}

xla::Shape XLATensor::shape() const {
  return data_->shape();
}
xla::GlobalData* XLATensor::xlaData() const {
  return data_->xlaData();
}

// Basic tensor operations used by the optimizers.
void XLATensor::add_(XLATensor& other, const at::Scalar& alpha) {
  data_->add_(other, alpha);
}

void XLATensor::mul_(XLATensor& other) {
  data_->mul_(other);
}

void XLATensor::mul_(const at::Scalar& other) {
  data_->mul_(other);
}

void XLATensor::zero_() {
  data_->zero_();
}

// Applies the queue of operations in preparation for using the data.
void XLATensor::applyOps() {
  data_->applyOps();
}

void XLATensor::detach_() {
  requires_grad_ = false;
}

void XLATensor::applyOpsMulti(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  // TODO(asuhan): Actually do a batch apply to minimize roundtrips.
  for (auto tensor : tensors) {
    tensor->applyOps();
  }
}

namespace {

// TODO(asuhan): de-dup with the version in xla_code_impl
std::vector<int64> xla_shape_sizes(const xla::Shape& shape) {
  std::vector<int64> shape_sizes(
      shape.dimensions().begin(), shape.dimensions().end());
  return shape_sizes;
}

} // namespace

void XLATensor::mulAddMulti(
    const double scale_dest,
    const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
    const double alpha,
    const std::vector<std::shared_ptr<XLATensor>>& source_tuple) {
  CHECK_EQ(dest_tuple.size(), source_tuple.size());
  applyOpsMulti(dest_tuple);
  applyOpsMulti(source_tuple);
  std::vector<xla::XlaOp> new_dest_tuple;
  std::vector<xla::GlobalData*> input_data;
  xla::XlaBuilder b("mulAddMulti");
  for (size_t i = 0; i < dest_tuple.size(); ++i) {
    auto old_dest =
        b.Parameter(2 * i, dest_tuple[i]->shape(), "dest_" + std::to_string(i));
    auto source = b.Parameter(
        2 * i + 1, source_tuple[i]->shape(), "source_" + std::to_string(i));
    if (alpha != 1) {
      const auto alpha_literal = xla::Literal::CreateR0<float>(alpha);
      const auto alpha_xla = b.ConstantLiteral(*alpha_literal);
      const auto alpha_source =
          b.Broadcast(alpha_xla, xla_shape_sizes(source_tuple[i]->shape()));
      source = b.Mul(source, alpha_source);
    }
    if (scale_dest != 1) {
      const auto scale_dest_literal = xla::Literal::CreateR0<float>(scale_dest);
      const auto scale_dest_xla = b.ConstantLiteral(*scale_dest_literal);
      const auto scale_dest_broadcast =
          b.Broadcast(scale_dest_xla, xla_shape_sizes(dest_tuple[i]->shape()));
      old_dest = b.Mul(old_dest, scale_dest_broadcast);
    }
    new_dest_tuple.push_back(b.Add(old_dest, source));
    input_data.push_back(dest_tuple[i]->xlaData());
    input_data.push_back(source_tuple[i]->xlaData());
  }
  b.Tuple(new_dest_tuple);
  auto computation = b.Build().ValueOrDie();
  auto client = XlaGetClient();
  auto result_tuple = client->ExecuteComputation(computation, input_data);
  auto new_dest_elements = client->DeconstructTuple(*result_tuple).ValueOrDie();
  setMultiFromResult(dest_tuple, new_dest_elements);
}

void XLATensor::zeroMulti(
    const std::vector<std::shared_ptr<XLATensor>>& dest_tuple) {
  applyOpsMulti(dest_tuple);
  xla::XlaBuilder b("zeroMulti");
  std::vector<xla::XlaOp> new_dest_tuple;
  for (auto& dest : dest_tuple) {
    const auto dest_shape = dest->shape();
    const auto zero =
        b.ConstantLiteral(xla::Literal::Zero(dest_shape.element_type()));
    new_dest_tuple.push_back(b.Broadcast(zero, xla_shape_sizes(dest_shape)));
  }
  b.Tuple(new_dest_tuple);
  auto computation = b.Build().ValueOrDie();
  auto client = XlaGetClient();
  auto result_tuple = client->ExecuteComputation(computation, {});
  auto new_dest_elements = client->DeconstructTuple(*result_tuple).ValueOrDie();
  setMultiFromResult(dest_tuple, new_dest_elements);
}

void XLATensor::setMultiFromResult(
    const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
    std::vector<std::unique_ptr<xla::GlobalData>>& new_dest_elements) {
  CHECK_EQ(new_dest_elements.size(), dest_tuple.size());
  for (size_t i = 0; i < dest_tuple.size(); ++i) {
    auto dest_tensor_data =
        std::dynamic_pointer_cast<XLATensorData>(dest_tuple[i]);
    if (dest_tensor_data) {
      dest_tensor_data->xla_data_ = std::move(new_dest_elements[i]);
    } else {
      dest_tuple[i]->data()->xla_data_ = std::move(new_dest_elements[i]);
    }
  }
}

XLATensorData::XLATensorData(const autograd::Variable& tensor)
    : grad_(nullptr), b_("XLATensor") {
  auto client_ = XlaGetClient();
  dtype_ = *make_xla_primitive_type(tensor.type().scalarType());
  shape_ = make_xla_shape(tensor.sizes(), dtype_);
  xla_data_ = tensor_to_xla(tensor, shape_, client_);
}

XLATensorData::XLATensorData(std::unique_ptr<xla::GlobalData> xla_data)
    : xla_data_(std::move(xla_data)), b_("XLATensor") {
  auto client_ = XlaGetClient();
  shape_ = client_->GetShape(*xla_data_).ValueOrDie();
  dtype_ = shape_.element_type();
}

XLATensorData::XLATensorData(
    std::unique_ptr<xla::GlobalData> xla_data,
    const xla::Shape& shape)
    : xla_data_(std::move(xla_data)),
      shape_(shape),
      dtype_(shape.element_type()),
      b_("XLATensor") {}

std::shared_ptr<XLATensor> XLATensorData::grad() const {
  return grad_;
}

void XLATensorData::setGrad(std::shared_ptr<XLATensor> grad) {
  grad_ = grad;
}

xla::Shape XLATensorData::shape() const {
  return shape_;
}

xla::GlobalData* XLATensorData::xlaData() const {
  return xla_data_.get();
}

at::Tensor XLATensorData::toTensor() {
  applyOps();
  // because there's no transferToClient, we'll define an `identity` graph, and
  // execute it
  xla::XlaBuilder b("identity");
  b.GetTupleElement(b.Tuple({b.Parameter(0, shape_, "x")}), 0);
  xla::XlaComputation identity = b.Build().ValueOrDie();

  auto client_ = XlaGetClient();
  auto result_literal =
      client_->ExecuteComputationAndTransfer(identity, {xla_data_.get()});
  auto return_tensor = make_tensor_from_xla_literal(*result_literal);
  return autograd::make_variable(return_tensor, false);
}

void XLATensorData::add_(XLATensor& other, const at::Scalar& alpha) {
  other.applyOps();
  const auto alpha_literal = xla::Literal::CreateR0<float>(alpha.toDouble());
  const auto alpha_xla = b_.ConstantLiteral(*alpha_literal);
  const auto old_tensor =
      operations_ ? *operations_ : b_.Parameter(0, shape_, "self");
  if (!operations_) {
    CHECK(operations_params_.empty());
    operations_params_.push_back(xla_data_.get());
  }
  operations_ = b_.Add(
      old_tensor,
      b_.Mul(
          b_.Parameter(operations_params_.size(), shape_, "other"),
          b_.Broadcast(alpha_xla, xla_shape_sizes(shape_))));
  operations_params_.push_back(other.xlaData());
}

void XLATensorData::mul_(XLATensor& other) {
  other.applyOps();
  const auto old_tensor =
      operations_ ? *operations_ : b_.Parameter(0, shape_, "self");
  if (!operations_) {
    CHECK(operations_params_.empty());
    operations_params_.push_back(xla_data_.get());
  }
  operations_ = b_.Mul(
      old_tensor, b_.Parameter(operations_params_.size(), shape_, "other"));
  operations_params_.push_back(other.xlaData());
}

void XLATensorData::mul_(const at::Scalar& other) {
  const auto old_tensor =
      operations_ ? *operations_ : b_.Parameter(0, shape_, "self");
  if (!operations_) {
    CHECK(operations_params_.empty());
    operations_params_.push_back(xla_data_.get());
  }
  const auto other_literal = xla::Literal::CreateR0<float>(other.toDouble());
  const auto other_xla = b_.ConstantLiteral(*other_literal);
  operations_ =
      b_.Mul(old_tensor, b_.Broadcast(other_xla, xla_shape_sizes(shape_)));
}

void XLATensorData::zero_() {
  resetOperationsState();
  const auto zero = b_.ConstantLiteral(xla::Literal::Zero(dtype_));
  operations_ = b_.Broadcast(zero, xla_shape_sizes(shape_));
  applyOps();
}

void XLATensorData::applyOps() {
  if (!operations_) {
    return;
  }
  auto computation = b_.Build().ValueOrDie();
  auto client = XlaGetClient();
  xla_data_ = client->ExecuteComputation(computation, operations_params_);
  resetOperationsState();
}

void XLATensorData::resetOperationsState() {
  operations_ = at::nullopt;
  operations_params_.clear();
}
