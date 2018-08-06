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

XLATensor::XLATensor(const autograd::Variable& tensor) : grad_(nullptr) {
  auto client_ = XlaGetClient();
  dtype_ = *make_xla_primitive_type(tensor.type().scalarType());
  shape_ = make_xla_shape(tensor.sizes(), dtype_);
  data_ = tensor_to_xla(tensor, shape_, client_);
  requires_grad_ = tensor.requires_grad();
}

XLATensor::XLATensor(const xla::Literal& literal) : grad_(nullptr) {
  auto client_ = XlaGetClient();
  data_ = client_->TransferParameterToServer(literal);
  shape_ = literal.shape();
  dtype_ = shape_.element_type();
  requires_grad_ = false;
}

std::shared_ptr<XLATensor> XLATensor::grad() const {
  return grad_;
}

void XLATensor::setGrad(std::shared_ptr<XLATensor> grad) {
  grad_ = grad;
}

xla::Shape XLATensor::shape() const {
  return shape_;
}

xla::GlobalData* XLATensor::data() const {
  return data_.get();
}

at::Tensor XLATensor::toTensor() {
  // because there's no transferToClient, we'll define an `identity` graph, and
  // execute it
  xla::XlaBuilder b("identity");
  b.GetTupleElement(b.Tuple({b.Parameter(0, shape_, "x")}), 0);
  xla::XlaComputation identity = b.Build().ValueOrDie();

  auto client_ = XlaGetClient();
  auto result_literal =
      client_->ExecuteComputationAndTransfer(identity, {data_.get()});
  auto return_tensor = make_tensor_from_xla_literal(*result_literal);
  return autograd::make_variable(return_tensor, requires_grad_);
}

namespace {

// TODO(asuhan): de-dup with the version in xla_code_impl
std::vector<int64> xla_shape_sizes(const xla::Shape& shape) {
  std::vector<int64> shape_sizes(
      shape.dimensions().begin(), shape.dimensions().end());
  return shape_sizes;
}

} // namespace

void XLATensor::add_(const XLATensor& other, const at::Scalar& alpha) {
  xla::XlaBuilder b("XLATensor::add_");
  const auto alpha_literal = xla::Literal::CreateR0<float>(alpha.toDouble());
  const auto alpha_xla = b.ConstantLiteral(*alpha_literal);
  b.Add(
      b.Parameter(0, shape_, "self"),
      b.Mul(
          b.Parameter(1, shape_, "other"),
          b.Broadcast(alpha_xla, xla_shape_sizes(shape_))));
  auto add_computation = b.Build().ValueOrDie();
  auto client = XlaGetClient();
  data_ = client->ExecuteComputation(
      add_computation, {data_.get(), other.data_.get()});
}

void XLATensor::mul_(const XLATensor& other) {
  xla::XlaBuilder b("XLATensor::mul_");
  b.Mul(b.Parameter(0, shape_, "self"), b.Parameter(1, shape_, "self"));
  auto mul_computation = b.Build().ValueOrDie();
  auto client = XlaGetClient();
  data_ = client->ExecuteComputation(
      mul_computation, {data_.get(), other.data_.get()});
}
