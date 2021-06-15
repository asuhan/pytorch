#include "lazy_tensor_core/csrc/ts_backend/aten_xla_type.h"

#include <ATen/native/BinaryOps.h>
#include <torch/library.h>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/function_call_tracker.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_autograd_ops_ts.h"
#include "lazy_tensor_core/csrc/ts_backend/aten_xla_type_default.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/nnc_computation_client.h"

namespace torch_lazy_tensors {
namespace {

std::pair<LazyTensor, LazyTensor> GetBinaryOperands(const at::Tensor& self,
                                                    const at::Tensor& other) {
  LazyTensor self_tensor;
  LazyTensor other_tensor;
  auto self_xtensor = bridge::TryGetLtcTensor(self);
  if (!self_xtensor) {
    other_tensor = bridge::GetLtcTensor(other);
    self_tensor = bridge::GetOrCreateLtcTensor(self, other_tensor.GetDevice());
  } else {
    self_tensor = *self_xtensor;
    other_tensor = bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice());
  }
  return std::pair<LazyTensor, LazyTensor>(self_tensor, other_tensor);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Tensor& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  std::pair<LazyTensor, LazyTensor> operands =
      GetBinaryOperands(self, UnwrapNumber(other, dtype));
  LazyTensor result = bin_op(operands.first, operands.second, dtype);
  return bridge::AtenFromLtcTensor(result);
}

template <typename B>
at::Tensor DoBinaryOp(const at::Tensor& self, const at::Scalar& other,
                      const B& bin_op) {
  at::ScalarType dtype = at::result_type(self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor result = bin_op(self_tensor, other, dtype);
  return bridge::AtenFromLtcTensor(result);
}

void CheckBinaryOpTypePromotion(const at::Tensor& out, const at::Tensor& self,
                                const at::Tensor& other) {
  at::ScalarType resultType = at::result_type(self, other);
  LTC_CHECK(at::canCast(/*from=*/resultType, /*to=*/out.scalar_type()));
}

void CheckBinaryOpTypePromotion(const at::Tensor& out, const at::Tensor& self,
                                const at::Scalar& other) {
  at::ScalarType resultType = at::result_type(self, other);
  LTC_CHECK(at::canCast(/*from=*/resultType, /*to=*/out.scalar_type()));
}

at::Tensor subtensor(const at::Tensor& tensor, int dim, int groups, int g) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  int64_t n = tensor.sizes()[dim] / groups;
  at::Tensor eager_tensor =
      tensor.to(lazy_tensors::NNCComputationClient::HardwareDeviceType());
  return bridge::CreateLtcTensor(
      eager_tensor.narrow(dim, n * g, n).contiguous(),
      bridge::GetLtcDevice(tensor));
}

}  // namespace

at::Tensor AtenXlaType::_softmax(const at::Tensor& self, int64_t dim,
                                 bool /* half_to_float */) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::softmax(bridge::GetLtcTensor(self), dim, c10::nullopt));
}

at::Tensor AtenXlaType::_softmax_backward_data(const at::Tensor& grad_output,
                                               const at::Tensor& output,
                                               int64_t dim,
                                               const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::ts_softmax_backward(
      bridge::GetLtcTensor(grad_output), bridge::GetLtcTensor(output), dim,
      bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::add(const at::Tensor& self, const at::Tensor& other,
                            const at::Scalar& alpha) {
  LTC_FN_COUNTER("xla::");
  at::native::alpha_check(at::result_type(self, other), alpha);
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::add(xself, xother, alpha, dtype);
                    });
}

at::Tensor AtenXlaType::alias(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return self;
}

at::Tensor AtenXlaType::as_strided(const at::Tensor& self, at::IntArrayRef size,
                                   at::IntArrayRef stride,
                                   c10::optional<int64_t> storage_offset) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return AtenXlaTypeDefault::as_strided(self, size, stride, storage_offset);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::as_strided(self_tensor, std::move(xsize), std::move(xstride),
                             Helpers::I64Optional(storage_offset)));
}

const at::Tensor& AtenXlaType::as_strided_(
    const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  auto xsize = Helpers::I64List(size);
  auto xstride = Helpers::I64List(stride);
  if (!ir::ops::AsStrided::StrideIsSupported(
          self_tensor.shape(), xsize, xstride, storage_offset.value_or(0))) {
    return AtenXlaTypeDefault::as_strided_(self, size, stride, storage_offset);
  }
  LazyTensor::as_strided_(self_tensor, std::move(xsize), std::move(xstride),
                          Helpers::I64Optional(storage_offset));
  return self;
}

at::Tensor AtenXlaType::bmm(const at::Tensor& self, const at::Tensor& mat2) {
  LTC_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat2)) {
    return AtenXlaTypeDefault::bmm(self, mat2);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::bmm(bridge::GetLtcTensor(self), bridge::GetLtcTensor(mat2)));
}

at::Tensor AtenXlaType::constant_pad_nd(const at::Tensor& self,
                                        at::IntArrayRef pad,
                                        const at::Scalar& value) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::constant_pad_nd(
      bridge::GetLtcTensor(self), Helpers::I64List(pad), value));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask) {
  if (groups > 1) {
    std::vector<at::Tensor> grad_input(groups);
    std::vector<at::Tensor> grad_weight(groups);
    std::vector<at::Tensor> grad_bias(groups);
    for (int g = 0; g < groups; ++g) {
      auto grad_output_g = subtensor(grad_output, 1, groups, g);
      auto input_g = subtensor(input, 1, groups, g);
      auto weight_g = subtensor(weight, 0, groups, g);
      auto x_result = convolution_backward_overrideable(
          grad_output_g, input_g, weight_g, stride, padding, dilation,
          transposed, output_padding, 1, output_mask);
      grad_input[g] = std::get<0>(x_result);
      grad_weight[g] = std::get<1>(x_result);
      grad_bias[g] = std::get<2>(x_result);
    }
    return {at::cat(grad_input, 1), at::cat(grad_weight, 0),
            grad_bias[0].defined() ? at::cat(grad_bias, 0) : grad_bias[0]};
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::convolution_backward_overrideable", 1);
  LTC_VLOG(3) << "LTC-TS convolution_backward_overrideable :"
              << " grad_output=" << grad_output.toString()
              << " input=" << input.toString()
              << " weight=" << weight.toString();
  const auto kernel_size = weight.sizes().slice(2);
  LTC_CHECK(kernel_size.size() == 2 || kernel_size.size() == 3);
  const at::DeviceType device_type =
      lazy_tensors::NNCComputationClient::HardwareDeviceType();
  if (transposed) {
    at::TensorOptions options = at::TensorOptions().device(device_type);
    auto&& x_result =
        kernel_size.size() == 2
            ? at::slow_conv_transpose2d_backward(
                  grad_output.to(device_type), input.to(device_type),
                  weight.to(device_type), kernel_size, stride, padding,
                  output_padding, dilation,
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Contiguous),
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Contiguous),
                  output_mask)
            : at::slow_conv_transpose3d_backward(
                  grad_output.to(device_type), input.to(device_type),
                  weight.to(device_type), kernel_size, stride, padding,
                  output_padding, dilation,
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Preserve),
                  at::empty_like(grad_output, options,
                                 at::MemoryFormat::Preserve),
                  output_mask);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        bridge::CreateLtcTensor(std::get<0>(x_result),
                                bridge::GetLtcDevice(grad_output)),
        bridge::CreateLtcTensor(std::get<1>(x_result),
                                bridge::GetLtcDevice(grad_output)),
        bridge::CreateLtcTensor(std::get<2>(x_result),
                                bridge::GetLtcDevice(grad_output)));
  }
  auto&& x_result =
      kernel_size.size() == 2
          ? at::slow_conv_dilated2d_backward(
                grad_output.to(device_type), input.to(device_type),
                weight.to(device_type), kernel_size, stride, padding, dilation,
                output_mask)
          : at::slow_conv_dilated3d_backward(
                grad_output.to(device_type), input.to(device_type),
                weight.to(device_type), kernel_size, stride, padding, dilation,
                output_mask);
  return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
      bridge::CreateLtcTensor(std::get<0>(x_result),
                              bridge::GetLtcDevice(grad_output)),
      bridge::CreateLtcTensor(std::get<1>(x_result),
                              bridge::GetLtcDevice(grad_output)),
      bridge::CreateLtcTensor(std::get<2>(x_result),
                              bridge::GetLtcDevice(grad_output)));
}

at::Tensor AtenXlaType::convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups) {
  if (groups != 1) {
    std::vector<at::Tensor> outputs(groups);
    for (int g = 0; g < groups; ++g) {
      auto input_g = subtensor(input, 1, groups, g);
      auto weight_g = subtensor(weight, 0, groups, g);
      auto bias_g = bias ? subtensor(*bias, 0, groups, g) : bias;
      outputs[g] =
          convolution_overrideable(input_g, weight_g, bias_g, stride, padding,
                                   dilation, transposed, output_padding, 1);
    }
    return at::cat(outputs, 1);
  }
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::convolution_overrideable", 1);
  LTC_VLOG(3) << "LTC-TS convolution_overrideable :"
              << " input=" << input.toString()
              << " weight=" << weight.toString();
  std::vector<at::Tensor> xlatens_tensors = {input, weight};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  std::vector<c10::optional<at::Tensor>> xlatens_opt_tensors = {bias};
  auto xlatens_opt = bridge::LtcCreateOptTensorList(xlatens_opt_tensors);
  const auto kernel_size = weight.sizes().slice(2);
  LTC_CHECK(kernel_size.size() == 2 || kernel_size.size() == 3);
  const at::DeviceType device_type =
      lazy_tensors::NNCComputationClient::HardwareDeviceType();
  if (transposed) {
    auto&& x_result =
        kernel_size.size() == 2
            ? at::slow_conv_transpose2d(
                  input.to(device_type), weight.to(device_type), kernel_size,
                  (bias && bias->defined()) ? bias->to(device_type) : bias,
                  stride, padding, output_padding, dilation)
            : at::slow_conv_transpose3d(
                  input.to(device_type), weight.to(device_type), kernel_size,
                  (bias && bias->defined()) ? bias->to(device_type) : bias,
                  stride, padding, output_padding, dilation);
    return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(input));
  }
  auto&& x_result =
      kernel_size.size() == 2
          ? at::slow_conv_dilated2d(
                input.to(device_type), weight.to(device_type), kernel_size,
                (bias && bias->defined()) ? bias->to(device_type) : bias,
                stride, padding, dilation)
          : at::slow_conv_dilated3d(
                input.to(device_type), weight.to(device_type), kernel_size,
                (bias && bias->defined()) ? bias->to(device_type) : bias,
                stride, padding, dilation);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(input));
}

at::Tensor AtenXlaType::_copy_from(const at::Tensor& self,
                                   const at::Tensor& dst, bool non_blocking) {
  LTC_FN_COUNTER("xla::");
  auto dst_tensor = bridge::TryGetLtcTensor(dst);
  auto self_tensor = bridge::TryGetLtcTensor(self);
  if (!self_tensor) {
    static bool sync_update =
        lazy_tensors::sys_util::GetEnvBool("XLA_TENSOR_UPDATE_SYNC", true);
    LTC_CHECK(dst_tensor);
    dst_tensor->UpdateFromTensor(self, /*sync=*/sync_update);
  } else if (!dst_tensor) {
    at::Tensor tensor = self_tensor->ToTensor(/*detached=*/true);
    at::Tensor typed_tensor =
        CopyTensor(tensor, dst.scalar_type(), /*copy=*/false);
    dst.resize_as_(typed_tensor).copy_(typed_tensor);
  } else {
    if (!dst_tensor->CurrentIrValue()) {
      auto dst_tensor_data = dst_tensor->CurrentTensorData();
      LTC_CHECK(dst_tensor_data);
      auto src_tensor_data = self_tensor->CurrentTensorData();
      if (src_tensor_data) {
        dst_tensor_data->copy_(*src_tensor_data);
      } else {
        dst_tensor_data->copy_(self_tensor->ToTensor(/*detached=*/true));
      }
    } else {
      LazyTensor::copy_(*dst_tensor, *self_tensor);
      bridge::ReplaceLtcTensor(dst, *dst_tensor);
    }
  }
  return dst;
}

at::Tensor AtenXlaType::div(const at::Tensor& self, const at::Tensor& other) {
  return div(self, other, /*rounding_mode=*/c10::nullopt);
}

at::Tensor AtenXlaType::div(const at::Tensor& self, const at::Tensor& other,
                            c10::optional<std::string> rounding_mode) {
  LTC_FN_COUNTER("xla::");
  at::ScalarType dtype = at::result_type(self, other);
  auto operands = GetBinaryOperands(self, other);
  return bridge::AtenFromLtcTensor(
      LazyTensor::div(operands.first, operands.second, rounding_mode, dtype));
}

at::Tensor AtenXlaType::div(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::div(bridge::GetLtcTensor(self), other));
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Tensor& other) {
  return div_(self, other, /*rounding_mode=*/c10::nullopt);
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Tensor& other,
                              c10::optional<std::string> rounding_mode) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::div_(self_tensor,
                   bridge::GetOrCreateLtcTensor(other, self_tensor.GetDevice()),
                   rounding_mode);
  return self;
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::div_(self_tensor, other);
  return self;
}

at::Tensor AtenXlaType::empty(at::IntArrayRef size,
                              c10::optional<at::ScalarType> dtype,
                              c10::optional<at::Layout> layout,
                              c10::optional<at::Device> device,
                              c10::optional<bool> pin_memory,
                              c10::optional<at::MemoryFormat> memory_format) {
  const auto device_type =
      lazy_tensors::NNCComputationClient::HardwareDeviceType();
  at::TensorOptions options = at::TensorOptions()
                                  .device(c10::Device(device_type))
                                  .layout(layout)
                                  .pinned_memory(pin_memory)
                                  .dtype(dtype);
  auto x_result = at::empty(size, options, memory_format);
  return bridge::CreateLtcTensor(x_result, bridge::GetLtcDevice(device));
}

at::Tensor AtenXlaType::empty_strided(at::IntArrayRef size,
                                      at::IntArrayRef stride,
                                      c10::optional<at::ScalarType> dtype,
                                      c10::optional<at::Layout> layout,
                                      c10::optional<at::Device> device,
                                      c10::optional<bool> pin_memory) {
  LTC_FN_COUNTER("xla::");
  at::Tensor t = empty(size, dtype, layout, device, pin_memory, c10::nullopt);
  return as_strided(t, size, stride, /*storage_offset=*/0);
}

at::Tensor AtenXlaType::eq(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::eq(bridge::GetLtcTensor(self), other));
}

at::Tensor AtenXlaType::eq(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::eq(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::eq_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::eq_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::eq_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::eq_(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor AtenXlaType::expand(const at::Tensor& self, at::IntArrayRef size,
                               bool implicit) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::expand(
      bridge::GetLtcTensor(self),
      lazy_tensors::util::ToVector<lazy_tensors::int64>(size)));
}

at::Tensor AtenXlaType::ge(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ge(bridge::GetLtcTensor(self), other));
}

at::Tensor AtenXlaType::ge(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ge(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::ge_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::ge_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::ge_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::ge_(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor AtenXlaType::gelu(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::gelu(bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::gelu_backward(const at::Tensor& grad,
                                      const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(LazyTensor::gelu_backward(
      bridge::GetLtcTensor(grad), bridge::GetLtcTensor(self)));
}

at::Tensor AtenXlaType::gt(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::gt(bridge::GetLtcTensor(self), other));
}

at::Tensor AtenXlaType::gt(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::gt(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::gt_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::gt_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::gt_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::gt_(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor AtenXlaType::le(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::le(bridge::GetLtcTensor(self), other));
}

at::Tensor AtenXlaType::le(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::le(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::le_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::le_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::le_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::le_(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor AtenXlaType::lt(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::lt(bridge::GetLtcTensor(self), other));
}

at::Tensor AtenXlaType::lt(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::lt(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::lt_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::lt_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::lt_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::lt_(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

at::Tensor AtenXlaType::max_pool2d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  return aten_autograd_ops_ts::MaxPool2dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::max_pool3d(const at::Tensor& self,
                                   at::IntArrayRef kernel_size,
                                   at::IntArrayRef stride,
                                   at::IntArrayRef padding,
                                   at::IntArrayRef dilation, bool ceil_mode) {
  return aten_autograd_ops_ts::MaxPool3dAutogradFunctionTS::apply(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::mm(const at::Tensor& self, const at::Tensor& mat2) {
  LTC_FN_COUNTER("xla::");
  // xla::dot doesn't support integer types.
  if (!at::native::is_floating_point(self) ||
      !at::native::is_floating_point(mat2)) {
    return AtenXlaTypeDefault::mm(self, mat2);
  }
  return bridge::AtenFromLtcTensor(
      LazyTensor::mm(/*input=*/bridge::GetLtcTensor(self),
                     /*weight=*/bridge::GetLtcTensor(mat2)));
}

at::Tensor AtenXlaType::mul(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const LazyTensor& xother,
                        at::ScalarType dtype) {
                      return LazyTensor::mul(xself, xother, dtype);
                    });
}

at::Tensor AtenXlaType::mul(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return DoBinaryOp(self, other,
                    [&](const LazyTensor& xself, const at::Scalar& other,
                        at::ScalarType dtype) {
                      return LazyTensor::mul(xself, other, dtype);
                    });
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::mul_(self_tensor, bridge::GetOrCreateLtcTensor(
                                    other, self_tensor.GetDevice()));
  return self;
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  CheckBinaryOpTypePromotion(self, self, other);
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::mul_(self_tensor, other);
  return self;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenXlaType::native_batch_norm(
    const at::Tensor& input, const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  LTC_FN_COUNTER("xla::");
  LazyTensor input_tensor = bridge::GetLtcTensor(input);
  const Device& device = input_tensor.GetDevice();
  LazyTensor running_mean_tensor =
      bridge::GetOrCreateLtcTensor(running_mean, device);
  LazyTensor running_var_tensor =
      bridge::GetOrCreateLtcTensor(running_var, device);
  auto outputs = LazyTensor::ts_native_batch_norm(
      bridge::GetLtcTensor(input), bridge::GetOrCreateLtcTensor(weight, device),
      bridge::GetOrCreateLtcTensor(bias, device), running_mean_tensor,
      running_var_tensor, training, momentum, eps);
  return std::make_tuple(bridge::AtenFromLtcTensor(std::get<0>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<1>(outputs)),
                         bridge::AtenFromLtcTensor(std::get<2>(outputs)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& running_mean,
    const c10::optional<at::Tensor>& running_var,
    const c10::optional<at::Tensor>& save_mean,
    const c10::optional<at::Tensor>& save_invstd, bool train, double eps,
    std::array<bool, 3> output_mask) {
  LTC_FN_COUNTER("xla::");
  LazyTensor grad_out_tensor = bridge::GetLtcTensor(grad_out);
  const Device& device = grad_out_tensor.GetDevice();
  auto gradients = LazyTensor::ts_native_batch_norm_backward(
      bridge::GetLtcTensor(grad_out), bridge::GetLtcTensor(input),
      bridge::GetOrCreateLtcTensor(weight, device),
      bridge::GetOrCreateLtcTensor(running_mean, device),
      bridge::GetOrCreateLtcTensor(running_var, device),
      bridge::GetOrCreateLtcTensor(save_mean, device),
      bridge::GetOrCreateLtcTensor(save_invstd, device), train, eps,
      output_mask);
  at::Tensor undefined;
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromLtcTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? bridge::AtenFromLtcTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? bridge::AtenFromLtcTensor(std::get<2>(gradients))
                     : undefined);
}

at::Tensor AtenXlaType::ne(const at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ne(bridge::GetLtcTensor(self), other));
}

at::Tensor AtenXlaType::ne(const at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::ne(bridge::GetLtcTensor(self), bridge::GetLtcTensor(other)));
}

at::Tensor& AtenXlaType::ne_(at::Tensor& self, const at::Scalar& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::ne_(self_tensor, other);
  return self;
}

at::Tensor& AtenXlaType::ne_(at::Tensor& self, const at::Tensor& other) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::ne_(self_tensor, bridge::GetLtcTensor(other));
  return self;
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return AtenXlaTypeDefault::max_pool2d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  return AtenXlaTypeDefault::max_pool2d_with_indices_backward(
      grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
      indices);
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool3d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  return AtenXlaTypeDefault::max_pool3d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

at::Tensor AtenXlaType::max_pool3d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
    const at::Tensor& indices) {
  return AtenXlaTypeDefault::max_pool3d_with_indices_backward(
      grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
      indices);
}

at::Tensor AtenXlaType::permute(const at::Tensor& self, at::IntArrayRef dims) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::permute(self_tensor, Helpers::I64List(dims)));
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  LTC_VLOG(3) << "XLA squeeze_ :"
              << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  xlatens[0].squeeze_();
  std::vector<size_t> xlatens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(xlatens_tensors, xlatens,
                                 xlatens_update_indices);
  } else {
    bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  }
  return self;
}

at::Tensor& AtenXlaType::squeeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_TRACK(3);
  LTC_COUNTER("aten::squeeze_", 1);
  LTC_VLOG(3) << "XLA squeeze_ :"
              << " self=" << self.toString();
  std::vector<at::Tensor> xlatens_tensors = {self};
  auto xlatens = bridge::LtcCreateTensorList(xlatens_tensors);
  xlatens[0].squeeze_(dim);
  std::vector<size_t> xlatens_update_indices = {0};
  if (bridge::IsInteropView(self)) {
    bridge::LtcUpdateTensorsMeta(xlatens_tensors, xlatens,
                                 xlatens_update_indices);
  } else {
    bridge::LtcUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);
  }
  return self;
}

at::Tensor AtenXlaType::t(const at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::transpose(bridge::GetLtcTensor(self), 0, 1));
}

at::Tensor& AtenXlaType::t_(at::Tensor& self) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::transpose_(self_tensor, 0, 1);
  return self;
}

at::Tensor AtenXlaType::transpose(const at::Tensor& self, int64_t dim0,
                                  int64_t dim1) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::transpose(bridge::GetLtcTensor(self), dim0, dim1));
}

at::Tensor& AtenXlaType::transpose_(at::Tensor& self, int64_t dim0,
                                    int64_t dim1) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::transpose_(self_tensor, dim0, dim1);
  return self;
}

at::Tensor AtenXlaType::unsqueeze(const at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  return bridge::AtenFromLtcTensor(
      LazyTensor::unsqueeze(bridge::GetLtcTensor(self), dim));
}

at::Tensor& AtenXlaType::unsqueeze_(at::Tensor& self, int64_t dim) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  LazyTensor::unsqueeze_(self_tensor, dim);
  return self;
}

at::Tensor AtenXlaType::view(const at::Tensor& self, at::IntArrayRef size) {
  LTC_FN_COUNTER("xla::");
  LazyTensor self_tensor = bridge::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::view(self_tensor, Helpers::I64List(size)));
}

void AtenXlaType::InitializeAtenBindings() {}

}  // namespace torch_lazy_tensors
