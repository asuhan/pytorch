#ifdef WITH_XLA
#include "torch/csrc/jit/xla_code_impl.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/passes/constant_folding.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

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

} // namespace

namespace torch {
namespace jit {

xla::XlaComputationClient client_;

xla::XlaComputationClient* XlaGetClient() {
  return &client_;
}

XlaCodeImpl::XlaCodeImpl(const std::shared_ptr<Graph>& graph) : graph_(graph) {}

at::optional<std::vector<xla::Shape>> XlaCodeImpl::captureInputShapes(
    const std::vector<at::Tensor>& inputs) const {
  std::vector<xla::Shape> parameter_shapes;
  for (const auto& tensor : inputs) {
    const auto tensor_element_type =
        make_xla_primitive_type(tensor.type().scalarType());
    if (!tensor_element_type) {
      return at::nullopt;
    }
    parameter_shapes.push_back(
        make_xla_shape(tensor.sizes(), *tensor_element_type));
  }
  return parameter_shapes;
}

namespace {

xla::XlaOp build_binary_op(
    const Node* node,
    const xla::XlaOp& lhs,
    const xla::XlaOp& rhs,
    xla::XlaBuilder* b) {
  switch (node->kind()) {
    case aten::add: {
      return b->Add(lhs, rhs);
    }
    case aten::mul: {
      return b->Mul(lhs, rhs);
    }
    default:
      LOG(FATAL) << "Invalid binary operator kind: " << node->kind();
  }
}

std::vector<int64_t> tensor_sizes(const Value* tensor) {
  const auto tensor_type = tensor->type()->cast<TensorType>();
  CHECK(tensor_type);
  return tensor_type->sizes();
}

std::vector<std::pair<int64, int64>> make_padding(const Node* node) {
  std::vector<std::pair<int64, int64>> dims_padding;
  const auto padding = int_list_attr(node, attr::padding);
  for (const auto dim_padding : padding) {
    dims_padding.emplace_back(dim_padding, dim_padding);
  }
  return dims_padding;
}

xla::XlaOp build_convolution(
    const Node* node,
    const xla::XlaOp& lhs,
    const xla::XlaOp& rhs,
    xla::XlaBuilder* b) {
  const auto window_strides = xla_i64_list(int_list_attr(node, attr::stride));
  const auto dims_padding = make_padding(node);
  return b->ConvWithGeneralPadding(lhs, rhs, window_strides, dims_padding);
}

xla::XlaOp build_convolution_bias(
    const Node* node,
    const xla::XlaOp& lhs,
    const xla::XlaOp& rhs,
    const xla::XlaOp& bias,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  CHECK_GE(node_inputs.size(), size_t(4));
  const auto window_strides = xla_i64_list(int_list_attr(node, attr::stride));
  const auto bias_size = tensor_sizes(node_inputs[3]);
  const auto node_outputs = node->outputs();
  auto broadcast_sizes = xla_i64_list(tensor_sizes(node_outputs[0]));
  CHECK_EQ(broadcast_sizes.size(), 4);
  // Remove the channels dimension.
  broadcast_sizes.erase(broadcast_sizes.begin() + 1);
  // Make the bias match the output dimensions.
  const auto bias_broadcast =
      b->Transpose(b->Broadcast(bias, broadcast_sizes), {0, 3, 1, 2});
  const auto conv = build_convolution(node, lhs, rhs, b);
  return b->Add(conv, bias_broadcast);
}

std::vector<int64> xla_shape_sizes(const xla::Shape& shape) {
  std::vector<int64> shape_sizes(
      shape.dimensions().begin(), shape.dimensions().end());
  return shape_sizes;
}

xla::XlaOp build_thnn_conv2d_backward_input(
    const Node* node,
    const xla::XlaOp& grad,
    const xla::XlaOp& weight,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), 9);
  const auto padding_attr = int_list_attr(node, attr::padding);
  CHECK_EQ(padding_attr.size(), 2);
  // Adjust input size to account for specified padding.
  auto input_size = tensor_sizes(node_inputs[1]);
  for (int i = 0; i < 2; ++i) {
    input_size[2 + i] += 2 * padding_attr[i];
  }
  tensorflow::TensorShape input_shape(xla_i64_list(input_size));
  const auto filter = b->Transpose(weight, {2, 3, 1, 0});
  const auto filter_size = xla_shape_sizes(b->GetShape(filter).ValueOrDie());
  tensorflow::TensorShape filter_shape(filter_size);
  tensorflow::TensorShape out_backprop_shape(
      xla_i64_list(tensor_sizes(node_inputs[0])));
  const auto stride_attr = int_list_attr(node, attr::stride);
  std::vector<int> strides{1, 1};
  std::copy(
      stride_attr.begin(), stride_attr.end(), std::back_inserter(strides));
  tensorflow::ConvBackpropDimensions dims;
  constexpr int num_spatial_dims = 2;
  std::vector<int> dilations{1, 1, 1, 1};
  const auto status = ConvBackpropComputeDimensionsV2(
      "thnn_conv2d_backward",
      num_spatial_dims,
      input_shape,
      filter_shape,
      out_backprop_shape,
      dilations,
      strides,
      tensorflow::Padding::VALID,
      tensorflow::TensorFormat::FORMAT_NCHW,
      &dims);
  CHECK(status.ok()) << status.error_message();

  constexpr int batch_dim = 0;
  constexpr int feature_dim = 1;

  // The input gradients are computed by a convolution of the output
  // gradients and the filter, with some appropriate padding. See the
  // comment at the top of conv_grad_ops.h for details.

  xla::ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(batch_dim);
  dnums.set_output_batch_dimension(batch_dim);
  dnums.set_input_feature_dimension(feature_dim);
  dnums.set_output_feature_dimension(feature_dim);

  // TF filter shape is [ H, W, ..., inC, outC ]
  // Transpose the input and output features for computing the gradient.
  dnums.set_kernel_input_feature_dimension(num_spatial_dims + 1);
  dnums.set_kernel_output_feature_dimension(num_spatial_dims);

  std::vector<int64> kernel_spatial_dims(num_spatial_dims);
  std::vector<std::pair<int64, int64>> padding(num_spatial_dims);
  std::vector<int64> lhs_dilation(num_spatial_dims);
  std::vector<int64> rhs_dilation(num_spatial_dims);
  std::vector<int64> ones(num_spatial_dims, 1);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int64 dim = 2 + i;
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(i);
    dnums.add_output_spatial_dimensions(dim);

    kernel_spatial_dims[i] = i;
    padding[i] = {dims.spatial_dims[i].pad_before,
                  dims.spatial_dims[i].pad_after};
    lhs_dilation[i] = dims.spatial_dims[i].stride;
    rhs_dilation[i] = dilations[dim];
  }

  // Mirror the filter in the spatial dimensions.
  xla::XlaOp mirrored_weights = b->Rev(filter, kernel_spatial_dims);

  // We'll need to undo the initial input padding once on the input backprop
  // result since edges are constant and have to be discarded for the gradient.
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(-padding_attr[i]);
    dims->set_edge_padding_high(-padding_attr[i]);
  }

  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);

  // activation gradients
  //   = gradients (with padding and dilation) <conv> mirrored_weights
  return b->Pad(
      b->ConvGeneralDilated(
          grad,
          mirrored_weights,
          /*window_strides=*/ones,
          padding,
          lhs_dilation,
          rhs_dilation,
          dnums),
      xla_zero,
      padding_config);
}

xla::PaddingConfig make_padding_config(const std::vector<int64_t> padding) {
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(padding[i]);
    dims->set_edge_padding_high(padding[i]);
  }
  return padding_config;
}

xla::XlaOp build_thnn_conv2d_backward_weight(
    const Node* node,
    const xla::XlaOp& grad,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  constexpr int n_dim = 0;
  constexpr int c_dim = 1;
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), 9);
  const auto padding_attr = int_list_attr(node, attr::padding);
  CHECK_EQ(padding_attr.size(), 2);
  // Adjust input size to account for specified padding.
  auto input_size = tensor_sizes(node_inputs[1]);
  for (int i = 0; i < 2; ++i) {
    input_size[2 + i] += 2 * padding_attr[i];
  }
  tensorflow::TensorShape activations_shape(xla_i64_list(input_size));
  const auto filter_size = xla_i64_list(tensor_sizes(node_inputs[2]));
  std::vector<int64> filter_size_backward{
      filter_size[2], filter_size[3], filter_size[1], filter_size[0]};
  tensorflow::TensorShape filter_shape(filter_size_backward);
  tensorflow::TensorShape out_backprop_shape(
      xla_i64_list(tensor_sizes(node_inputs[0])));
  const auto stride_attr = int_list_attr(node, attr::stride);
  std::vector<int> strides{1, 1};
  std::copy(
      stride_attr.begin(), stride_attr.end(), std::back_inserter(strides));
  tensorflow::ConvBackpropDimensions dims;
  constexpr int num_spatial_dims = 2;
  std::vector<int> dilations{1, 1, 1, 1};
  const auto status = ConvBackpropComputeDimensionsV2(
      "thnn_conv2d_backward",
      num_spatial_dims,
      activations_shape,
      filter_shape,
      out_backprop_shape,
      dilations,
      strides,
      tensorflow::Padding::VALID,
      tensorflow::TensorFormat::FORMAT_NCHW,
      &dims);
  CHECK(status.ok()) << status.error_message();

  // The filter gradients are computed by a convolution of the input
  // activations and the output gradients, with some appropriate padding.
  // See the comment at the top of conv_grad_ops.h for details.

  xla::ConvolutionDimensionNumbers dnums;

  // The activations (inputs) form the LHS of the convolution.
  // Activations have shape: [batch, in_rows, in_cols, ..., in_depth]
  // For the gradient computation, we flip the roles of the batch and
  // feature dimensions.
  // Each spatial entry has size in_depth * batch

  // Swap n_dim and c_dim in the activations.
  dnums.set_input_batch_dimension(c_dim);
  dnums.set_input_feature_dimension(n_dim);

  // The gradients become the RHS of the convolution.
  // The gradients have shape [batch, out_rows, out_cols, ..., out_depth]
  // where the batch becomes the input feature for the convolution.
  dnums.set_kernel_input_feature_dimension(n_dim);
  dnums.set_kernel_output_feature_dimension(c_dim);

  std::vector<std::pair<int64, int64>> padding(num_spatial_dims);
  std::vector<int64> rhs_dilation(num_spatial_dims);
  std::vector<int64> window_strides(num_spatial_dims);
  std::vector<int64> ones(num_spatial_dims, 1);

  // Tensorflow filter shape is [ H, W, ..., inC, outC ].
  for (int i = 0; i < num_spatial_dims; ++i) {
    dnums.add_output_spatial_dimensions(i);
  }
  dnums.set_output_batch_dimension(num_spatial_dims);
  dnums.set_output_feature_dimension(num_spatial_dims + 1);

  for (int i = 0; i < num_spatial_dims; ++i) {
    int64 dim = 2 + i;
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(dim);

    // We will also need to pad the input with zeros such that after the
    // convolution, we get the right size for the filter.
    // The padded_in_rows should be such that when we convolve this with the
    // expanded_out_rows as a filter, we should get filter_rows back.
    //
    const int64 padded_in_size = dims.spatial_dims[i].expanded_output_size +
        (dims.spatial_dims[i].filter_size - 1) * dilations[dim];

    // However it can be smaller than input_rows: in this
    // case it means some of the inputs are not used.
    //
    // An example is to have input_cols = 3, filter_cols = 2 and stride = 2:
    //
    // INPUT =  [ A  B  C ]
    //
    // FILTER = [ x y ]
    //
    // and the output will only have one column: a = A * x + B * y
    //
    // and input "C" is not used at all.
    //
    // We apply negative padding in this case.
    const int64 pad_total = padded_in_size - dims.spatial_dims[i].input_size;

    // Pad the bottom/right side with the remaining space.
    const int64 pad_before = 0;

    padding[i] = {pad_before, pad_total - pad_before};
    rhs_dilation[i] = dims.spatial_dims[i].stride;
    window_strides[i] = dilations[dim];
  }

  // Redo the initial input padding.
  const auto padding_config = make_padding_config(padding_attr);

  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);

  const auto padded_input = b->Pad(input, xla_zero, padding_config);

  return b->ConvGeneralDilated(
      padded_input,
      grad,
      window_strides,
      padding,
      /*lhs_dilation=*/ones,
      rhs_dilation,
      dnums);
}

struct Conv2DGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

xla::XlaComputation CreateAddComputation() {
  xla::XlaBuilder reduction_builder("xla_add_computation");
  const auto x = reduction_builder.Parameter(
      0, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "x");
  const auto y = reduction_builder.Parameter(
      1, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "y");
  reduction_builder.Add(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

Conv2DGrads build_thnn_conv2d_backward(
    const Node* node,
    const xla::XlaOp& grad,
    const xla::XlaOp& input,
    const xla::XlaOp& weight,
    xla::XlaBuilder* b) {
  const auto grad_input =
      build_thnn_conv2d_backward_input(node, grad, weight, b);
  // TODO: support weight and bias gradients
  const auto grad_weight =
      build_thnn_conv2d_backward_weight(node, grad, input, b);
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto grad_bias =
      b->Reduce(grad, xla_zero, CreateAddComputation(), {0, 2, 3});
  return {grad_input, grad_weight, grad_bias};
}

xla::XlaComputation CreateMaxComputation() {
  xla::XlaBuilder reduction_builder("xla_max_computation");
  const auto x = reduction_builder.Parameter(
      0, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "x");
  const auto y = reduction_builder.Parameter(
      1, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "y");
  reduction_builder.Max(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

xla::XlaOp build_max_pool2d(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto max_computation = CreateMaxComputation();
  const auto init_value = xla::Literal::MinValue(xla::PrimitiveType::F32);
  const auto node_inputs = node->inputs();
  CHECK_GE(node_inputs.size(), size_t(4));
  const auto kernel_size = xla_i64_list(int_list_attr(node, attr::kernel_size));
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  const auto stride = int_list_attr(node, attr::stride);
  if (stride.empty()) {
    window_strides = window_dimensions;
  } else {
    window_strides.resize(2, 1);
    const auto stride_attr = xla_i64_list(stride);
    window_strides.insert(
        window_strides.end(), stride_attr.begin(), stride_attr.end());
  }
  const auto padding_config =
      make_padding_config(int_list_attr(node, attr::padding));
  const auto xla_init_value = b->ConstantLiteral(init_value);
  const auto padded_input = b->Pad(input, xla_init_value, padding_config);
  return b->ReduceWindow(
      padded_input,
      xla_init_value,
      max_computation,
      window_dimensions,
      window_strides,
      xla::Padding::kValid);
}

xla::XlaComputation CreateGeComputation() {
  xla::XlaBuilder reduction_builder("xla_ge_computation");
  const auto x = reduction_builder.Parameter(
      0, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "x");
  const auto y = reduction_builder.Parameter(
      1, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "y");
  reduction_builder.Ge(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

xla::XlaOp build_max_pool2d_backward(
    const Node* node,
    const xla::XlaOp& out_backprop,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto init_value = b->ConstantLiteral(*zero_literal);
  const auto select = CreateGeComputation();
  const auto scatter = CreateAddComputation();
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  const auto kernel_size = int_list_attr(node, attr::kernel_size);
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  const auto stride = int_list_attr(node, attr::stride);
  if (stride.empty()) {
    window_strides = window_dimensions;
  } else {
    window_strides.resize(2, 1);
    window_strides.insert(window_strides.end(), stride.begin(), stride.end());
  }
  const auto spatial_padding = make_padding(node);
  std::vector<std::pair<int64, int64>> window_padding;
  window_padding.resize(2);
  window_padding.insert(
      window_padding.end(), spatial_padding.begin(), spatial_padding.end());
  return b->SelectAndScatterWithGeneralPadding(
      input,
      select,
      window_dimensions,
      window_strides,
      window_padding,
      out_backprop,
      init_value,
      scatter);
}

bool avg_pool2d_supported(const Node* node) {
  const auto node_inputs = node->inputs();
  CHECK_GE(node_inputs.size(), size_t(6));
  const auto ceil_mode = int_attr(node, attr::ceil_mode);
  if (ceil_mode) {
    LOG(INFO) << "ceil_mode not supported for avg_pool2d yet";
    return false;
  }
  return true;
}

at::optional<xla::XlaOp> build_avg_pool2d(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  // Inspired from tf2xla.
  if (!avg_pool2d_supported(node)) {
    return at::nullopt;
  }
  const auto node_inputs = node->inputs();
  CHECK_GE(node_inputs.size(), size_t(6));
  const auto kernel_size = xla_i64_list(int_list_attr(node, attr::kernel_size));
  const auto stride = xla_i64_list(int_list_attr(node, attr::stride));
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  if (stride.empty()) {
    window_strides = window_dimensions;
  } else {
    window_strides.resize(2, 1);
    window_strides.insert(window_strides.end(), stride.begin(), stride.end());
  }
  const auto padding = int_list_attr(node, attr::padding);
  CHECK_EQ(padding.size(), 2);
  const auto padding_config = make_padding_config(padding);
  const auto add_computation = CreateAddComputation();
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto padded_input = b->Pad(input, xla_zero, padding_config);
  const auto sum = b->ReduceWindow(
      padded_input,
      xla_zero,
      add_computation,
      window_dimensions,
      window_strides,
      xla::Padding::kValid);
  const auto count_include_pad = int_attr(node, attr::count_include_pad);
  if (count_include_pad) {
    const auto kernel_elements = std::accumulate(
        kernel_size.begin(),
        kernel_size.end(),
        1,
        [](const int64 lhs, const int64 rhs) { return lhs * rhs; });
    const auto count_literal = xla::Literal::CreateR0<float>(kernel_elements);
    const auto count = b->ConstantLiteral(*count_literal);
    return b->Div(sum, count);
  } else {
    auto input_size = xla_i64_list(tensor_sizes(node_inputs[0]));
    // Build a matrix of all 1s, with the same width/height as the input.
    const auto one_literal = xla::Literal::CreateR0<float>(1);
    const auto ones =
        b->Broadcast(b->ConstantLiteral(*one_literal), input_size);
    // Pad it like the sum matrix.
    const auto padded_ones = b->Pad(ones, xla_zero, padding_config);
    const auto counts = b->ReduceWindow(
        padded_ones,
        xla_zero,
        add_computation,
        window_dimensions,
        window_strides,
        xla::Padding::kValid);
    return b->Div(sum, counts);
  }
}

xla::PaddingConfig::PaddingConfigDimension make_avg_pool2d_backprop_padding(
    const int64_t kernel_size,
    const int64_t input_size,
    const int64_t stride,
    const int64_t output_size,
    const int64_t input_padding) {
  xla::PaddingConfig::PaddingConfigDimension padding_config_dimension;
  const auto expanded_output_size = (output_size - 1) * stride + 1;
  const auto padded_out_size = input_size + 2 * input_padding + kernel_size - 1;
  const auto pad_before = kernel_size - 1;
  padding_config_dimension.set_edge_padding_low(pad_before);
  padding_config_dimension.set_edge_padding_high(
      padded_out_size - expanded_output_size - pad_before);
  padding_config_dimension.set_interior_padding(stride - 1);
  return padding_config_dimension;
}

at::optional<xla::XlaOp> build_avg_pool2d_backward(
    const Node* node,
    const xla::XlaOp& out_backprop,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  // Inspired from tf2xla.
  if (!avg_pool2d_supported(node)) {
    return at::nullopt;
  }
  const auto kernel_size = xla_i64_list(int_list_attr(node, attr::kernel_size));
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  const auto stride = xla_i64_list(int_list_attr(node, attr::stride));
  if (stride.empty()) {
    window_strides = window_dimensions;
  } else {
    CHECK_EQ(stride.size(), 2);
    window_strides.resize(2, 1);
    window_strides.insert(window_strides.end(), stride.begin(), stride.end());
  }
  const auto padding = int_list_attr(node, attr::padding);
  CHECK_EQ(padding.size(), 2);

  const auto node_inputs = node->inputs();
  auto output_size = tensor_sizes(node_inputs[0]);
  auto input_size = tensor_sizes(node_inputs[1]);

  const auto add_computation = CreateAddComputation();
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto count_include_pad = int_attr(node, attr::count_include_pad);
  std::vector<int64> one_strides(4, 1LL);

  xla::PaddingConfig remove_padding_config;
  for (int i = 0; i < 2; ++i) {
    remove_padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    auto dims = remove_padding_config.add_dimensions();
    dims->set_edge_padding_low(-padding[i]);
    dims->set_edge_padding_high(-padding[i]);
  }

  if (count_include_pad) {
    xla::PaddingConfig padding_config;
    for (int i = 0; i < 2; ++i) {
      padding_config.add_dimensions();
    }
    for (int i = 0; i < 2; ++i) {
      auto dims = padding_config.add_dimensions();
      *dims = make_avg_pool2d_backprop_padding(
          kernel_size[i],
          input_size[2 + i],
          window_strides[2 + i],
          output_size[2 + i],
          padding[i]);
    }
    const auto padded_out_backprop =
        b->Pad(out_backprop, xla_zero, padding_config);
    const auto sum = b->ReduceWindow(
        padded_out_backprop,
        xla_zero,
        add_computation,
        window_dimensions,
        one_strides,
        xla::Padding::kValid);
    const auto kernel_elements = std::accumulate(
        kernel_size.begin(),
        kernel_size.end(),
        1,
        [](const int64 lhs, const int64 rhs) { return lhs * rhs; });
    const auto count_literal = xla::Literal::CreateR0<float>(kernel_elements);
    const auto count = b->ConstantLiteral(*count_literal);
    const auto sum_removed_padding =
        b->Pad(sum, xla_zero, remove_padding_config);
    return b->Div(sum_removed_padding, count);
  } else {
    // Build a matrix of all 1s, with the same width/height as the input.
    const auto one_literal = xla::Literal::CreateR0<float>(1);
    const auto xla_one = b->ConstantLiteral(*one_literal);
    const auto ones = b->Broadcast(xla_one, xla_i64_list(input_size));
    // Pad it like the sum matrix.
    xla::PaddingConfig ones_padding_config;
    for (int i = 0; i < 2; ++i) {
      ones_padding_config.add_dimensions();
    }
    for (int i = 0; i < 2; ++i) {
      auto dims = ones_padding_config.add_dimensions();
      dims->set_edge_padding_low(padding[i]);
      dims->set_edge_padding_high(padding[i]);
    }
    const auto padded_ones = b->Pad(ones, xla_zero, ones_padding_config);
    const auto counts = b->ReduceWindow(
        padded_ones,
        xla_zero,
        add_computation,
        window_dimensions,
        window_strides,
        xla::Padding::kValid);
    const auto out_backprop_div = b->Div(out_backprop, counts);
    std::vector<int64> filter_dims(4);
    for (int i = 0; i < 2; ++i) {
      filter_dims[i] = kernel_size[i];
    }
    int64_t depth = output_size[1];
    filter_dims[2] = filter_dims[3] = depth;
    tensorflow::TensorShape filter_shape(filter_dims);
    tensorflow::TensorShape out_backprop_shape(
        xla_i64_list(tensor_sizes(node_inputs[0])));
    auto gradients_size = tensor_sizes(node_inputs[1]);
    for (int i = 0; i < 2; ++i) {
      gradients_size[2 + i] += 2 * padding[i];
    }
    tensorflow::TensorShape gradients_shape(xla_i64_list(gradients_size));
    // Reuse the logic from Conv2DBackpropInput to compute padding.
    tensorflow::ConvBackpropDimensions dims;
    std::vector<int> strides;
    std::copy(
        window_strides.begin(),
        window_strides.end(),
        std::back_inserter(strides));
    const auto status = ConvBackpropComputeDimensions(
        "avg_pool2d_backward",
        /*num_spatial_dims=*/2,
        gradients_shape,
        filter_shape,
        out_backprop_shape,
        strides,
        tensorflow::Padding::VALID,
        tensorflow::TensorFormat::FORMAT_NCHW,
        &dims);
    CHECK(status.ok()) << status.error_message();

    // Pad the gradients in the spatial dimensions. We use the same padding
    // as Conv2DBackpropInput.
    xla::PaddingConfig grad_padding_config = xla::MakeNoPaddingConfig(4);
    for (int i = 0; i < 2; ++i) {
      int dim = 2 + i;
      auto* padding = grad_padding_config.mutable_dimensions(dim);
      padding->set_edge_padding_low(dims.spatial_dims[i].pad_before);
      padding->set_edge_padding_high(dims.spatial_dims[i].pad_after);
      padding->set_interior_padding(dims.spatial_dims[i].stride - 1);
    }
    auto padded_gradients =
        b->Pad(out_backprop_div, xla_zero, grad_padding_config);

    // in_backprop = padded_gradients <conv> ones
    const auto in_backprop = b->ReduceWindow(
        padded_gradients,
        xla_zero,
        add_computation,
        window_dimensions,
        /* window_strides=*/one_strides,
        xla::Padding::kValid);
    return b->Pad(in_backprop, xla_zero, remove_padding_config);
  }
}

at::optional<xla::XlaOp> build_log_softmax(
    const Node* node,
    const xla::XlaOp& logits,
    xla::XlaBuilder* b) {
  // Inspired from tf2xla.
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), size_t(2));
  int64_t dim = int_attr(node, attr::dim);

  auto input_size = tensor_sizes(node_inputs[0]);

  std::vector<int64> broadcast_dimensions;
  for (size_t broadcast_dim = 0; broadcast_dim < input_size.size();
       ++broadcast_dim) {
    if (broadcast_dim == dim) {
      continue;
    }
    broadcast_dimensions.push_back(broadcast_dim);
  }

  const auto max_func = CreateMaxComputation();
  const auto min_value = xla::Literal::MinValue(xla::PrimitiveType::F32);
  const auto logits_max =
      b->Reduce(logits, b->ConstantLiteral(min_value), max_func, {dim});
  const auto shifted_logits = b->Sub(logits, logits_max, broadcast_dimensions);
  const auto exp_shifted = b->Exp(shifted_logits);
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto reduce =
      b->Reduce(exp_shifted, xla_zero, CreateAddComputation(), {dim});
  return b->Sub(shifted_logits, b->Log(reduce), broadcast_dimensions);
}

at::optional<xla::XlaOp> build_log_softmax_grad(
    const Node* node,
    const xla::XlaOp& grad_output,
    const xla::XlaOp& output,
    xla::XlaBuilder* b) {
  // Inspired from tf2xla.
  int64 dim = int_attr(node, attr::dim);

  const auto node_inputs = node->inputs();
  auto input_size = tensor_sizes(node_inputs[0]);
  std::vector<int64> broadcast_dimensions;
  for (size_t broadcast_dim = 0; broadcast_dim < input_size.size();
       ++broadcast_dim) {
    if (broadcast_dim == dim) {
      continue;
    }
    broadcast_dimensions.push_back(broadcast_dim);
  }

  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto sum =
      b->Reduce(grad_output, xla_zero, CreateAddComputation(), {dim});

  return b->Sub(grad_output, b->Mul(b->Exp(output), sum, broadcast_dimensions));
}

xla::XlaOp build_threshold(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  const auto threshold_literal =
      xla::Literal::CreateR0<float>(float_attr(node, attr::threshold));
  const auto threshold = b->ConstantLiteral(*threshold_literal);
  const auto value_literal =
      xla::Literal::CreateR0<float>(float_attr(node, attr::value));
  const auto value = b->ConstantLiteral(*value_literal);
  const auto input_sizes = tensor_sizes(node_inputs[0]);
  std::vector<int64> broadcast_sizes(input_sizes.begin(), input_sizes.end());
  return b->Select(
      b->Gt(input, threshold), input, b->Broadcast(value, broadcast_sizes));
}

at::optional<xla::XlaOp> build_view(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), 2);
  const auto input_sizes = tensor_sizes(node_inputs[0]);
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  std::vector<int64_t> output_sizes;
  if (node_outputs[0]->type()->cast<TensorType>()) {
    output_sizes = tensor_sizes(node_outputs[0]);
  } else {
    output_sizes = int_list_attr(node, attr::size);
  }
  const auto it = std::find_if(
      output_sizes.begin(), output_sizes.end(), [](const int64_t dim_size) {
        return dim_size < 0;
      });
  if (it != output_sizes.end()) {
    LOG(INFO) << "Cannot infer target size for aten::view";
    return at::nullopt;
  }
  return b->Reshape(input, xla_i64_list(output_sizes));
}

xla::XlaOp build_expand(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  CHECK_GE(node_inputs.size(), 1);
  auto input_sizes = tensor_sizes(node_inputs[0]);
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  const auto output_sizes = tensor_sizes(node_outputs[0]);
  // Adjust the rank of the input to match the rank of the output.
  CHECK_LE(input_sizes.size(), output_sizes.size());
  for (size_t i = 0; i < output_sizes.size() - input_sizes.size(); ++i) {
    input_sizes.insert(input_sizes.begin(), 1);
  }
  const auto implicit_reshape = b->Reshape(input, xla_i64_list(input_sizes));
  // Squeeze the trivial (of size 1) dimensions.
  std::vector<int64> non_singleton_dimensions;
  std::copy_if(
      input_sizes.begin(),
      input_sizes.end(),
      std::back_inserter(non_singleton_dimensions),
      [](const size_t dim_size) { return dim_size != 1; });
  const auto squeezed_input =
      b->Reshape(implicit_reshape, non_singleton_dimensions);
  // Broadcast the squeezed tensor, the additional dimensions are to the left.
  std::vector<int64> broadcast_sizes;
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (input_sizes[i] == 1) {
      broadcast_sizes.push_back(output_sizes[i]);
    }
  }
  const auto broadcast = b->Broadcast(squeezed_input, broadcast_sizes);
  // Bring the dimensions added by broadcast where the trivial dimensions were.
  std::vector<int64> reshape_permutation;
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (input_sizes[i] == 1) {
      reshape_permutation.push_back(i);
    }
  }
  for (size_t i = 0; i < input_sizes.size(); ++i) {
    if (input_sizes[i] != 1) {
      reshape_permutation.push_back(i);
    }
  }
  return b->Reshape(broadcast, reshape_permutation, xla_i64_list(output_sizes));
}

std::vector<const Value*> input_list_attr(const Node* parent, const size_t id) {
  const auto nodes = parent->owningGraph()->block()->nodes();
  std::vector<const Value*> result;
  for (const auto node : nodes) {
    if (node->kind() != prim::ListConstruct) {
      continue;
    }
    const auto node_outputs = node->outputs();
    CHECK_EQ(node_outputs.size(), size_t(1));
    const auto output = node_outputs[0];
    if (output->unique() != id) {
      continue;
    }
    const auto node_inputs = node->inputs();
    for (const auto input : node_inputs) {
      result.push_back(input);
    }
    return result;
  }
  CHECK(false) << "Constant with id " << id << " not found.";
}

class XlaNode {
 public:
  XlaNode(xla::XlaOp op) : op_(op) {}

  XlaNode(xla::XlaOp op, const std::vector<int64>& logical_shape)
      : op_(op), logical_shape_(logical_shape) {}

  xla::XlaOp op(xla::XlaBuilder* b) const {
    if (logical_shape_.empty()) {
      return op_;
    }
    return b->Reshape(op_, logical_shape_);
  }

  xla::XlaOp opNoPerm() const {
    return op_;
  }

  const std::vector<int64>& logicalShape() const {
    return logical_shape_;
  }

 private:
  xla::XlaOp op_;
  std::vector<int64> logical_shape_;
};

at::optional<xla::XlaOp> build_stack(
    const Node* node,
    const std::unordered_map<size_t, XlaNode>& node_xla_ops,
    const std::unordered_set<size_t>& undefined_inputs,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), size_t(2));
  const auto stack_inputs = input_list_attr(node, node_inputs[0]->unique());
  const auto dim = int_attr(node, attr::dim);
  std::vector<xla::XlaOp> reshaped_inputs;
  // Reshape inputs along the dim axis.
  for (size_t i = 0; i < stack_inputs.size(); ++i) {
    auto reshaped_input_size = xla_i64_list(tensor_sizes(stack_inputs[i]));
    reshaped_input_size.insert(reshaped_input_size.begin() + dim, 1);
    const auto stack_input = stack_inputs[i];
    if (undefined_inputs.find(stack_input->unique()) !=
        undefined_inputs.end()) {
      return at::nullopt;
    }
    const auto xla_op_it = node_xla_ops.find(stack_input->unique());
    CHECK(xla_op_it != node_xla_ops.end());
    reshaped_inputs.push_back(
        b->Reshape(xla_op_it->second.op(b), reshaped_input_size));
  }
  return b->ConcatInDim(reshaped_inputs, dim);
}

struct BatchNormOutput {
  xla::XlaOp output;
  xla::XlaOp save_mean; // batch_mean
  xla::XlaOp save_invstd_eps; // 1 / sqrt(batch_var + eps)
};

BatchNormOutput build_batch_norm(
    const Node* node,
    const xla::XlaOp& input,
    const xla::XlaOp& weight,
    const xla::XlaOp& bias,
    xla::XlaBuilder* b) {
  const auto epsf = float_attr(node, attr::eps);
  const auto eps_literal = xla::Literal::CreateR0<float>(epsf);
  const auto eps = b->ConstantLiteral(*eps_literal);
  const auto one_literal = xla::Literal::CreateR0<float>(1.0f);
  const auto one = b->ConstantLiteral(*one_literal);
  const auto half_literal = xla::Literal::CreateR0<float>(0.5f);
  const auto half = b->ConstantLiteral(*half_literal);

  auto outputs = b->BatchNormTraining(input, weight, bias, epsf, 1);
  auto output = b->GetTupleElement(outputs, 0);
  auto save_mean = b->GetTupleElement(outputs, 1);
  auto save_var = b->GetTupleElement(outputs, 2);
  auto save_invstd_eps = b->Div(one, b->Pow(b->Add(save_var, eps), half));
  return {output, save_mean, save_invstd_eps};
}

struct BatchNormGrads {
  xla::XlaOp grad_input;
  xla::XlaOp grad_weight;
  xla::XlaOp grad_bias;
};

BatchNormGrads build_batch_norm_backward(
    const Node* node,
    const xla::XlaOp& grad,
    const xla::XlaOp& input,
    const xla::XlaOp& weight,
    const xla::XlaOp& save_mean,
    const xla::XlaOp& save_invstd_eps,
    xla::XlaBuilder* b) {
  const auto epsf = float_attr(node, attr::eps);
  const auto eps_literal = xla::Literal::CreateR0<float>(epsf);
  const auto eps = b->ConstantLiteral(*eps_literal);
  const auto one_literal = xla::Literal::CreateR0<float>(1.0f);
  const auto one = b->ConstantLiteral(*one_literal);
  const auto two_literal = xla::Literal::CreateR0<float>(2.0f);
  const auto two = b->ConstantLiteral(*two_literal);
  const auto save_var = b->Sub(b->Pow(b->Div(one, save_invstd_eps), two), eps);
  const auto grads =
      b->BatchNormGrad(input, weight, save_mean, save_var, grad, epsf, 1);
  const auto grad_input = b->GetTupleElement(grads, 0);
  const auto grad_weight = b->GetTupleElement(grads, 1);
  const auto grad_bias = b->GetTupleElement(grads, 2);
  return {grad_input, grad_weight, grad_bias};
}

xla::XlaOp build_compare_op(
    const Node* node,
    const xla::XlaOp& operand,
    xla::XlaBuilder* b) {
  const float other = float_attr(node, attr::other);
  const auto other_literal = xla::Literal::CreateR0<float>(other);
  const auto xla_other = b->ConstantLiteral(*other_literal);
  xla::XlaOp pred;
  switch (node->kind()) {
    case aten::gt: {
      pred = b->Gt(operand, xla_other);
      break;
    }
    default:
      LOG(FATAL) << "Invalid binary operator kind: " << node->kind();
  }
  return b->ConvertElementType(pred, xla::PrimitiveType::S8);
}

at::optional<xla::XlaOp> build_type_as(
    const Node* node,
    const xla::XlaOp& operand,
    xla::XlaBuilder* b) {
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  const auto output_tensor_type = node_outputs[0]->type()->cast<TensorType>();
  CHECK(output_tensor_type);
  const auto target_type_maybe =
      make_xla_primitive_type(output_tensor_type->scalarType());
  if (!target_type_maybe) {
    return at::nullopt;
  }
  return b->ConvertElementType(operand, *target_type_maybe);
}

at::optional<xla::XlaOp> build_sum(
    const Node* node,
    const xla::XlaOp& operand,
    xla::XlaBuilder* b) {
  if (int_attr(node, attr::keepdim)) {
    LOG(INFO) << "Sum with keepdim set not supported yet";
    return at::nullopt;
  }
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  return b->Reduce(
      operand,
      xla_zero,
      CreateAddComputation(),
      xla_i64_list(int_list_attr(node, attr::dim)));
}

at::optional<xla::XlaOp> xla_op_for_input(
    const Node* node,
    const size_t input_index,
    const std::unordered_map<size_t, XlaNode>& node_xla_ops,
    const std::unordered_set<size_t> undefined_inputs,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  const auto input = node_inputs.at(input_index);

  // check if is prim::Undefined
  const auto undefined_it = undefined_inputs.find(input->unique());
  if (undefined_it != undefined_inputs.end()) {
    return at::nullopt;
  }

  // check in constructed xla ops
  const auto xla_op_it = node_xla_ops.find(input->unique());
  CHECK(xla_op_it != node_xla_ops.end());
  return xla_op_it->second.op(b);
}

size_t output_id(const Node* node) {
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  return node_outputs[0]->unique();
}

// Create an identity operation of `ret` to make it the root of the computation.
void activate_return_node(const xla::XlaOp& ret, xla::XlaBuilder* b) {
  b->GetTupleElement(b->Tuple({ret}), 0);
}

XlaNode to_rank1(
    const xla::XlaOp op,
    const std::vector<int64>& permutation_in,
    xla::XlaBuilder* b) {
  const auto op_shape = b->GetShape(op).ValueOrDie();
  if (!permutation_in.empty()) {
    CHECK_EQ(op_shape.dimensions_size(), permutation_in.size());
  }
  const auto op_elems = std::accumulate(
      op_shape.dimensions().begin(),
      op_shape.dimensions().end(),
      1,
      [](const int64 lhs, const int64 rhs) { return lhs * rhs; });
  auto permutation = permutation_in;
  if (permutation.empty()) {
    permutation.resize(op_shape.dimensions_size());
    std::iota(permutation.begin(), permutation.end(), 0);
  }
  std::vector<int64> logical_size(op_shape.dimensions_size());
  for (size_t i = 0; i < op_shape.dimensions_size(); ++i) {
    logical_size[i] = op_shape.dimensions(permutation[i]);
  }
  return {b->Reshape(op, permutation, {op_elems}), logical_size};
}

} // namespace

#define XLA_OP(input_index) \
  xla_op_for_input(node, input_index, node_xla_ops, undefined_inputs, &b)

at::optional<XlaComputationResult> XlaCodeImpl::buildXlaComputation(
    const std::vector<xla::Shape>& parameter_shapes,
    const std::vector<std::vector<int64>>& logical_parameter_shapes) const {
  xla::XlaBuilder b("xla_computation");
  std::unordered_map<size_t, XlaNode> node_xla_ops;
  std::unordered_set<size_t> undefined_inputs;
  std::unordered_set<size_t> all_zero_inputs;

  auto nodes = graph_->block()->nodes();
  const auto graph_inputs = graph_->inputs();
  CHECK_EQ(parameter_shapes.size(), logical_parameter_shapes.size());
  for (size_t parameter_number = 0, xla_parameter_number = 0;
       parameter_number < graph_inputs.size();
       ++parameter_number) {
    Value* graph_input = graph_inputs[parameter_number];
    if (parameter_shapes[parameter_number].element_type() ==
        xla::PrimitiveType::PRIMITIVE_TYPE_INVALID) {
      all_zero_inputs.emplace(graph_input->unique());
      continue;
    }
    auto parameter = b.Parameter(
        xla_parameter_number,
        parameter_shapes[parameter_number],
        "parameter_" + std::to_string(xla_parameter_number));
    if (!logical_parameter_shapes[parameter_number].empty()) {
      parameter =
          b.Reshape(parameter, logical_parameter_shapes[parameter_number]);
    }
    const auto it_ok = node_xla_ops.emplace(graph_input->unique(), parameter);
    CHECK(it_ok.second);
    ++xla_parameter_number;
  }
  for (auto node : nodes) {
    switch (node->kind()) {
      case aten::add:
      case aten::mul: {
        const auto node_inputs = node->inputs();
        if (node_inputs.size() < 2) {
          LOG(INFO) << "Unsupported arity";
          return at::nullopt;
        }
        xla::XlaOp xla_output;
        if (all_zero_inputs.find(node_inputs[0]->unique()) !=
            all_zero_inputs.end()) {
          CHECK(node->kind() == aten::add);
          xla_output = *XLA_OP(1);
        } else {
          xla_output = build_binary_op(node, *XLA_OP(0), *XLA_OP(1), &b);
        }
        const auto current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::gt: {
        if (node->inputs().size() != 2) {
          LOG(INFO) << "Unsupported arity";
          return at::nullopt;
        }
        xla::XlaOp xla_output = build_compare_op(node, *XLA_OP(0), &b);
        const auto current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::type_as: {
        CHECK_EQ(node->inputs().size(), 2);
        const auto xla_output_maybe = build_type_as(node, *XLA_OP(0), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::convolution:
      case aten::thnn_conv2d_forward: {
        if (node->inputs().size() < 3) {
          LOG(INFO) << "Unsupported convolution";
          return at::nullopt;
        }

        xla::XlaOp xla_output;
        if (XLA_OP(3)) { // bias exists
          xla_output = build_convolution_bias(
              node, *XLA_OP(0), *XLA_OP(1), *XLA_OP(3), &b);
        } else {
          xla_output = build_convolution(node, *XLA_OP(0), *XLA_OP(1), &b);
        }
        const auto xla_output_rank1 = to_rank1(xla_output, {}, &b);
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, xla_output_rank1);
        CHECK(it_ok.second);
        break;
      }
      case aten::thnn_conv2d_backward: {
        CHECK_EQ(node->inputs().size(), 9);
        const auto conv2d_grads = build_thnn_conv2d_backward(
            node, *XLA_OP(0), *XLA_OP(1), *XLA_OP(2), &b);
        const auto node_outputs = node->outputs();
        {
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[0]->unique(), conv2d_grads.grad_input);
          CHECK(it_ok.second);
        }
        {
          auto grad_weight = conv2d_grads.grad_weight;
          std::vector<int64> permutation{3, 2, 0, 1};
          const auto grad_weight_rank1 = to_rank1(grad_weight, permutation, &b);
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[1]->unique(), grad_weight_rank1);
          CHECK(it_ok.second);
        }
        {
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[2]->unique(), conv2d_grads.grad_bias);
          CHECK(it_ok.second);
        }
        break;
      }
      case aten::t: {
        CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output = b.Transpose(*XLA_OP(0), {1, 0});
        const auto current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::addmm: {
        if (node->inputs().size() < 3) {
          LOG(INFO) << "Unsupported linear layer";
          return at::nullopt;
        }
        xla::XlaOp xla_output =
            b.Add(b.Dot(*XLA_OP(1), *XLA_OP(2)), *XLA_OP(0));
        const auto current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::mm: {
        CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output = b.Dot(*XLA_OP(0), *XLA_OP(1));
        const auto current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::max_pool2d_with_indices: {
        CHECK_GE(node->inputs().size(), 1);
        CHECK_GE(node->outputs().size(), 1);
        xla::XlaOp xla_output = build_max_pool2d(node, *XLA_OP(0), &b);
        const auto node_outputs = node->outputs();
        CHECK_GE(node_outputs.size(), 1);
        const auto current_unique = node_outputs[0]->unique();
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::max_pool2d_with_indices_backward: {
        CHECK_EQ(node->inputs().size(), 8);
        xla::XlaOp xla_output =
            build_max_pool2d_backward(node, *XLA_OP(0), *XLA_OP(1), &b);
        const auto current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::avg_pool2d: {
        CHECK_GE(node->inputs().size(), 1);
        const auto xla_output_maybe = build_avg_pool2d(node, *XLA_OP(0), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::avg_pool2d_backward: {
        CHECK_GE(node->inputs().size(), 2);
        const auto xla_output_maybe =
            build_avg_pool2d_backward(node, *XLA_OP(0), *XLA_OP(1), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::relu: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto zero_literal = xla::Literal::CreateR0<float>(0);
        const auto xla_zero = b.ConstantLiteral(*zero_literal);
        xla::XlaOp xla_output = b.Max(*XLA_OP(0), xla_zero);
        const auto xla_output_rank1 = to_rank1(xla_output, {}, &b);
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, xla_output_rank1);
        CHECK(it_ok.second);
        break;
      }
      case aten::threshold: {
        CHECK_EQ(node->inputs().size(), 3);
        xla::XlaOp xla_output = build_threshold(node, *XLA_OP(0), &b);
        const auto current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::log_softmax: {
        CHECK_EQ(node->inputs().size(), size_t(2));
        const auto xla_output_maybe = build_log_softmax(node, *XLA_OP(0), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::log_softmax_backward_data: {
        CHECK_EQ(node->inputs().size(), 4);
        const auto xla_output_maybe =
            build_log_softmax_grad(node, *XLA_OP(0), *XLA_OP(1), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::view: {
        CHECK_EQ(node->inputs().size(), 2);
        const auto xla_output_maybe = build_view(node, *XLA_OP(0), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::expand: {
        CHECK_GE(node->inputs().size(), 1);
        xla::XlaOp xla_output = build_expand(node, *XLA_OP(0), &b);
        const auto current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::stack: {
        CHECK_EQ(node->inputs().size(), 2);
        const auto xla_output_maybe =
            build_stack(node, node_xla_ops, undefined_inputs, &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::thnn_batch_norm_forward:
      case aten::batch_norm: {
        CHECK_EQ(node->inputs().size(), 8);
        const auto outputs =
            build_batch_norm(node, *XLA_OP(0), *XLA_OP(1), *XLA_OP(2), &b);
        const auto node_outputs = node->outputs();
        {
          const auto it_ok =
              node_xla_ops.emplace(node_outputs[0]->unique(), outputs.output);
          CHECK(it_ok.second);
        }
        if (node->kind() == aten::batch_norm) {
          CHECK_EQ(node->outputs().size(), 1);
        }
        // aten::batch_norm only has 1 output
        // thnn_batch_norm_forward has output, save_mean, save_std
        if (node->kind() == aten::thnn_batch_norm_forward) {
          {
            const auto it_ok = node_xla_ops.emplace(
                node_outputs[1]->unique(), outputs.save_mean);
            CHECK(it_ok.second);
          }
          {
            const auto it_ok = node_xla_ops.emplace(
                node_outputs[2]->unique(), outputs.save_invstd_eps);
            CHECK(it_ok.second);
          }
        }
        break;
      }
      case aten::thnn_batch_norm_backward: {
        CHECK_EQ(node->inputs().size(), 10);
        auto grads = build_batch_norm_backward(
            node,
            *XLA_OP(0), // grad_output
            *XLA_OP(1), // input
            *XLA_OP(2), // weight
            *XLA_OP(7), // save_mean
            *XLA_OP(8), // save_std
            &b);
        const auto node_outputs = node->outputs();
        {
          const auto it_ok =
              node_xla_ops.emplace(node_outputs[0]->unique(), grads.grad_input);
          CHECK(it_ok.second);
        }
        {
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[1]->unique(), grads.grad_weight);
          CHECK(it_ok.second);
        }
        {
          const auto it_ok =
              node_xla_ops.emplace(node_outputs[2]->unique(), grads.grad_bias);
          CHECK(it_ok.second);
        }
        break;
      }
      case aten::sum: {
        CHECK_GE(node->inputs().size(), 1);
        const auto xla_output_maybe = build_sum(node, *XLA_OP(0), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        const auto current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case prim::Constant:
      case prim::ListConstruct: {
        break;
      }
      case prim::Undefined: {
        const auto current_unique = output_id(node);
        undefined_inputs.emplace(current_unique);
        break;
      }
      default:
        LOG(INFO) << "Unsupported operator: " << node->kind().toQualString();
        return at::nullopt;
    }
  }
  const auto return_node = graph_->return_node();
  const auto node_inputs = return_node->inputs();
  // TODO: tighten the id check for returned tuples.
  if (return_node->kind() != prim::Return || node_inputs.empty()) {
    LOG(INFO) << "Unexpected end of graph";
    return at::nullopt;
  }
  std::vector<std::vector<int64>> ret_logical_shapes;
  if (node_inputs.size() > 1) {
    std::vector<xla::XlaOp> returned_tuple;
    for (const auto return_input : node_inputs) {
      const auto it = node_xla_ops.find(return_input->unique());
      CHECK(it != node_xla_ops.end());
      returned_tuple.push_back(it->second.opNoPerm());
      ret_logical_shapes.push_back(it->second.logicalShape());
    }
    b.Tuple(returned_tuple);
  } else {
    const auto it = node_xla_ops.find(node_inputs[0]->unique());
    CHECK(it != node_xla_ops.end());
    const auto ret = it->second;
    // Ensure that the returned value is the root of the computation.
    activate_return_node(ret.opNoPerm(), &b);
    ret_logical_shapes.push_back(ret.logicalShape());
  }
  return XlaComputationResult{b.Build().ValueOrDie(), ret_logical_shapes};
}

#undef XLA_OP

} // namespace jit
} // namespace torch

#endif // WITH_XLA
