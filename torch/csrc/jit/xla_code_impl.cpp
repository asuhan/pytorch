#ifdef WITH_XLA
#include "torch/csrc/jit/xla_code_impl.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "torch/csrc/jit/passes/constant_folding.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/remove_expands.h"

namespace {

using int64 = long long;

std::vector<int64> xla_i64_list(const std::vector<int64_t>& input) {
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
  }
}

} // namespace

namespace torch {
namespace jit {

XlaCodeImpl::XlaCodeImpl(const std::shared_ptr<Graph>& graph) : graph_(graph) {
  RemoveExpands(graph_);
  ConstantFold(graph_);
  EliminateDeadCode(graph_);
}

namespace {

at::Tensor make_tensor_from_xla_literal(const xla::Literal& literal) {
  const auto result_slice = literal.data<float>();
  std::vector<int64_t> dimensions;
  const auto& result_shape = literal.shape();
  for (const auto result_dimension : result_shape.dimensions()) {
    dimensions.push_back(result_dimension);
  }
  at::Tensor result_tensor = at::empty(at::CPU(at::kFloat), dimensions);
  std::copy(
      result_slice.begin(), result_slice.end(), result_tensor.data<float>());
  return result_tensor;
}

} // namespace

at::optional<std::vector<at::Tensor>> XlaCodeImpl::run(
    const std::vector<at::Tensor>& inputs) const {
  const auto parameter_shapes = captureInputShapes(inputs);
  if (!parameter_shapes) {
    return at::nullopt;
  }
  auto compilation_result = buildXlaComputation(*parameter_shapes);
  if (!compilation_result) {
    return at::nullopt;
  }
  const auto& computation = *compilation_result;
  std::vector<xla::GlobalData*> arguments;
  int parameter_index = 0;
  for (int parameter_index = 0; parameter_index < parameter_shapes->size();
       ++parameter_index) {
    CHECK_LT(parameter_index, inputs.size());
    const at::Tensor& param_tensor = inputs[parameter_index];
    auto data = tensor_to_xla(
        param_tensor, (*parameter_shapes)[parameter_index], &client_);
    arguments.push_back(data.release());
  }
  auto result_literal = client_.ExecuteComputation(computation, arguments);
  if (xla::ShapeUtil::IsTuple(result_literal->shape())) {
    const auto tuple_elements = result_literal->DecomposeTuple();
    std::vector<at::Tensor> result_tensors;
    for (const auto& tuple_element : tuple_elements) {
      result_tensors.push_back(make_tensor_from_xla_literal(tuple_element));
    }
    return result_tensors;
  }
  return std::vector<at::Tensor>{make_tensor_from_xla_literal(*result_literal)};
}

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
  const auto padding = node->is(attr::padding);
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
  const auto window_strides = xla_i64_list(node->is(attr::stride));
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
  CHECK_EQ(node_inputs.size(), 3); // input, weight, bias
  const auto bias_size = tensor_sizes(node_inputs[2]);
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

xla::XlaOp build_thnn_conv2d_backward_input(
    const Node* node,
    const xla::XlaOp& grad,
    const xla::XlaOp& input,
    const xla::XlaOp& weight,
    const xla::XlaOp& finput,
    const xla::XlaOp& fweight,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), 5);
  const auto& padding_attr = node->is(attr::padding);
  CHECK_EQ(padding_attr.size(), 2);
  // Adjust input size to account for specified padding.
  auto input_size = tensor_sizes(node_inputs[1]);
  for (int i = 0; i < 2; ++i) {
    input_size[2 + i] += 2 * padding_attr[i];
  }
  tensorflow::TensorShape input_shape(xla_i64_list(input_size));
  const auto filter_size = xla_i64_list(tensor_sizes(node_inputs[2]));
  std::vector<int64> filter_size_backward{
      filter_size[2], filter_size[3], filter_size[1], filter_size[0]};
  tensorflow::TensorShape filter_shape(filter_size_backward);
  tensorflow::TensorShape out_backprop_shape(
      xla_i64_list(tensor_sizes(node_inputs[0])));
  const auto stride_attr = node->is(attr::stride);
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
  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);

  std::vector<int64> kernel_spatial_dims(num_spatial_dims);
  std::vector<std::pair<int64, int64>> padding(num_spatial_dims);
  std::vector<int64> lhs_dilation(num_spatial_dims);
  std::vector<int64> rhs_dilation(num_spatial_dims);
  std::vector<int64> ones(num_spatial_dims, 1);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int64 dim = 2 + i;
    dnums.add_input_spatial_dimensions(dim);
    dnums.add_kernel_spatial_dimensions(dim);
    dnums.add_output_spatial_dimensions(dim);

    kernel_spatial_dims[i] = dim;
    padding[i] = {dims.spatial_dims[i].pad_before,
                  dims.spatial_dims[i].pad_after};
    lhs_dilation[i] = dims.spatial_dims[i].stride;
    rhs_dilation[i] = dilations[dim];
  }

  // Mirror the filter in the spatial dimensions.
  xla::XlaOp mirrored_weights = b->Rev(weight, kernel_spatial_dims);

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

xla::XlaOp build_thnn_conv2d_backward_weight(
    const Node* node,
    const xla::XlaOp& grad,
    const xla::XlaOp& input,
    const xla::XlaOp& weight,
    const xla::XlaOp& finput,
    const xla::XlaOp& fweight,
    xla::XlaBuilder* b) {
  constexpr int n_dim = 0;
  constexpr int c_dim = 1;
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), 5);
  const auto& padding_attr = node->is(attr::padding);
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
  const auto stride_attr = node->is(attr::stride);
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
    dnums.add_output_spatial_dimensions(2 + i);
  }
  dnums.set_output_batch_dimension(1);
  dnums.set_output_feature_dimension(0);

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
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(padding_attr[i]);
    dims->set_edge_padding_high(padding_attr[i]);
  }

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
    const xla::XlaOp& finput,
    const xla::XlaOp& fweight,
    xla::XlaBuilder* b) {
  const auto grad_input = build_thnn_conv2d_backward_input(
      node, grad, input, weight, finput, fweight, b);
  // TODO: support weight and bias gradients
  const auto grad_weight = build_thnn_conv2d_backward_weight(
      node, grad, input, weight, finput, fweight, b);
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto grad_bias = b->Reduce(grad, xla_zero,
				CreateAddComputation(),
				{0, 2, 3});
  return {grad_input, grad_weight, grad_bias};
}

xla::XlaOp build_addmm(
    const Node* node,
    const xla::XlaOp& bias,
    const xla::XlaOp& weights,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  const auto bias_size = tensor_sizes(node_inputs[0]);
  CHECK_EQ(bias_size.size(), 1);
  std::vector<int64> reshaped_bias_sizes;
  reshaped_bias_sizes.push_back(1);
  reshaped_bias_sizes.push_back(bias_size.front());
  xla::XlaOp dot = b->Dot(weights, input);
  return b->Add(dot, b->Reshape(bias, reshaped_bias_sizes));
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
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  const auto kernel_size = xla_i64_list(node->is(attr::kernel_size));
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  const auto stride = node->is(attr::stride);
  if (stride.empty()) {
    window_strides = window_dimensions;
  } else {
    window_strides.resize(2, 1);
    const auto stride_attr = xla_i64_list(stride);
    window_strides.insert(
        window_strides.end(), stride_attr.begin(), stride_attr.end());
  }
  const auto spatial_padding = make_padding(node);
  std::vector<std::pair<int64, int64>> window_padding;
  window_padding.resize(2);
  window_padding.insert(
      window_padding.end(), spatial_padding.begin(), spatial_padding.end());
  return b->ReduceWindowWithGeneralPadding(
      input,
      b->ConstantLiteral(init_value),
      max_computation,
      window_dimensions,
      window_strides,
      window_padding);
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
  const auto kernel_size = xla_i64_list(node->is(attr::kernel_size));
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  const auto stride = node->is(attr::stride);
  if (stride.empty()) {
    window_strides = window_dimensions;
  } else {
    window_strides.resize(2, 1);
    const auto stride_attr = xla_i64_list(stride);
    window_strides.insert(
        window_strides.end(), stride_attr.begin(), stride_attr.end());
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
  const auto ceil_mode = node->i(attr::ceil_mode);
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
  const auto kernel_size = xla_i64_list(node->is(attr::kernel_size));
  const auto stride = xla_i64_list(node->is(attr::stride));
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  window_strides.resize(2, 1);
  window_strides.insert(window_strides.end(), stride.begin(), stride.end());
  const auto& padding = node->is(attr::padding);
  CHECK_EQ(padding.size(), 2);
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(padding[i]);
    dims->set_edge_padding_high(padding[i]);
  }
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
  const auto count_include_pad = node->i(attr::count_include_pad);
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
    const auto node_inputs = node->inputs();
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
  const auto kernel_size = xla_i64_list(node->is(attr::kernel_size));
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  const auto stride = xla_i64_list(node->is(attr::stride));
  if (stride.empty()) {
    window_strides = window_dimensions;
  } else {
    CHECK_EQ(stride.size(), 2);
    window_strides.resize(2, 1);
    window_strides.insert(window_strides.end(), stride.begin(), stride.end());
  }
  const auto& padding = node->is(attr::padding);
  CHECK_EQ(padding.size(), 2);
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  const auto node_inputs = node->inputs();
  auto output_size = tensor_sizes(node_inputs[0]);
  auto input_size = tensor_sizes(node_inputs[1]);
  for (int i = 0; i < 2; ++i) {
    auto dims = padding_config.add_dimensions();
    *dims = make_avg_pool2d_backprop_padding(
        kernel_size[i],
        input_size[2 + i],
        window_strides[2 + i],
        output_size[2 + i],
        padding[i]);
  }
  const auto add_computation = CreateAddComputation();
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto padded_out_backprop =
      b->Pad(out_backprop, xla_zero, padding_config);
  std::vector<int64> one_strides(4, 1LL);
  const auto sum = b->ReduceWindow(
      padded_out_backprop,
      xla_zero,
      add_computation,
      window_dimensions,
      one_strides,
      xla::Padding::kValid);
  xla::PaddingConfig remove_padding_config;
  for (int i = 0; i < 2; ++i) {
    remove_padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    auto dims = remove_padding_config.add_dimensions();
    dims->set_edge_padding_low(-padding[i]);
    dims->set_edge_padding_high(-padding[i]);
  }
  const auto count_include_pad = node->i(attr::count_include_pad);
  if (!count_include_pad) {
    LOG(INFO)
        << "avg_pool2d_backward with count_include_pad=False not supported yet";
    return at::nullopt;
  }
  const auto kernel_elements = std::accumulate(
      kernel_size.begin(),
      kernel_size.end(),
      1,
      [](const int64 lhs, const int64 rhs) { return lhs * rhs; });
  const auto count_literal = xla::Literal::CreateR0<float>(kernel_elements);
  const auto count = b->ConstantLiteral(*count_literal);
  const auto sum_removed_padding = b->Pad(sum, xla_zero, remove_padding_config);
  return b->Div(sum_removed_padding, count);
}

at::optional<xla::XlaOp> build_log_softmax(
    const Node* node,
    const xla::XlaOp& logits,
    xla::XlaBuilder* b) {
  // Inspired from tf2xla.
  int64 dim = node->i(attr::dim);

  const auto node_inputs = node->inputs();
  auto input_size = tensor_sizes(node_inputs[0]);

  std::vector<int64> broadcast_dimensions;
  for (int64 broadcast_dim = 0; broadcast_dim < input_size.size();
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

double one_elem_tensor_value(const at::Tensor& t) {
  switch (t.type().scalarType()) {
    case at::ScalarType::Long:
      return *t.data<int64_t>();
    case at::ScalarType::Double:
      return *t.data<double>();
    default:
      LOG(FATAL) << "Type not supported";
  }
}

xla::XlaOp build_threshold(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto& threshold_tensor = node->t(attr::threshold);
  CHECK_EQ(threshold_tensor.ndimension(), 0);
  const auto threshold_literal =
      xla::Literal::CreateR0<float>(one_elem_tensor_value(threshold_tensor));
  const auto threshold = b->ConstantLiteral(*threshold_literal);
  const auto& value_tensor = node->t(attr::value);
  CHECK_EQ(value_tensor.ndimension(), 0);
  const auto value_literal =
      xla::Literal::CreateR0<float>(one_elem_tensor_value(value_tensor));
  const auto value = b->ConstantLiteral(*value_literal);
  const auto node_inputs = node->inputs();
  const auto input_sizes = tensor_sizes(node_inputs[0]);
  std::vector<int64> broadcast_sizes(input_sizes.begin(), input_sizes.end());
  std::copy(input_sizes.begin(), input_sizes.end(), broadcast_sizes.begin());
  return b->Select(
      b->Gt(input, threshold), input, b->Broadcast(value, broadcast_sizes));
}

xla::XlaOp build_view(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), 1);
  const auto input_sizes = tensor_sizes(node_inputs[0]);
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  const auto output_sizes = tensor_sizes(node_outputs[0]);
  return b->Reshape(input, xla_i64_list(output_sizes));
}

xla::XlaOp build_expand(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  CHECK_EQ(node_inputs.size(), 1);
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
    if (output_sizes[i] != input_sizes[i]) {
      CHECK_EQ(input_sizes[i], 1);
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

xla::XlaOp build_stack(
    const Node* node,
    const std::vector<xla::XlaOp>& inputs,
    xla::XlaBuilder* b) {
  const auto dim = node->i(attr::dim);
  std::vector<xla::XlaOp> reshaped_inputs;
  const auto node_inputs = node->inputs();
  CHECK_EQ(inputs.size(), node_inputs.size());
  // Reshape inputs along the dim axis.
  for (size_t i = 0; i < node_inputs.size(); ++i) {
    auto reshaped_input_size = xla_i64_list(tensor_sizes(node_inputs[i]));
    reshaped_input_size.insert(reshaped_input_size.begin() + dim, 1);
    reshaped_inputs.push_back(b->Reshape(inputs[i], reshaped_input_size));
  }
  return b->ConcatInDim(reshaped_inputs, dim);
}

struct BatchNormOutput {
  xla::XlaOp output;
  xla::XlaOp save_mean; // batch_mean
  xla::XlaOp save_invstd_eps;  // 1 / sqrt(batch_var + eps)
};


BatchNormOutput build_batch_norm(
    const Node* node,
    const xla::XlaOp& input,
    const xla::XlaOp& weight,
    const xla::XlaOp& bias,
    xla::XlaBuilder* b) {
  const auto epsf = node->f(attr::eps);
  const auto eps_literal = xla::Literal::CreateR0<float>(epsf);
  const auto eps = b->ConstantLiteral(*eps_literal);
  const auto one_literal = xla::Literal::CreateR0<float>(1.0f);
  const auto one = b->ConstantLiteral(*one_literal);
  const auto half_literal = xla::Literal::CreateR0<float>(0.5f);
  const auto half = b->ConstantLiteral(*half_literal);

  auto outputs = b->BatchNormTraining(input, weight, bias, epsf, 1);
  auto output = b->GetTupleElement(outputs, 2);
  auto save_mean = b->GetTupleElement(outputs, 1);
  auto save_var = b->GetTupleElement(outputs, 0);
  auto save_invstd_eps = b->Div(one, b->Pow(half, b->Add(save_var, eps)));
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
  const auto epsf = node->f(attr::eps);
  const auto eps_literal = xla::Literal::CreateR0<float>(epsf);
  const auto eps = b->ConstantLiteral(*eps_literal);
  const auto one_literal = xla::Literal::CreateR0<float>(1.0f);
  const auto one = b->ConstantLiteral(*one_literal);
  const auto two_literal = xla::Literal::CreateR0<float>(2.0f);
  const auto two = b->ConstantLiteral(*two_literal);
  const auto save_var = b->Sub(b->Pow(b->Div(one, save_invstd_eps), two), eps);
  const auto grads = b->BatchNormGrad(input, weight, save_mean, save_var, grad, epsf, 1);
  const auto grad_input = b->GetTupleElement(grads, 0);
  const auto grad_weight = b->GetTupleElement(grads, 1);
  const auto grad_bias = b->GetTupleElement(grads, 2);
  return {grad_input, grad_weight, grad_bias};
}

xla::XlaOp build_compare_op(
    const Node* node,
    const xla::XlaOp& operand,
    xla::XlaBuilder* b) {
  const float other = one_elem_tensor_value(node->t(attr::other));
  const auto other_literal = xla::Literal::CreateR0<float>(other);
  const auto xla_other = b->ConstantLiteral(*other_literal);
  const auto node_inputs = node->inputs();
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

at::optional<const xla::XlaOp&> xla_op_for_input(
    const Node* node,
    const size_t input_index,
    const std::unordered_map<size_t, xla::XlaOp>& node_xla_ops,
    const std::unordered_set<size_t> undefined_inputs) {
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
  return xla_op_it->second;
}

size_t output_id(const Node* node) {
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  return node_outputs[0]->unique();
}

bool graph_is_supported(const Graph* graph) {
  const auto nodes = graph->block()->nodes();
  // Index output of max_pool2d must not be used, not implemented yet.
  std::unordered_set<size_t> must_be_unused;
  for (const auto node : nodes) {
    if (node->kind() == aten::max_pool2d) {
      const auto node_outputs = node->outputs();
      must_be_unused.emplace(node_outputs[1]->unique());
    }
  }
  for (const auto node : nodes) {
    for (const auto input : node->inputs()) {
      const auto it = must_be_unused.find(input->unique());
      if (it != must_be_unused.end()) {
        LOG(INFO) << "Graph not supported; index output of max_pool2d is used.";
        return false;
      }
    }
  }
  return true;
}

} // namespace

#define XLA_OP(input_index) \
  xla_op_for_input(node, input_index, node_xla_ops, undefined_inputs)

at::optional<xla::XlaComputation> XlaCodeImpl::buildXlaComputation(
    const std::vector<xla::Shape>& parameter_shapes) const {
  if (!graph_is_supported(graph_.get())) {
    return at::nullopt;
  }

  xla::XlaBuilder b("xla_computation");
  std::unordered_map<size_t, xla::XlaOp> node_xla_ops;
  std::unordered_set<size_t> undefined_inputs;

  auto nodes = graph_->block()->nodes();
  const auto graph_inputs = graph_->inputs();
  for (size_t parameter_number = 0; parameter_number < graph_inputs.size();
       ++parameter_number) {
    Value* graph_input = graph_inputs[parameter_number];
    const auto it_ok = node_xla_ops.emplace(
        graph_input->unique(),
        b.Parameter(
            parameter_number,
            parameter_shapes[parameter_number],
            "parameter_" + std::to_string(parameter_number)));
    CHECK(it_ok.second);
  }
  size_t current_unique = 0;
  for (auto node : nodes) {
    switch (node->kind()) {
      case aten::add:
      case aten::mul: {
        if (node->inputs().size() != 2) {
          LOG(INFO) << "Unsupported arity";
          return at::nullopt;
        }
        xla::XlaOp xla_output =
            build_binary_op(node, *XLA_OP(0), *XLA_OP(1), &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::gt: {
        if (node->inputs().size() != 1) {
          LOG(INFO) << "Unsupported arity";
          return at::nullopt;
        }
        xla::XlaOp xla_output = build_compare_op(node, *XLA_OP(0), &b);
        current_unique = output_id(node);
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
        current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::convolution:
      case aten::thnn_conv2d_forward: {
        if (node->inputs().size() != 3) {
          LOG(INFO) << "Unsupported convolution";
          return at::nullopt;
        }

        xla::XlaOp xla_output;
        if (XLA_OP(2)) { // bias exists
          xla_output = build_convolution_bias(
              node, *XLA_OP(0), *XLA_OP(1), *XLA_OP(2), &b);
        } else {
          xla_output = build_convolution(node, *XLA_OP(0), *XLA_OP(1), &b);
        }
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::thnn_conv2d_backward: {
        CHECK_EQ(node->inputs().size(), 5);
        const auto conv2d_grads = build_thnn_conv2d_backward(
            node,
            *XLA_OP(0),
            *XLA_OP(1),
            *XLA_OP(2),
            *XLA_OP(3),
            *XLA_OP(4),
            &b);
        const auto node_outputs = node->outputs();
        {
          current_unique = node_outputs[0]->unique();
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[0]->unique(), conv2d_grads.grad_input);
          CHECK(it_ok.second);
        }
        {
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[1]->unique(), conv2d_grads.grad_weight);
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
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::addmm: {
        if (node->inputs().size() != 3) {
          LOG(INFO) << "Unsupported linear layer";
          return at::nullopt;
        }
        xla::XlaOp xla_output =
            build_addmm(node, *XLA_OP(0), *XLA_OP(1), *XLA_OP(2), &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::max_pool2d: {
        CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output = build_max_pool2d(node, *XLA_OP(0), &b);
        current_unique = node->outputs()[0]->unique(); // ignore indices
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::max_pool2d_backward: {
        CHECK_EQ(node->inputs().size(), 3);
        xla::XlaOp xla_output =
            build_max_pool2d_backward(node, *XLA_OP(0), *XLA_OP(1), &b);
        current_unique = node->outputs()[0]->unique(); // ignore indices
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::avg_pool2d: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto xla_output_maybe = build_avg_pool2d(node, *XLA_OP(0), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::avg_pool2d_backward: {
        CHECK_EQ(node->inputs().size(), 2);
        const auto xla_output_maybe =
            build_avg_pool2d_backward(node, *XLA_OP(0), *XLA_OP(1), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        current_unique = output_id(node);
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
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::threshold: {
        CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output = build_threshold(node, *XLA_OP(0), &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::log_softmax: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto xla_output_maybe = build_log_softmax(node, *XLA_OP(0), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        current_unique = output_id(node);
        const auto it_ok =
            node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      case aten::view: {
        CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output = build_view(node, *XLA_OP(0), &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::expand: {
        CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output = build_expand(node, *XLA_OP(0), &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::stack: {
        CHECK_GE(node->inputs().size(), 1);
        std::vector<xla::XlaOp> xla_ops;
        for (size_t i = 0; i < node->inputs().size(); ++i) {
          const auto xla_op = XLA_OP(i);
          if (!xla_op) {
            return at::nullopt;
          }
          xla_ops.push_back(*xla_op);
        }
        xla::XlaOp xla_output = build_stack(node, xla_ops, &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::thnn_batch_norm_forward:
      case aten::batch_norm: {
        CHECK_EQ(node->inputs().size(), 5);
	const auto outputs =
	  build_batch_norm(node, *XLA_OP(0), *XLA_OP(1), *XLA_OP(2), &b);
        const auto node_outputs = node->outputs();
        {
          current_unique = node_outputs[0]->unique();
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[0]->unique(), outputs.output);
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
        CHECK_EQ(node->inputs().size(), 7);
	auto grads = build_batch_norm_backward(
            node,
            *XLA_OP(0), // grad_output
            *XLA_OP(1), // input
            *XLA_OP(2), // weight
            *XLA_OP(5), // save_mean
            *XLA_OP(6), // save_std
            &b);
        const auto node_outputs = node->outputs();
        {
          current_unique = node_outputs[0]->unique();
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[0]->unique(), grads.grad_input);
          CHECK(it_ok.second);
        }
        {
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[1]->unique(), grads.grad_weight);
          CHECK(it_ok.second);
        }
        {
          const auto it_ok = node_xla_ops.emplace(
              node_outputs[2]->unique(), grads.grad_bias);
          CHECK(it_ok.second);
        }
        break;
      }
      case prim::Undefined: {
        current_unique = output_id(node);
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
  if (return_node->kind() != prim::Return || node_inputs.empty() ||
      node_inputs[0]->unique() != current_unique) {
    LOG(INFO) << "Unexpected end of graph";
    return at::nullopt;
  }
  if (node_inputs.size() > 1) {
    std::vector<xla::XlaOp> returned_tuple;
    for (const auto return_input : node_inputs) {
      const auto it = node_xla_ops.find(return_input->unique());
      CHECK(it != node_xla_ops.end());
      returned_tuple.push_back(it->second);
    }
    b.Tuple(returned_tuple);
  }
  return b.Build().ValueOrDie();
}

#undef XLA_OP

} // namespace jit
} // namespace torch

#endif // WITH_XLA
