#ifdef WITH_XLA
#include "torch/csrc/jit/xla_code_impl.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
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
std::vector<NativeT> linearize_tensor(
    const at::Tensor& t,
    const size_t total_elements);

template <>
std::vector<float> linearize_tensor<float>(
    const at::Tensor& t,
    const size_t total_elements) {
  JIT_ASSERT(
      t.is_contiguous()); // the logic below works only for contiguous Tensors
  std::vector<float> values(total_elements);
  std::copy(t.data<float>(), t.data<float>() + total_elements, values.begin());
  return values;
}

template <>
std::vector<int64> linearize_tensor<int64>(
    const at::Tensor& t,
    const size_t total_elements) {
  JIT_ASSERT(
      t.is_contiguous()); // the logic below works only for contiguous Tensors
  std::vector<int64> values(total_elements);
  std::copy(
      t.data<int64_t>(), t.data<int64_t>() + total_elements, values.begin());
  return values;
}

template <class NativeT>
std::unique_ptr<xla::GlobalData> tensor_to_xla_impl(
    const at::Tensor& param_tensor,
    const xla::Shape& param_shape) {
  std::vector<int64> dimension_sizes;
  size_t total_elements = 1;
  for (const auto dimension_size : param_tensor.sizes()) {
    dimension_sizes.push_back(dimension_size);
    total_elements *= dimension_size;
  }
  xla::Array<NativeT> parameter_xla_array(dimension_sizes);
  parameter_xla_array.SetValues(
      linearize_tensor<NativeT>(param_tensor, total_elements));
  xla::Literal literal(param_shape);
  literal.PopulateFromArray(parameter_xla_array);
  return TransferParameterToServer(literal);
}

std::unique_ptr<xla::GlobalData> tensor_to_xla(
    const at::Tensor& param_tensor,
    const xla::Shape& param_shape) {
  switch (param_tensor.type().scalarType()) {
    case at::ScalarType::Float:
      return tensor_to_xla_impl<float>(param_tensor, param_shape);
    case at::ScalarType::Long:
      return tensor_to_xla_impl<int64>(param_tensor, param_shape);
  }
}

} // namespace

namespace torch {
namespace jit {

XlaCodeImpl::XlaCodeImpl(const std::shared_ptr<Graph>& graph) : graph_(graph) {
  RemoveExpands(graph_);
  EliminateDeadCode(graph_);
}

at::optional<at::Tensor> XlaCodeImpl::run(
    const std::vector<at::Tensor>& inputs) const {
  const auto parameter_shapes = captureInputShapes(inputs);
  if (!parameter_shapes.has_value()) {
    return at::nullopt;
  }
  auto compilation_result = buildXlaComputation(*parameter_shapes);
  if (!compilation_result.has_value()) {
    return at::nullopt;
  }
  const auto& computation = *compilation_result;
  std::vector<xla::GlobalData*> arguments;
  int parameter_index = 0;
  for (int parameter_index = 0; parameter_index < parameter_shapes->size();
       ++parameter_index) {
    CHECK_LT(parameter_index, inputs.size());
    const at::Tensor& param_tensor = inputs[parameter_index];
    auto data =
        tensor_to_xla(param_tensor, (*parameter_shapes)[parameter_index]);
    arguments.push_back(data.release());
  }
  auto result_literal = xla::ExecuteComputation(computation, arguments);
  const auto result_slice = result_literal->data<float>();
  std::vector<int64_t> dimensions;
  const auto& result_shape = result_literal->shape();
  for (const auto result_dimension : result_shape.dimensions()) {
    dimensions.push_back(result_dimension);
  }
  at::Tensor result_tensor = at::empty(at::CPU(at::kFloat), dimensions);
  std::copy(
      result_slice.begin(), result_slice.end(), result_tensor.data<float>());
  return result_tensor;
}

at::optional<std::vector<xla::Shape>> XlaCodeImpl::captureInputShapes(
    const std::vector<at::Tensor>& inputs) const {
  std::vector<xla::Shape> parameter_shapes;
  for (const auto& tensor : inputs) {
    const auto tensor_element_type =
        make_xla_primitive_type(tensor.type().scalarType());
    if (!tensor_element_type.has_value()) {
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

xla::XlaOp build_convolution(
    const Node* node,
    const xla::XlaOp& lhs,
    const xla::XlaOp& rhs,
    xla::XlaBuilder* b) {
  const auto stride_sym = Symbol::attr("stride");
  CHECK(node->hasAttribute(stride_sym));
  const auto window_strides = xla_i64_list(node->is(stride_sym));
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  return b->Conv(lhs, rhs, window_strides, xla::Padding::kValid);
}

xla::XlaOp build_convolution_bias(
    const Node* node,
    const xla::XlaOp& lhs,
    const xla::XlaOp& rhs,
    const xla::XlaOp& bias,
    xla::XlaBuilder* b) {
  const auto stride_sym = Symbol::attr("stride");
  CHECK(node->hasAttribute(stride_sym));
  const auto window_strides = xla_i64_list(node->is(stride_sym));
  const auto node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  const auto bias_size = xla_i64_list(tensor_sizes(node_outputs[0]));
  const auto bias_scalar = b->Reshape(bias, {});
  return b->Add(
      b->Conv(lhs, rhs, window_strides, xla::Padding::kValid),
      b->Broadcast(bias_scalar, bias_size));
}

xla::XlaOp build_addmm(
    const Node* node,
    const xla::XlaOp& bias,
    const xla::XlaOp& weights,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto& node_inputs = node->inputs();
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
  const auto kernel_size_sym = Symbol::attr("kernel_size");
  CHECK(node->hasAttribute(kernel_size_sym));
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  const auto kernel_size = xla_i64_list(node->is(kernel_size_sym));
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  const auto window_strides = window_dimensions;
  return b->ReduceWindow(
      input,
      b->ConstantLiteral(init_value),
      max_computation,
      window_dimensions,
      window_strides,
      xla::Padding::kValid);
}

Symbol make_attr(const std::string& name, const Node* node) {
  const auto sym = Symbol::attr(name);
  CHECK(node->hasAttribute(sym));
  return sym;
}

#define ATTR(name) make_attr(name, node)

bool avg_pool2d_supported(const Node* node) {
  const auto ceil_mode = node->i(ATTR("ceil_mode"));
  if (ceil_mode) {
    LOG(INFO) << "ceil_mode not supported for avg_pool2d yet";
    return false;
  }
  return true;
}

xla::XlaComputation CreateAddComputation() {
  xla::XlaBuilder reduction_builder("xla_add_computation");
  const auto x = reduction_builder.Parameter(
      0, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "x");
  const auto y = reduction_builder.Parameter(
      1, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "y");
  reduction_builder.Add(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

at::optional<xla::XlaOp> build_avg_pool2d(
    const Node* node,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  // Inspired from tf2xla.
  if (!avg_pool2d_supported(node)) {
    return at::nullopt;
  }
  const auto& kernel_size = xla_i64_list(node->is(ATTR("kernel_size")));
  const auto& stride = xla_i64_list(node->is(ATTR("stride")));
  const auto add_computation = CreateAddComputation();
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  std::vector<int64> window_dimensions;
  window_dimensions.resize(2, 1);
  window_dimensions.insert(
      window_dimensions.end(), kernel_size.begin(), kernel_size.end());
  std::vector<int64> window_strides;
  window_strides.resize(2, 1);
  window_strides.insert(window_strides.end(), stride.begin(), stride.end());
  const auto& padding = node->is(ATTR("padding"));
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
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto padded_input = b->Pad(input, xla_zero, padding_config);
  const auto sum = b->ReduceWindow(
      padded_input,
      xla_zero,
      add_computation,
      window_dimensions,
      window_strides,
      xla::Padding::kValid);
  const auto count_include_pad = node->i(ATTR("count_include_pad"));
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
    const auto& node_inputs = node->inputs();
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
        CreateAddComputation(),
        window_dimensions,
        window_strides,
        xla::Padding::kValid);
    return b->Div(sum, counts);
  }
}

#undef ATTR

at::optional<xla::XlaOp> build_log_softmax(
    const Node* node,
    const xla::XlaOp& logits,
    xla::XlaBuilder* b) {
  // Inspired from tf2xla.
  const auto dim_sym = Symbol::attr("dim");
  CHECK(node->hasAttribute(dim_sym));

  int64_t dim = node->i(dim_sym);
  if (dim != 0 && dim != 1) {
    LOG(INFO) << "log_softmax not supported for dim=" << node->i(dim_sym);
    return at::nullopt;
  }

  int batch_dim = 0;
  int class_dim = 1;

  if (dim == 0) {
    std::swap(batch_dim, class_dim);
  }

  const auto max_func = CreateMaxComputation();
  const auto min_value = xla::Literal::MinValue(xla::PrimitiveType::F32);
  const auto logits_max =
      b->Reduce(logits, b->ConstantLiteral(min_value), max_func, {class_dim});
  const auto shifted_logits = b->Sub(logits, logits_max, {batch_dim});
  const auto exp_shifted = b->Exp(shifted_logits);
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto reduce =
      b->Reduce(exp_shifted, xla_zero, CreateAddComputation(), {class_dim});
  return b->Sub(shifted_logits, b->Log(reduce), {batch_dim});
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
  const auto threshold_sym = Symbol::attr("threshold");
  CHECK(node->hasAttribute(threshold_sym));
  const auto& threshold_tensor = node->t(threshold_sym);
  CHECK_EQ(threshold_tensor.ndimension(), 0);
  const auto threshold_literal =
      xla::Literal::CreateR0<float>(one_elem_tensor_value(threshold_tensor));
  const auto threshold = b->ConstantLiteral(*threshold_literal);
  const auto value_sym = Symbol::attr("value");
  CHECK(node->hasAttribute(value_sym));
  const auto& value_tensor = node->t(value_sym);
  CHECK_EQ(value_tensor.ndimension(), 0);
  const auto value_literal =
      xla::Literal::CreateR0<float>(one_elem_tensor_value(value_tensor));
  const auto value = b->ConstantLiteral(*value_literal);
  const auto& node_inputs = node->inputs();
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
  const auto dim_sym = Symbol::attr("dim");
  CHECK(node->hasAttribute(dim_sym));
  const auto dim = node->i(dim_sym);
  std::vector<xla::XlaOp> reshaped_inputs;
  const auto& node_inputs = node->inputs();
  CHECK_EQ(inputs.size(), node_inputs.size());
  // Reshape inputs along the dim axis.
  for (size_t i = 0; i < node_inputs.size(); ++i) {
    auto reshaped_input_size = xla_i64_list(tensor_sizes(node_inputs[i]));
    reshaped_input_size.insert(reshaped_input_size.begin() + dim, 1);
    reshaped_inputs.push_back(b->Reshape(inputs[i], reshaped_input_size));
  }
  return b->ConcatInDim(reshaped_inputs, dim);
}

xla::XlaOp build_batch_norm(
    const Node* node,
    const xla::XlaOp& input,
    const xla::XlaOp& weight,
    const xla::XlaOp& bias,
    xla::XlaBuilder* b) {
  const auto eps_sym = Symbol::attr("eps");
  CHECK(node->hasAttribute(eps_sym));
  const auto eps = node->f(eps_sym);
  return b->GetTupleElement(
      b->BatchNormTraining(input, weight, bias, eps, 0), 0);
}

at::optional<const xla::XlaOp&> xla_op_for_input(
    const Node* node,
    const size_t input_index,
    const std::unordered_map<size_t, xla::XlaOp>& node_xla_ops,
    const std::unordered_set<size_t> undefined_inputs) {
  const auto& node_inputs = node->inputs();
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
  const auto& node_outputs = node->outputs();
  CHECK_EQ(node_outputs.size(), 1);
  return node_outputs[0]->unique();
}

} // namespace

#define XLA_OP(input_index) \
  xla_op_for_input(node, input_index, node_xla_ops, undefined_inputs)

at::optional<xla::XlaComputation> XlaCodeImpl::buildXlaComputation(
    const std::vector<xla::Shape>& parameter_shapes) const {
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
      case aten::_convolution: {
        if (node->inputs().size() != 3) {
          LOG(INFO) << "Unsupported convolution";
          return at::nullopt;
        }

        xla::XlaOp xla_output;
        if (XLA_OP(2).has_value()) { // bias exists
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
          if (!xla_op.has_value()) {
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
      case aten::batch_norm: {
        CHECK_EQ(node->inputs().size(), 5);
        xla::XlaOp xla_output =
            build_batch_norm(node, *XLA_OP(0), *XLA_OP(1), *XLA_OP(2), &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
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
  if (return_node->kind() != prim::Return ||
      return_node->inputs().size() != 1 ||
      return_node->input()->unique() != current_unique) {
    LOG(INFO) << "Unexpected end of graph";
    return at::nullopt;
  }
  return b.Build().ValueOrDie();
}

#undef XLA_OP

} // namespace jit
} // namespace torch

#endif // WITH_XLA
