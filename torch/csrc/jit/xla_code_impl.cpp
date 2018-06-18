#ifdef WITH_XLA
#include "torch/csrc/jit/xla_code_impl.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/remove_expands.h"

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
  if (tensor_dimensions.size() == 1 && tensor_dimensions.front() == 1) {
    return xla::ShapeUtil::MakeShapeWithLayout(type, {}, {});
  }
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
    default:
      LOG(INFO) << "Type not supported";
      return at::nullopt;
  }
}

std::unique_ptr<xla::GlobalData> tensor_to_xla(
    const at::Tensor& param_tensor,
    const xla::Shape& param_shape) {
  std::vector<int64> dimension_sizes;
  size_t total_elements = 1;
  for (const auto dimension_size : param_tensor.sizes()) {
    dimension_sizes.push_back(dimension_size);
    total_elements *= dimension_size;
  }
  if (total_elements == 1) {
    const auto scalar_literal =
        xla::Literal::CreateR0<float>(*param_tensor.data<float>());
    return TransferParameterToServer(*scalar_literal);
  }
  xla::Array<float> parameter_xla_array(dimension_sizes);
  std::vector<float> values_container(total_elements);
  std::copy(
      param_tensor.data<float>(),
      param_tensor.data<float>() + total_elements,
      values_container.begin());
  parameter_xla_array.SetValues(values_container);
  xla::Literal literal(param_shape);
  literal.PopulateFromArray(parameter_xla_array);
  return TransferParameterToServer(literal);
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

xla::XlaOp build_convolution(
    const Node* node,
    const xla::XlaOp& lhs,
    const xla::XlaOp& rhs,
    const xla::XlaOp& bias,
    xla::XlaBuilder* b) {
  const auto stride_sym = Symbol::attr("stride");
  CHECK(node->hasAttribute(stride_sym));
  const auto window_strides = xla_i64_list(node->is(stride_sym));
  return b->Add(b->Conv(lhs, rhs, window_strides, xla::Padding::kValid), bias);
}

xla::XlaOp build_addmm(
    const Node* node,
    const xla::XlaOp& bias,
    const xla::XlaOp& weights,
    const xla::XlaOp& input,
    xla::XlaBuilder* b) {
  const auto& node_inputs = node->inputs();
  xla::XlaOp dot = b->Dot(weights, input);
  const auto bias_type = node_inputs[0]->type()->cast<TensorType>();
  CHECK(bias_type);
  const std::vector<int64_t>& bias_size = bias_type->sizes();
  CHECK_EQ(bias_size.size(), 1);
  std::vector<int64> reshaped_bias_sizes;
  reshaped_bias_sizes.push_back(1);
  reshaped_bias_sizes.push_back(bias_size.front());
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

xla::XlaComputation CreateAddComputation() {
  xla::XlaBuilder reduction_builder("xla_add_computation");
  const auto x = reduction_builder.Parameter(
      0, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "x");
  const auto y = reduction_builder.Parameter(
      1, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}), "y");
  reduction_builder.Add(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

at::optional<xla::XlaOp> build_log_softmax(const Node* node,
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
  const auto logits_max = b->Reduce(
      logits, b->ConstantLiteral(min_value), max_func, {class_dim});
  const auto shifted_logits = b->Sub(logits, logits_max, {batch_dim});
  const auto exp_shifted = b->Exp(shifted_logits);
  const auto zero_literal = xla::Literal::CreateR0<float>(0);
  const auto xla_zero = b->ConstantLiteral(*zero_literal);
  const auto reduce =
        b->Reduce(exp_shifted, xla_zero,
                  CreateAddComputation(), {class_dim});
  return b->Sub(shifted_logits, b->Log(reduce), {batch_dim});
}

const xla::XlaOp& xla_op_for_input(
    const Node* node,
    const size_t input_index,
    const std::unordered_map<size_t, xla::XlaOp>& node_xla_ops) {
  const auto& node_inputs = node->inputs();
  const auto input = node_inputs.at(input_index);
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

#define XLA_OP(input_index) xla_op_for_input(node, input_index, node_xla_ops)

at::optional<xla::XlaComputation> XlaCodeImpl::buildXlaComputation(
    const std::vector<xla::Shape>& parameter_shapes) const {
  xla::XlaBuilder b("xla_computation");
  std::unordered_map<size_t, xla::XlaOp> node_xla_ops;
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
        xla::XlaOp xla_output = build_binary_op(node, XLA_OP(0), XLA_OP(1), &b);
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
        xla::XlaOp xla_output =
            build_convolution(node, XLA_OP(0), XLA_OP(1), XLA_OP(2), &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::t: {
        CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output = b.Transpose(XLA_OP(0), {1, 0});
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
            build_addmm(node, XLA_OP(0), XLA_OP(1), XLA_OP(2), &b);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::max_pool2d: {
        CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output = build_max_pool2d(node, XLA_OP(0), &b);
        current_unique = node->outputs()[0]->unique(); // ignore indices
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::relu: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto zero_literal = xla::Literal::CreateR0<float>(0);
        const auto xla_zero = b.ConstantLiteral(*zero_literal);
        xla::XlaOp xla_output = b.Max(XLA_OP(0), xla_zero);
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
        CHECK(it_ok.second);
        break;
      }
      case aten::log_softmax: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto xla_output_maybe = build_log_softmax(node, XLA_OP(0), &b);
        if (!xla_output_maybe) {
          return at::nullopt;
        }
        current_unique = output_id(node);
        const auto it_ok = node_xla_ops.emplace(current_unique, *xla_output_maybe);
        CHECK(it_ok.second);
        break;
      }
      default:
        LOG(INFO) << "Unsupported operator";
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
