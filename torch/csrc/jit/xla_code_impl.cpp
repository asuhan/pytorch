#ifdef WITH_XLA
#include "torch/csrc/jit/xla_code_impl.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"

using int64 = long long;

namespace {

xla::Shape make_xla_shape(const at::IntList& tensor_dimensions,
                          const xla::PrimitiveType type) {
  std::vector<int64> dimensions(tensor_dimensions.size());
  std::copy(tensor_dimensions.begin(), tensor_dimensions.end(),
            dimensions.begin());
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

}  // namespace

namespace torch {
namespace jit {

XlaCodeImpl::XlaCodeImpl(const std::shared_ptr<Graph>& graph) : graph_(graph) {}

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
    std::vector<int64> dimension_sizes;
    size_t total_elements = 1;
    for (const auto dimension_size : param_tensor.sizes()) {
      dimension_sizes.push_back(dimension_size);
      total_elements *= dimension_size;
    }
    xla::Array<float> parameter_xla_array(dimension_sizes);
    std::vector<float> values_container(total_elements);
    std::copy(param_tensor.data<float>(),
              param_tensor.data<float>() + total_elements,
              values_container.begin());
    parameter_xla_array.SetValues(values_container);
    xla::Literal literal((*parameter_shapes)[parameter_index]);
    literal.PopulateFromArray(parameter_xla_array);
    auto data = TransferParameterToServer(literal);
    arguments.push_back(data.release());
  }
  auto result_literal = xla::ExecuteComputation(computation, arguments);
  const auto result_slice = result_literal->data<float>();
  std::vector<int64_t> dimensions;
  const auto& result_shape = result_literal->shape();
  for (const auto result_dimension : result_shape.dimensions()) {
    dimensions.push_back(result_dimension);
  }
  at::Tensor result_tensor = inputs.front();
  result_tensor.resize_(dimensions);
  std::copy(result_slice.begin(), result_slice.end(),
            result_tensor.data<float>());
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
        b.Parameter(parameter_number, parameter_shapes[parameter_number],
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
        const Value* lhs = node->inputs()[0];
        const Value* rhs = node->inputs()[1];
        const auto lhs_it = node_xla_ops.find(lhs->unique());
        CHECK(lhs_it != node_xla_ops.end());
        const auto rhs_it = node_xla_ops.find(rhs->unique());
        CHECK(rhs_it != node_xla_ops.end());
        xla::XlaOp xla_output =
            buildBinaryXlaOp(node->kind(), lhs_it->second, rhs_it->second, &b);
        CHECK_EQ(node->outputs().size(), 1);
        const Value* res = node->outputs()[0];
        current_unique = res->unique();
        const auto it_ok = node_xla_ops.emplace(current_unique, xla_output);
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

xla::XlaOp XlaCodeImpl::buildBinaryXlaOp(const NodeKind kind,
                                         const xla::XlaOp& lhs,
                                         const xla::XlaOp& rhs,
                                         xla::XlaBuilder* b) const {
  switch (kind) {
    case aten::add: {
      return b->Add(lhs, rhs);
    }
    case aten::mul: {
      return b->Mul(lhs, rhs);
    }
    default:
      LOG(FATAL) << "Invalid binary operator kind: " << kind;
  }
}

}  // namespace jit
}  // namespace torch

#endif  // WITH_XLA
