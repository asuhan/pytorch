#ifdef WITH_XLA
#include "torch/csrc/jit/xla_code_impl.h"
#include "tensorflow/compiler/xla/rpc/computation_client.h"

using int64 = long long;

namespace {

// Create an XLA shape out of tensor dimensions and element type.
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

// Convert tensor type to XLA primitive type.
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

XlaCodeImpl::XlaCodeImpl(const std::shared_ptr<Graph>& graph) : graph_(graph) {
  initStageEnd();
}

void XlaCodeImpl::initStageEnd() {
  for (auto node : graph_->block()->nodes()) {
    nodes_.push_back(node);
    if (node->kind() == prim::Load) {
      stage_end_.push_back(nodes_.size());
    }
  }
}

namespace {

// Return true iff the XLA backend supports the given sequence of nodes.
bool validate_graph_with_fusion(const std::vector<Node*>& nodes,
                                size_t start_node_index,
                                size_t end_node_index) {
  if (end_node_index - start_node_index != 3) {
    LOG(INFO) << "Unsupported graph";
    return false;
  }
  Node* store_node = nodes[start_node_index];
  if (store_node->kind() != prim::Store) {
    LOG(INFO) << "First node should be store";
    return false;
  }
  Node* fusion_node = nodes[start_node_index + 1];
  if (fusion_node->kind() != prim::FusionGroup) {
    LOG(INFO) << "Second node should be fusion";
    return false;
  }
  Node* load_node = nodes[start_node_index + 2];
  if (load_node->kind() != prim::Load) {
    LOG(INFO) << "Third node should be load";
    return false;
  }
  return true;
}

}  // namespace

// Compiles and runs an XLA computation.
bool XlaCodeImpl::runComputation(Stack& stack, size_t current_stage) const {
  const auto parameter_shapes = captureInputShapes(stack, current_stage);
  if (!parameter_shapes.has_value()) {
    return false;
  }
  size_t start_node_index = startNodeIndex(current_stage);
  size_t end_node_index = endNodeIndex(current_stage);
  if (!validate_graph_with_fusion(nodes_, start_node_index, end_node_index)) {
    return false;
  }
  auto fusion_subgraph = nodes_[start_node_index + 1]->g(attr::Subgraph);
  auto compilation_result =
      buildFusionGroupXlaComputation(fusion_subgraph.get());
  if (!compilation_result.has_value()) {
    return false;
  }
  // Holds XLA parameters indexed by their position in the input stack.
  std::unordered_map<size_t, xla::GlobalData*> arguments_map;
  int parameter_index = 0;
  Node* store_node = nodes_[start_node_index];
  const auto store_node_outputs = store_node->outputs();
  CHECK_EQ(store_node_outputs.size(), parameter_shapes->size());
  // Create the arguments map from the tensors on the stack.
  for (int parameter_index = 0; parameter_index < parameter_shapes->size();
       ++parameter_index) {
    at::Tensor param_tensor =
        peek(stack, parameter_index, parameter_shapes->size());
    // Populate XLA literal with values from the input tensor.
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
    // Transfer XLA literal to the server as a parameter of the computation.
    auto data = TransferParameterToServer(literal);
    const auto it_ok = arguments_map.emplace(
        store_node_outputs[parameter_index]->unique(), data.release());
    CHECK(it_ok.second);
  }
  const auto fusion_subgraph_inputs = nodes_[start_node_index + 1]->inputs();
  // Create the XLA parameter list in the order required by the fusion group.
  std::vector<xla::GlobalData*> arguments;
  for (const auto input : fusion_subgraph_inputs) {
    const auto argument_it = arguments_map.find(input->unique());
    CHECK(argument_it != arguments_map.end());
    arguments.push_back(argument_it->second);
  }
  auto result_literal = xla::ExecuteComputation(*compilation_result, arguments);
  // Retrieve the result of the XLA computation into the result tensor.
  const auto result_slice = result_literal->data<float>();
  std::vector<int64_t> dimensions;
  const auto& result_shape = result_literal->shape();
  for (const auto result_dimension : result_shape.dimensions()) {
    dimensions.push_back(result_dimension);
  }
  CHECK(!stack.empty());
  at::Tensor result_tensor = peek(stack, 0, 1).type().tensor();
  result_tensor.resize_(dimensions);
  std::copy(result_slice.begin(), result_slice.end(),
            result_tensor.data<float>());
  drop(stack, parameter_shapes->size());
  stack.push_back(std::move(result_tensor));
  return true;
}

size_t XlaCodeImpl::startNodeIndex(size_t current_stage) const {
  size_t start_node_index = 0;
  if (current_stage) {
    CHECK_LE(current_stage, stage_end_.size());
    start_node_index = stage_end_[current_stage - 1];
  }
  return start_node_index;
}

size_t XlaCodeImpl::endNodeIndex(size_t current_stage) const {
  CHECK_LT(current_stage, stage_end_.size());
  return stage_end_[current_stage];
}

at::optional<std::vector<xla::Shape>> XlaCodeImpl::captureInputShapes(
    Stack& stack, size_t current_stage) const {
  std::vector<xla::Shape> parameter_shapes;
  const auto node = nodes_[startNodeIndex(current_stage)];
  CHECK_EQ(node->kind(), prim::Store);
  size_t parameter_count = node->outputs().size();
  for (int parameter_number = 0; parameter_number < parameter_count;
       ++parameter_number) {
    at::Tensor tensor = peek(stack, parameter_number, parameter_count);
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

at::optional<xla::XlaComputation> XlaCodeImpl::buildFusionGroupXlaComputation(
    Graph* fusion_subgraph) {
  xla::XlaBuilder b("xla_fusion_compiler");
  auto parameter_shapes = captureInputShapes(fusion_subgraph);
  if (!parameter_shapes.has_value()) {
    return at::nullopt;
  }
  // node_xla_ops holds a map from unique ids of each node in the graph to the
  // XLA expression which computes it. Initialize it with the input parameters.
  auto node_xla_ops = bindInputs(fusion_subgraph, *parameter_shapes, &b);
  for (auto node : fusion_subgraph->block()->nodes()) {
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
        const auto it_ok = node_xla_ops.emplace(res->unique(), xla_output);
        CHECK(it_ok.second);
        break;
      }
      default:
        LOG(INFO) << "Unsupported operator";
        return at::nullopt;
    }
  }
  return b.Build().ValueOrDie();
}

at::optional<std::vector<xla::Shape>> XlaCodeImpl::captureInputShapes(
    Graph* fusion_subgraph) {
  auto fusion_inputs = fusion_subgraph->inputs();
  std::vector<xla::Shape> parameter_shapes;
  for (int parameter_number = 0; parameter_number < fusion_inputs.size();
       ++parameter_number) {
    Value* input_value = fusion_inputs[parameter_number];
    auto input_type = input_value->type();
    auto input_tensor_type = input_type->cast<TensorType>();
    if (!input_tensor_type) {
      LOG(INFO) << "Fusion input type: " << *input_type << " not supported";
    }
    auto dtype = make_xla_primitive_type(input_tensor_type->scalarType());
    if (!dtype.has_value()) {
      return at::nullopt;
    }
    auto parameter_shape = make_xla_shape(input_tensor_type->sizes(), *dtype);
    parameter_shapes.push_back(parameter_shape);
  }
  return parameter_shapes;
}

std::unordered_map<size_t, xla::XlaOp> XlaCodeImpl::bindInputs(
    Graph* fusion_subgraph, const std::vector<xla::Shape>& parameter_shapes,
    xla::XlaBuilder* b) {
  std::unordered_map<size_t, xla::XlaOp> node_xla_ops;
  auto fusion_inputs = fusion_subgraph->inputs();
  // Create XlaParameter's for the fusion group inputs.
  for (int parameter_number = 0; parameter_number < fusion_inputs.size();
       ++parameter_number) {
    const auto& parameter_shape = parameter_shapes[parameter_number];
    Value* input_value = fusion_inputs[parameter_number];
    const auto it_ok = node_xla_ops.emplace(
        input_value->unique(),
        b->Parameter(parameter_number, parameter_shape,
                     "parameter_" + std::to_string(parameter_number)));
    CHECK(it_ok.second);
  }
  return node_xla_ops;
}

xla::XlaOp XlaCodeImpl::buildBinaryXlaOp(const NodeKind kind,
                                         const xla::XlaOp& lhs,
                                         const xla::XlaOp& rhs,
                                         xla::XlaBuilder* b) {
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
