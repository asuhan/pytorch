#include "torch/csrc/jit/passes/constant_folding.h"

namespace torch {
namespace jit {

namespace {

// Evaluates and keeps track of potentially constant values. Currently, the
// values can only be (potentially singleton) integer lists.
class ConstantEvaluator {
 public:
  using IntList = std::vector<int64_t>;

  // Evaluates the given value using the constants seen so far.
  void eval(const Value* value);

  // Returns the evaluated constant for the given value, if present.
  at::optional<IntList> lookup(const Value* value) const;

 private:
  at::optional<IntList> evalAtenSize(const Value* value) const;

  at::optional<IntList> evalAtenStack(const Value* value) const;

  at::optional<IntList> evalConstant(const Value* value) const;

  std::unordered_map<size_t, IntList> constant_values_;
};

void ConstantEvaluator::eval(const Value* value) {
  at::optional<ConstantEvaluator::IntList> maybe_constant_value;
  const auto node = value->node();
  switch (node->kind()) {
    case aten::size: {
      maybe_constant_value = evalAtenSize(value);
      break;
    }
    case aten::stack: {
      maybe_constant_value = evalAtenStack(value);
      break;
    }
    case prim::Constant: {
      maybe_constant_value = evalConstant(value);
      break;
    }
    default:
      break;
  }
  if (maybe_constant_value) {
    const auto it_ok =
        constant_values_.emplace(value->unique(), *maybe_constant_value);
    JIT_ASSERT(it_ok.second);
  }
}

at::optional<ConstantEvaluator::IntList> ConstantEvaluator::evalAtenSize(
    const Value* value) const {
  const auto node = value->node();
  const auto node_inputs = node->inputs();
  JIT_ASSERT(node_inputs.size() == 1);
  const auto input = node_inputs[0];
  const auto tensor_type = input->type()->cast<TensorType>();
  if (!tensor_type) {
    return at::nullopt;
  }
  const auto tensor_sizes = tensor_type->sizes();
  const auto dim = node->i(attr::dim);
  JIT_ASSERT(dim < tensor_sizes.size());
  return ConstantEvaluator::IntList{tensor_sizes[dim]};
}

at::optional<ConstantEvaluator::IntList> ConstantEvaluator::evalAtenStack(
    const Value* value) const {
  const auto node = value->node();
  const auto dim = node->i(attr::dim);
  if (dim != 0) {
    return at::nullopt;
  }
  ConstantEvaluator::IntList result;
  for (const auto input : node->inputs()) {
    const auto maybe_constant_value = lookup(input);
    if (!maybe_constant_value || maybe_constant_value->size() != 1) {
      return at::nullopt;
    }
    result.push_back(maybe_constant_value->front());
  }
  return result;
}

at::optional<ConstantEvaluator::IntList> ConstantEvaluator::lookup(
    const Value* value) const {
  const auto it = constant_values_.find(value->unique());
  if (it == constant_values_.end()) {
    return at::nullopt;
  }
  return it->second;
}

at::optional<ConstantEvaluator::IntList> ConstantEvaluator::evalConstant(
    const Value* value) const {
  const auto node = value->node();
  JIT_ASSERT(node->inputs().empty());
  const auto node_outputs = node->outputs();
  JIT_ASSERT(node_outputs.size() == 1);
  const auto tensor_type = node_outputs[0]->type()->cast<TensorType>();
  if (!tensor_type || tensor_type->scalarType() != at::ScalarType::Long ||
      !tensor_type->sizes().empty()) {
    return at::nullopt;
  }
  const auto& one_long_tensor = node->t(attr::value);
  JIT_ASSERT(one_long_tensor.ndimension() == 0);
  return ConstantEvaluator::IntList{*one_long_tensor.data<int64_t>()};
}

// Uses the evaluated constants to replace inputs in the graph nodes. Only do
// it for aten::view for now.
void ApplyConstantsToGraph(Graph* graph, const ConstantEvaluator& evaluator) {
  auto nodes = graph->block()->nodes();
  for (auto node : nodes) {
    if (node->kind() != aten::view) {
      continue;
    }
    const auto view_inputs = node->inputs();
    if (view_inputs.size() != 2) {
      continue;
    }
    const auto maybe_target_size = evaluator.lookup(view_inputs[1]);
    if (!maybe_target_size) {
      continue;
    }
    node->removeInput(1);
    JIT_ASSERT(!node->hasAttribute(attr::size));
    node->is_(attr::size, *maybe_target_size);
  }
}

} // namespace

void ConstantFold(std::shared_ptr<Graph>& graph) {
  ConstantEvaluator evaluator;
  auto nodes = graph->block()->nodes();
  for (auto node : nodes) {
    const auto node_outputs = node->outputs();
    if (node_outputs.size() != 1) {
      continue;
    }
    evaluator.eval(node_outputs[0]);
  }
  ApplyConstantsToGraph(graph.get(), evaluator);
}

} // namespace jit
} // namespace torch
