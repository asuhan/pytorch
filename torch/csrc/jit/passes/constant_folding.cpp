#include "torch/csrc/jit/passes/constant_folding.h"
#include "torch/csrc/jit/autodiff.h"

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

  at::optional<IntList> evalNumToTensor(const Value* value) const;

  at::optional<IntList> evalTensorToNum(const Value* value) const;

  at::optional<IntList> evalTensorToFromNum(const Value* value) const;

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
    case prim::NumToTensor: {
      maybe_constant_value = evalNumToTensor(value);
      break;
    }
    case prim::TensorToNum: {
      maybe_constant_value = evalTensorToNum(value);
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
  JIT_ASSERT(node_inputs.size() == 2);
  const auto input = node_inputs[0];
  const auto tensor_type = input->type()->cast<CompleteTensorType>();
  if (!tensor_type) {
    return c10::nullopt;
  }
  const auto tensor_sizes = tensor_type->sizes();
  const auto dim = int_attr(node, attr::dim);
  JIT_ASSERT(dim >= 0);
  JIT_ASSERT(static_cast<size_t>(dim) < tensor_sizes.size());
  return ConstantEvaluator::IntList{tensor_sizes[dim]};
}

at::optional<ConstantEvaluator::IntList> ConstantEvaluator::evalNumToTensor(
    const Value* value) const {
  return evalTensorToFromNum(value);
}

at::optional<ConstantEvaluator::IntList> ConstantEvaluator::evalTensorToNum(
    const Value* value) const {
  return evalTensorToFromNum(value);
}

at::optional<ConstantEvaluator::IntList> ConstantEvaluator::evalTensorToFromNum(
    const Value* value) const {
  const auto node = value->node();
  const auto node_inputs = node->inputs();
  JIT_ASSERT(node_inputs.size() == 1);
  const auto input = node_inputs[0];
  const auto maybe_constant_value = lookup(input);
  if (!maybe_constant_value || maybe_constant_value->size() != 1) {
    return c10::nullopt;
  }
  return *maybe_constant_value;
}

at::optional<ConstantEvaluator::IntList> ConstantEvaluator::lookup(
    const Value* value) const {
  const auto it = constant_values_.find(value->unique());
  if (it == constant_values_.end()) {
    return c10::nullopt;
  }
  return it->second;
}

// Uses the evaluated constants to replace inputs in the graph nodes. Only do
// it for prim::ListConstruct for now.
void ApplyConstantsToGraph(Graph* graph, const ConstantEvaluator& evaluator) {
  auto nodes = graph->block()->nodes();
  for (auto node : nodes) {
    if (node->kind() != prim::ListConstruct) {
      continue;
    }
    const auto list_inputs = node->inputs();
    if (list_inputs.size() != 2) {
      continue;
    }
    for (size_t elem_idx = 0; elem_idx < list_inputs.size(); ++elem_idx) {
      const auto maybe_element = evaluator.lookup(list_inputs[elem_idx]);
      if (!maybe_element) {
        continue;
      }
      const auto element = *maybe_element;
      if (element.size() != 1) {
        continue;
      }
      WithInsertPoint insert_point_guard(node);
      node->replaceInput(elem_idx, graph->insertConstant(element[elem_idx]));
    }
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
