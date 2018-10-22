#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"

namespace torch {
namespace jit {

static void UnwrapBufferedFunctions(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      UnwrapBufferedFunctions(sub);
    if (it->kind() == aten::convolution) {
      WithInsertPoint guard(*it);

      auto graph = block->owningGraph();
      auto node = *it;

      const auto weight = node->namedInput(attr::weight);
      const auto weight_type = weight->type()->expect<CompleteTensorType>();
      const auto& weight_size = weight_type->sizes();
      const auto kernel_size = graph->insertConstant(
          std::vector<int64_t>{weight_size[2], weight_size[3]});
      const auto stride =
          graph->insertConstant(int_list_attr(node, attr::stride));
      const auto padding =
          graph->insertConstant(int_list_attr(node, attr::padding));

      auto convNode = graph->create(aten::thnn_conv2d_forward, 3);

      graph->insertNode(convNode);
      convNode->addInput(node->namedInput(attr::input));
      convNode->addInput(weight);
      convNode->addInput(kernel_size);
      convNode->addInput(node->namedInput(attr::bias));
      convNode->addInput(stride);
      convNode->addInput(padding);

      convNode->outputs()[0]->setType(it->outputs()[0]->type());
      it->output()->replaceAllUsesWith(convNode->outputs()[0]);
      it.destroyCurrent();
    } else if (it->kind() == aten::batch_norm) {
      WithInsertPoint guard(*it);
      auto graph = block->owningGraph();
      auto node = *it;
      auto bnNode = graph->create(aten::thnn_batch_norm_forward, 3);

      graph->insertNode(bnNode);
      const auto node_inputs = node->inputs();
      JIT_ASSERT(node_inputs.size() == 9);
      for (size_t i = 0; i < node_inputs.size() - 1; ++i) {
        bnNode->addInput(node_inputs[i]);
      }
      bnNode->outputs()[0]->setType(it->outputs()[0]->type());
      it->output()->replaceAllUsesWith(bnNode->outputs()[0]);
      it.destroyCurrent();
    }
  }
}

void UnwrapBufferedFunctions(const std::shared_ptr<Graph>& graph) {
  UnwrapBufferedFunctions(graph->block());
}

} // namespace jit
} // namespace torch
