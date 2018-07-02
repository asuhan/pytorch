#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"

namespace torch {
namespace jit {

static void ConvolutionUnwrap(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      ConvolutionUnwrap(sub);
    if (it->kind() == aten::convolution) {
      WithInsertPoint guard(*it);

      auto graph = block->owningGraph();
      auto node = *it;

      const auto weight = node->namedInput(attr::weight);
      const auto weight_type = weight->type()->expect<TensorType>();
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
    }
  }
}

void ConvolutionUnwrap(const std::shared_ptr<Graph>& graph) {
  ConvolutionUnwrap(graph->block());
}

} // namespace jit
} // namespace torch
