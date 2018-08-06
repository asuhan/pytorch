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
      auto weights = node->inputs()[1]->type()->expect<TensorType>();
      auto convNode = graph->create(aten::thnn_conv2d_forward, 3)
                          ->is_(attr::stride, node->is(attr::stride))
                          ->is_(attr::padding, node->is(attr::padding))
                          ->is_(
                              attr::kernel_size,
                              {weights->sizes()[2], weights->sizes()[3]});

      graph->insertNode(convNode);
      for (auto input : node->inputs()) {
        convNode->addInput(input);
      }
      convNode->outputs()[0]->setType(it->outputs()[0]->type());
      it->output()->replaceAllUsesWith(convNode->outputs()[0]);
      it.destroyCurrent();
    } else if (it->kind() == aten::batch_norm) {
      WithInsertPoint guard(*it);
      auto graph = block->owningGraph();
      auto node = *it;
      auto bnNode = graph->create(aten::thnn_batch_norm_forward, 3)
                        ->i_(attr::training, node->i(attr::training))
                        ->f_(attr::momentum, node->f(attr::momentum))
                        ->f_(attr::eps, node->f(attr::eps));

      graph->insertNode(bnNode);
      for (auto input : node->inputs()) {
        bnNode->addInput(input);
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
