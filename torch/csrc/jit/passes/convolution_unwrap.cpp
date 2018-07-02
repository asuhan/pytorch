#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"

namespace torch { namespace jit {

static void ConvolutionUnwrap(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks())
      ConvolutionUnwrap(sub);
    if (it->kind() == aten::convolution) {
      WithInsertPoint guard(*it);

      auto graph = block->owningGraph();
      auto node = *it;
      auto weights = node->inputs()[1]->type()->expect<TensorType>();
      auto convNode = graph->create(aten::thnn_conv2d_forward, 3)
	->is_(attr::stride, node->is(attr::stride))
	->is_(attr::padding, node->is(attr::padding))
	->is_(attr::kernel_size, {weights->sizes()[2], weights->sizes()[3]});

      graph->insertNode(convNode);
      for (auto input: node->inputs()) {
	convNode->addInput(input);
      }
      convNode->outputs()[0]->setType(it->outputs()[0]->type());
      it->output()->replaceAllUsesWith(convNode->outputs()[0]);
      it.destroyCurrent();
    }
  }
}

void ConvolutionUnwrap(const std::shared_ptr<Graph>& graph) {
  ConvolutionUnwrap(graph->block());
}


}}
