#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"

namespace {
using namespace torch::jit;
static void eraseOutput(Node* node, int output_nr, Gradient& gradient) {
  Value* output = node->outputs()[output_nr];
  auto num_outputs = gradient.f->outputs().size();

  // find index of this output in fgraph outputs
  int32_t output_idx = -1;
  for (uint32_t i = 0; i < num_outputs; i++) {
    if (gradient.f->outputs()[i] == output) {
      output_idx = i;
    }
  }
  if (output_idx == -1) {
    return;
  }

  // remove from fgraph outputs
  gradient.f->eraseOutput(output_idx);

  // remove from given node's output
  node->eraseOutput(output_nr);

  // find indices in df's input
  auto df_gradout_idx = output_idx;
  auto df_out_idx = num_outputs + gradient.df_input_captured_inputs.size();
  int df_input_captured_outputs_idx = -1;
  for (uint32_t i = 0; i < gradient.df_input_captured_outputs.size(); i++) {
    if (static_cast<uint64_t>(output_idx) ==
        gradient.df_input_captured_outputs[i]) {
      df_out_idx += i;
      df_input_captured_outputs_idx = i;
    }
  }
  JIT_ASSERT(df_input_captured_outputs_idx != -1);
  gradient.df_input_captured_outputs.erase(
      gradient.df_input_captured_outputs.begin() +
      df_input_captured_outputs_idx);
  for (uint32_t i = 0; i < gradient.df_input_captured_outputs.size(); i++) {
    if (gradient.df_input_captured_outputs[i] >
        static_cast<uint64_t>(df_input_captured_outputs_idx)) {
      gradient.df_input_captured_outputs[i] -= 1;
    }
  }

  // remove from all inputs of gradient.df
  Value* grad_output = gradient.df->inputs()[df_gradout_idx];
  Value* captured_output = gradient.df->inputs()[df_out_idx];

  // remove grad_output and captured_output from all nodes of gradient.df
  for (auto it = gradient.df->nodes().begin(), end = gradient.df->nodes().end();
       it != end;
       ++it) {
    int32_t node_grad_output_idx = -1;
    int32_t node_output_idx = -1;
    for (uint32_t i = 0; i < it->inputs().size(); i++) {
      if (it->inputs()[i] == grad_output) {
        node_grad_output_idx = i;
      }
      if (it->inputs()[i] == captured_output) {
        node_output_idx = i;
      }
    }
    JIT_ASSERT(node_grad_output_idx == -1); // assert it's not used anywhere
    if (node_output_idx != -1) {
      it->replaceInput(
          node_output_idx,
          gradient.df->insertConstant(at::empty({})));
    }
  }

  gradient.df->eraseInput(df_out_idx);
  gradient.df->eraseInput(df_gradout_idx);
}

} // namespace

namespace torch {
namespace jit {

void XlaRemoveUnused(Gradient& gradient) {
  for (auto it = gradient.f->nodes().begin(), end = gradient.f->nodes().end();
       it != end;
       ++it) {
    JIT_ASSERT(it->blocks().size() == 0);
    switch (it->kind()) {
      case aten::thnn_conv2d_forward: {
        at::ArrayRef<Value*> outputs = it->outputs();
        JIT_ASSERT(outputs.size() == 3);
        eraseOutput(*it, 2, gradient);
        eraseOutput(*it, 1, gradient);
        break;
      }
      case aten::max_pool2d_with_indices: {
        at::ArrayRef<Value*> outputs = it->outputs();
        JIT_ASSERT(outputs.size() == 2);
        eraseOutput(*it, 1, gradient);
        break;
      }
    }
  }
}

} // namespace jit
} // namespace torch
