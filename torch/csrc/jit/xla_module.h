#pragma once
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/jit/xla_code_impl.h"
#include "torch/csrc/jit/xla_tensor.h"
#include "torch/csrc/utils/disallow_copy.h"

#include <memory>

namespace torch {
namespace jit {

struct XlaModule : public std::enable_shared_from_this<XlaModule> {
  TH_DISALLOW_COPY_AND_ASSIGN(XlaModule);

  XlaModule(script::Module& module,
	    std::vector<autograd::Variable>& inputs,
	    bool backward=true);

  std::vector<std::shared_ptr<XLATensor> > forward(const std::vector<std::shared_ptr<XLATensor> >& inputs);
  void backward(const std::vector<std::shared_ptr<XLATensor> >& grad_outputs);

 private:
  std::vector<std::shared_ptr<XLATensor>> params_;
  xla::XlaComputation forward_graph_;
  xla::XlaComputation backward_graph_;
  bool backward_graph_initialized;

  std::shared_ptr<Graph> df_;

  // info for backwrd captures
  uint64_t f_real_outputs;
  std::vector<uint64_t> df_input_captured_inputs;
  std::vector<uint64_t> df_input_captured_outputs;

  // TODO: captured_outputs only needs shape, no need for holding onto full Tensor
  std::vector<std::shared_ptr<XLATensor>> inputs_;
  std::vector<std::shared_ptr<XLATensor>> captured_outputs_;
  std::vector<std::shared_ptr<XLATensor>> captured_inputs_outputs_;
};

} // namespace jit
} // namespace torch
