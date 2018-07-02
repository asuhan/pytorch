#include "torch/csrc/jit/autodiff.h"

#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/utils/functional.h"

#include <algorithm>

namespace torch { namespace jit {

using value_map = std::unordered_map<Value*, Value*>;
using value_set = std::unordered_set<Value*>;

bool hasOneValuedAttribute(Node *n, torch::jit::Symbol name) {
  return n->hasAttribute(name) && at::Scalar(n->t(name)).toDouble() == 1.0;
}

bool isDifferentiable(Node * n) {
  static std::unordered_set<Symbol> differentiable_kinds = {
    aten::add, aten::sub, aten::mul, prim::Constant, prim::ReplaceIfUndef,
    aten::sigmoid, aten::tanh, aten::mm, aten::chunk, aten::split, aten::t, aten::neg,
    aten::unsqueeze, aten::expand, aten::addmm, aten::gt, aten::lt, aten::eq, aten::ne, aten::ge, aten::le, aten::type_as,
    aten::relu, aten::threshold, aten::exp, aten::max_pool2d, aten::avg_pool2d, prim::AutogradAdd,
    aten::thnn_conv2d_forward
  };
  // TODO: check this more generally via schema
  // This check ensures that the `alpha` and `beta` attributes on this addmm
  // node are constant and equivalent to 1.0
  if (n->kind() == aten::addmm) {
    if (n->inputs().size() > 3)
      return false;
    if (!hasOneValuedAttribute(n, attr::alpha) || !hasOneValuedAttribute(n, attr::beta))
      return false;
  }
  if (n->kind() == aten::type_as && !n->inputs().at(1)->isTensor()) {
    return false;
  }

  // linear blocks may appear as inputs to graph executors, but they are removed
  // before differentiation occurs
  if (n->kind() == prim::GradOf) {
    auto body = n->blocks().at(0);
    return std::all_of(
        body->nodes().begin(),
        body->nodes().end(),
        static_cast<bool (*)(Node*)>(isDifferentiable));
  }

  return differentiable_kinds.count(n->kind()) > 0;
}


bool isDifferentiable(Graph & g) {
  return std::all_of(g.nodes().begin(), g.nodes().end(),
                     static_cast<bool(*)(Node*)>(isDifferentiable));
}


bool outputRequiresGrad(Node* node, std::function<bool(Value*)> requires_grad) {
  switch (node->kind()) {
    case aten::le:
    case aten::ge:
    case aten::lt:
    case aten::gt:
    case aten::ne:
    case aten::eq:
      return false;
    case aten::type_as:
    //type_as has two inputs, the second of which (setting type) might require grad, but it still won't affect the output of type_as requiring grad.
      return requires_grad(node->inputs().at(0));
    default:
      return std::any_of(node->inputs().begin(), node->inputs().end(), requires_grad);
  }
}



static std::vector<Value*> gradientForNode(Node* node, ArrayRef<Value*> grad_values) {
  const auto build_sym_grad = [node](const std::vector<SymbolicVariable>& grads) -> std::vector<SymbolicVariable> {
    auto inputs = fmap<SymbolicVariable>(node->inputs());
    auto outputs = fmap<SymbolicVariable>(node->outputs());
    switch(node->kind()) {
      case aten::add:
        // o = a - alpha*other
        if(inputs.size() == 1)
          return { grads.at(0) };
          // o = a + alpha*b
        return {grads.at(0), grads.at(0) * at::Scalar(node->t(attr::alpha)) };
      case aten::sub:
        // o = a - alpha*other
        if(inputs.size() == 1)
          return {grads.at(0)};
        // o = a - alpha*b
        return {grads.at(0), -grads.at(0) * at::Scalar(node->t(attr::alpha))};
      case aten::mul:
        // o = a * other
        if(inputs.size() == 1)
          return {grads.at(0) * at::Scalar(node->t(attr::other))};
        // o = a * b
        return {grads.at(0) * inputs.at(1), grads.at(0) * inputs.at(0)};
      case prim::Constant:
        return {};
      case prim::ReplaceIfUndef:
        return {grads.at(0), grads.at(0)};
      case aten::sigmoid:
        return {grads.at(0) * outputs.at(0) * (1 - outputs.at(0))};
      case aten::tanh:
        return {grads.at(0) * (1 - outputs.at(0) * outputs.at(0))};
      case aten::relu:
        return {grads.at(0) * (outputs.at(0) > at::Scalar(0)).type_as(outputs.at(0))};
      case aten::threshold:
        return {grads.at(0) * (inputs.at(0) > at::Scalar(node->t(attr::threshold))).type_as(outputs.at(0))};
      case aten::exp:
        return {grads.at(0) * (outputs.at(0))};
      case aten::chunk:
      case aten::split:
        return {SymbolicVariable::cat(grads, node->i(attr::dim))};
      case aten::t:
        return {grads.at(0).t()};
      case aten::neg:
        return {-grads.at(0)};
      case aten::view:
        return {grads.at(0).view(inputs.at(0).sizes())};
      case aten::type_as:
        return {grads.at(0).type_as(inputs.at(0))};
      case aten::unsqueeze:
        return {grads.at(0).squeeze(node->i(attr::dim))};
      case aten::mm: {
        SymbolicVariable dmat1, dmat2;
        if (auto type = inputs.at(0).value()->type()->cast<TensorType>()) {
	  // if sizes are statically known, optimize transposes
          auto sizes = type->sizes(), strides = type->strides();
          if (strides.at(0) == 1 && strides.at(1) == sizes.at(0)) {
            dmat1 = inputs.at(1).mm(grads.at(0).t()).t();
          } else { // sizes are not statically known, do generic matrix multiply
            dmat1 = grads.at(0).mm(inputs.at(1).t());
          }
        } else {
          dmat1 = grads.at(0).mm(inputs.at(1).t());
        }
        if (auto type = inputs.at(1).value()->type()->cast<TensorType>()) {
	  // if sizes are statically known, optimize transposes
          auto sizes = type->sizes(), strides = type->strides();
          if (strides.at(0) == 1 && strides.at(1) == sizes.at(0)) {
            dmat2 = grads.at(0).t().mm(inputs.at(0)).t();
          } else {
            dmat2 = inputs.at(0).t().mm(grads.at(0));
          }
        } else { // sizes are not statically known, do generic matrix multiply
          dmat2 = inputs.at(0).t().mm(grads.at(0));
        }
        return {dmat1, dmat2};
      }
      case aten::expand: {
        const auto& input_sizes = inputs.at(0).sizes();
        if (input_sizes.size() == 0)
          return {grads.at(0).sum()};
        auto grad_sizes = node->is(attr::size);
        auto grad = grads.at(0);
        while (grad_sizes.size() > input_sizes.size()) {
          grad = grad.sum(0, false);
          grad_sizes.erase(grad_sizes.begin());
        }
        for (size_t i = 0; i < input_sizes.size(); ++i) {
          if (input_sizes[i] == 1 && grad_sizes[i] > 1) {
            grad = grad.sum(i, true);
          }
        }
        return {grad};
      }
      case aten::squeeze: {
        const auto& sizes = inputs.at(0).sizes();
        if (node->hasAttribute(attr::dim)) {
          int dim = node->i(attr::dim);
          return {sizes.at(dim) > 1 ? grads.at(0) : grads.at(0).unsqueeze(dim)};
        } else {
          std::vector<size_t> squeezed_dims;
          for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i] != 1) continue;
            squeezed_dims.push_back(i);
          }
          SymbolicVariable returned_grad = grads.at(0);
          for (auto it = squeezed_dims.rbegin(); it != squeezed_dims.rend(); ++it)
            returned_grad = returned_grad.unsqueeze(*it);
          return {returned_grad};
        }
      }
      case aten::cat: {
        int dim = node->i(attr::dim);
        const auto& first_sizes = inputs.at(0).sizes();
        const auto has_first_sizes = [&first_sizes](SymbolicVariable var) {
          return var.sizes() == first_sizes;
        };
        // NB: this is a specialization for the common case where all inputs are
        // of equal sizes. We can use a single split operation to handle that.
        if (std::all_of(inputs.begin(), inputs.end(), has_first_sizes)) {
          return grads.at(0).chunk(inputs.size(), dim);
        } else {
          size_t offset = 0;
          auto grad = grads.at(0);
          std::vector<SymbolicVariable> returned_grads;
          for (auto input : inputs) {
            returned_grads.push_back(grad.narrow(dim, offset, input.sizes()[dim]));
            offset += input.sizes()[dim];
          }
          return returned_grads;
        }
      }
      case aten::max_pool2d: {
        JIT_ASSERT(grads.size() == 2);
        return {SymbolicVariable::max_pool2d_backward(grads.at(0), inputs.at(0),
                                                      outputs.at(1),
                                                      node->is(attr::kernel_size),
                                                      node->is(attr::stride),
                                                      node->is(attr::padding),
                                                      node->is(attr::dilation),
                                                      node->i(attr::ceil_mode))};
      }
      case aten::avg_pool2d: {
        JIT_ASSERT(grads.size() == 1);
        return {SymbolicVariable::avg_pool2d_backward(grads.at(0), inputs.at(0),
                                                      node->is(attr::kernel_size),
                                                      node->is(attr::stride),
                                                      node->is(attr::padding),
                                                      node->i(attr::ceil_mode),
                                                      node->i(attr::count_include_pad))};
      }
      case aten::thnn_conv2d_forward: {
	auto graph = node->owningGraph();
	auto convNode = graph->create(aten::thnn_conv2d_backward, 3)
	  ->is_(attr::stride, node->is(attr::stride))
	  ->is_(attr::padding, node->is(attr::padding))
	  ->is_(attr::kernel_size, node->is(attr::kernel_size))
	  ->is_(attr::output_mask, std::vector<int64_t>{1, 1, 1});
	auto f = grads.at(0);
	convNode->addInput(f);
	convNode->addInput(inputs.at(0));
	convNode->addInput(inputs.at(1));
	convNode->addInput(outputs.at(1));
	convNode->addInput(outputs.at(2));
	graph->insertNode(convNode);
	return fmap<SymbolicVariable>(convNode->outputs());
      }
    }
    throw std::runtime_error(std::string("don't support differentiation of `") +
                            node->kind().toDisplayString() + "`");
  };
  if (!isDifferentiable(node)) {
    throw std::runtime_error(std::string("differentiation of ") + node->kind().toDisplayString() + " "
                             "is not supported, or it is missing necessary type information");
  }
  auto sym_grads = build_sym_grad(fmap<SymbolicVariable>(grad_values));
  return fmap(sym_grads, [](const SymbolicVariable &v) { return v.value(); });
}

static value_set findAllRequiresGradNodes(
        Graph& graph, const std::vector<bool>& input_requires_grad) {
  JIT_ASSERT(graph.inputs().size() == input_requires_grad.size());
  std::unordered_set<Value*> requires_grad_set;
  const auto requires_grad = [&](Value *v) { return requires_grad_set.count(v) > 0; };

  auto inputs = graph.inputs();
  for (size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
    if (!input_requires_grad[i]) continue;
    requires_grad_set.emplace(inputs[i]);
  }

  for (Node * node : graph.nodes()) {
    if (!outputRequiresGrad(node, requires_grad)) continue;
    for (Value * output : node->outputs())
      requires_grad_set.emplace(output);
  }

  return requires_grad_set;
}


// If we have a function y = f(x) with jacobian J, the backwards of f is dx = J^t dy.
// Note that because the backwards always implements this matrix multiply,
// we know that it maps an input vector of zeros to an output vector of zero
// regardless of what operations it choses to do inside to actually implement
// the matrix multiply (most use some optimized form and never generate J^t).
// More generally, we know that all of the backward computations are linear and
// can use this property to do more aggressive optimizations later.
// It is ok to replace any backward function with known-zero inputs with something
// that produces known-zero outputs. This function encloses each know-linear
// backward function in a 'GradOf' sub-block so that we can perform optimizations
// using this information. In particular, specializeUndef will observe if
// all the inputs to the linear block are Undef, which the autograd uses to represent
// zeros, and then propagate the undefs to the outputs of the block.
static std::vector<Value*> linearGradientForNode(Node* node, ArrayRef<Value*> grad_values) {
  auto & graph = *node->owningGraph();
  auto linear = graph.insertNode(graph.create(prim::GradOf, {grad_values}, 0));
  // to make reading gradient graphs easier, remember the name of the forward op
  linear->s_(attr::name, node->kind().toDisplayString());
  auto block = linear->addBlock();
  {
    WithInsertPoint guard(block);
    auto results = gradientForNode(node, grad_values);
    for(auto r : results) {
      block->registerOutput(r);
      linear->addOutput()->copyMetadata(r);
    }
  }
  return linear->outputs();
}

struct ReverseDetails {
  ReverseDetails(value_map&& grad_map, value_set&& requires_grad_set, Block * reverse_block)
    : grad_map(std::move(grad_map))
    , requires_grad_set(std::move(requires_grad_set))
    , reverse_block(reverse_block) {}

  value_map grad_map;
  value_set requires_grad_set;
  Block * reverse_block;
};

// AutogradAdd is a special addition function that handles Undef
// AutogradAdd(a, b) == a + b if defined(a) and defined(b)
// AutogradAdd(Undef, b) == b
// AutogradAdd(a, Undef) == a
// AutogradAdd(Undef, Undef) == Undef
static Value* createAutogradAdd(Value* a, Value* b) {
  auto graph = a->owningGraph();
  return graph->insertNode(graph->create(prim::AutogradAdd, {a, b}))->output();
}

// Before:
//   - grad_desc has field f initialized to the original 0-stage graph
// After:
//   - the last node of f (f->nodes().reverse()[0]) is a gradient node
//     whose block has vjp inputs for all outputs that require_grad
//     and vjp outputs for all primal inputs that require_grad
//   - grad_desc has df_input_vjps and df_output_vjps set
//     (but df_input_vjps will be modified later as well)
static ReverseDetails addReverseInline(Gradient& grad_desc,
                                  const std::vector<bool>& input_requires_grad) {
  auto & graph = *grad_desc.f;
  // note: reverse_node is intentionally not inserted to avoid
  // accidentally acting on it (e.g. in elminate dead code),
  // std::cout << *reverse_node << to view its state.
  auto reverse_node = graph.create(prim::Reverse, 0);
  auto reverse_block = reverse_node->addBlock();
  WithInsertPoint guard(reverse_block);
  auto requires_grad_set = findAllRequiresGradNodes(graph, input_requires_grad);
  const auto requires_grad = [&](Value *v) { return requires_grad_set.count(v) > 0; };

  value_map grad_map; // x -> dx mapping
  const auto get_grad = [&](Value* v) -> Value* {
    auto it = grad_map.find(v);
    if (it == grad_map.end()) {
      auto undef = graph.insertNode(graph.createUndefined());
      std::tie(it, std::ignore) = grad_map.emplace(v, undef->output());
    }
    return it->second;
  };
  const auto set_grad = [&](Value *x, Value *dx) {
    if (Value * prev_grad = grad_map[x]) {
      grad_map[x] = createAutogradAdd(prev_grad, dx);
    } else {
      grad_map[x] = dx;
    }
  };

  auto outputs = graph.outputs();
  for (size_t i = 0, num_outputs = outputs.size(); i < num_outputs; ++i) {
    Value * output = outputs[i];
    if (!requires_grad(output))
      continue;
    Value * output_grad = reverse_block->addInput()->setType(output->type());
    set_grad(output, output_grad);
    grad_desc.df_input_vjps.push_back(i);
  }

  for (auto it = graph.nodes().rbegin(), end = graph.nodes().rend(); it != end; ++it) {
    Node *node = *it;
    auto inputs = node->inputs();
    if (!outputRequiresGrad(node, requires_grad)) continue;

    value_list grad_inputs = linearGradientForNode(node, fmap(node->outputs(), get_grad));
    JIT_ASSERT(grad_inputs.size() == node->inputs().size());
    for (size_t i = 0, num_inputs = grad_inputs.size(); i < num_inputs; ++i) {
      set_grad(inputs[i], grad_inputs[i]);
    }
  }

  auto inputs = graph.inputs();
  for (size_t i = 0, num_inputs = inputs.size(); i < num_inputs; ++i) {
    Value * input = inputs[i];
    if (!requires_grad(input))
      continue;
    reverse_block->registerOutput(get_grad(input));
    grad_desc.df_output_vjps.push_back(i);
  }
  return ReverseDetails(std::move(grad_map), std::move(requires_grad_set), reverse_block);
}

// Takes a grad_desc.f returned from `addReverseInline` and splits off the
// reverse_block into its own graph, storing it in df.
// All intermediates needed in the second stage are added to
// outputs of f, and taken as inputs in df. For a more
// detailed description see Note [Gradient graphs] in autodiff.h.
// This function also initializes the fields in grad_desc that were undefined after
// `addReverseInline` (and extends `df_input_vjps` with vjps for captured temporaries).
static void lambdaLiftReverse(Gradient& grad_desc, ReverseDetails& rev_info) {
  auto & graph = *grad_desc.f;
  auto primal_block = graph.block();
  auto reverse_block = rev_info.reverse_block;

  // --------------------------------------------------------------------------
  // 1. Find values of f that need to be captured.
  // --------------------------------------------------------------------------
  // First, we need to find all values that are produced in f,
  // and used in df. They will need to be added as inputs of the df
  // and some of them may also need to be appended as outputs of f if
  // they are not already an input or an output of f
  value_set reverse_captures_set;
  value_list reverse_captures; // Invariant: topo sorted
  auto check_uses = [&](Value *v) {
    for (auto use : v->uses()) {
      if (use.user->owningBlock() == primal_block)
        continue;
      if (/* bool unseen = */ reverse_captures_set.emplace(v).second) {
        reverse_captures.push_back(v);
      }
    }
  };
  for (Value * input : graph.inputs()) {
    if (input->stage() != 0) break;
    check_uses(input);
  }
  for (Node * node : graph.nodes()) {
    if (node->stage() != 0) break;
    for (Value * output : node->outputs())
      check_uses(output);
  }

  // --------------------------------------------------------------------------
  // 2. Prepare input/outputs lists for f and df
  // --------------------------------------------------------------------------
  // It's simple to construct primal_inputs/reverse_outputs,
  // but primal_outputs/reverse_inputs are much more subtle.
  // Here's a summary of how they are supposed to look like:
  //
  // Primal outputs:
  //   [original outputs], [temporaries]
  //
  // Reverse inputs:
  //   [output vjps (aka grad_outputs)], [temporary vjps]
  //   [captured primal values, in topological order],

  // -- Construct primal_outputs, df_input_captures, f_real_outputs ----
  grad_desc.f_real_outputs = graph.outputs().size();

  std::unordered_map<Value*, size_t> orig_primal_outputs_idx;
  std::unordered_map<Value*, size_t> orig_primal_inputs_idx;
  // NOTE: we use emplace to avoid replacing an existing index if an output is repeated
  for (size_t i = 0, num_outputs = graph.outputs().size(); i < num_outputs; ++i)
    orig_primal_outputs_idx.emplace(graph.outputs()[i], i);
  for (size_t i = 0, num_inputs = graph.inputs().size(); i < num_inputs; ++i)
    orig_primal_inputs_idx[graph.inputs()[i]] = i;

  // NB: reverse_captures are already deduplicated, and in topo order
  for (Value * capture_val : reverse_captures) {
    // If it's already an output we don't have to add anything,
    // but register the fact that it needs to be captured.
    if (orig_primal_outputs_idx.count(capture_val) > 0) {
      grad_desc.df_input_captured_outputs.push_back(orig_primal_outputs_idx[capture_val]);
    // If it's an input, we could add it as an output but in fact it's
    // more efficient to use a special kind of capture.
    } else if (orig_primal_inputs_idx.count(capture_val) > 0) {
      grad_desc.df_input_captured_inputs.push_back(orig_primal_inputs_idx.at(capture_val));
    // Otherwise it's just a regular intermediate value that we need to add as an output
    } else {
      // we need to create a new temporary output for this capture because it wasn't availiable.
      graph.registerOutput(capture_val);
      grad_desc.df_input_captured_outputs.emplace_back(graph.outputs().size() - 1);
    }
  }

  // -- Add VJPs for temporaries, adjust df_input_vjps -------------------------
  // NB [possible optimization]: use the newly added vjp input as soon as the first
  // vjp for that value is generated, to reduce the lifespan of this input
  // (currently we add it to the final vjp after all adds).
  for (size_t i = grad_desc.f_real_outputs; i < graph.outputs().size(); ++i) {
    Value * tmp = graph.outputs().at(i);
    // Add VJP inputs only for intermediates that actually required grad.
    if (rev_info.requires_grad_set.count(tmp) == 0) continue;
    Value * tmp_vjp_in = reverse_block->addInput()->setType(tmp->type());
    Value * tmp_vjp_prev = rev_info.grad_map.at(tmp);
    // This is quite weird because we can't first make a sum and then replace all uses
    // of tmp_vjp_prev (that would replace its use in the sum too!), so we create an
    // incorrect sum that doesn't use prev vjp, replace uses, and fix the sum.
    Value * new_vjp = createAutogradAdd(tmp_vjp_in, tmp_vjp_in);
    new_vjp->node()->moveAfter(tmp_vjp_prev->node());
    tmp_vjp_prev->replaceAllUsesWith(new_vjp);
    new_vjp->node()->replaceInput(1, tmp_vjp_prev);
    grad_desc.df_input_vjps.emplace_back(i);
  }

  // add the captures as formal arguments to the reverse_block
  // afterward inputs: [output vjps][temporary vjps][captures]
  // construct a map from captured 'value' to the index in the input list
  // used to extract this block into its own function
  std::unordered_map<Value*, size_t> capture_to_formal_index;
  const auto & add_capture = [&](Value * captured) {
    capture_to_formal_index[captured] = reverse_block->inputs().size();
    reverse_block->addInput()->copyMetadata(captured);
  };
  for(auto & offset : grad_desc.df_input_captured_inputs)
    add_capture(graph.inputs()[offset]);
  for(auto & offset : grad_desc.df_input_captured_outputs)
    add_capture(graph.outputs()[offset]);

  grad_desc.df = std::make_shared<Graph>();
  grad_desc.df->block()->cloneFrom(reverse_block, [&](Value* v) {
    return grad_desc.df->inputs()[capture_to_formal_index.at(v)];
  });
  // reverse_node was just to hold onto reverse_block in a debuggable way
  // we can remove it now.
  reverse_block->owningNode()->destroy();
}

Gradient differentiate(std::shared_ptr<Graph>& _graph, const std::vector<bool>& requires_grad) {
  Gradient grad_desc;
  // Take ownership of the graph
  JIT_ASSERTM(_graph.use_count() == 1,
              "differentiate will mutate and destroy the graph, so it requires "
              "graph.use_count() == 1, but found %d", _graph.use_count());
  std::swap(_graph, grad_desc.f);
  // XXX: Take care when handling outputs - they can be duplicated!

  WithInsertPoint guard(grad_desc.f->block());
  // Fills in df_input_vjps and df_output_vjps
  auto rev_info = addReverseInline(grad_desc, requires_grad);
  // addReverseInline has to call gradientForNode if *any* of the outputs
  // require grad, but it will emit vjps for *all* outputs. Use DCE to remove
  // unnecessary nodes.
  EliminateDeadCode(grad_desc.f);
  // Fills in f, df, f_real_outputs, df_input_captures,
  // modifies df_input_vjps (new vjps are added for temporaries)
  lambdaLiftReverse(grad_desc, rev_info);
  return grad_desc;
}

}}
