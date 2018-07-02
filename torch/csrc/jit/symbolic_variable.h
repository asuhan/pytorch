#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

struct SymbolicVariable {
  SymbolicVariable() : v(nullptr) {}
  /* implicit */ SymbolicVariable(Value * v) : v(v) {}
  // we allow implicit conversions to/from Value since
  // this type truly just provides more methods for value
  operator Value*() {
    return v;
  }
  static SymbolicVariable asNewInput(Graph & g, std::string name = "") {
    return g.addInput(name);
  }
  static SymbolicVariable asNewInput(Graph & g, TypePtr type) {
    return g.addInput()->setType(std::move(type));
  }
  const std::vector<int64_t>& sizes() {
    return v->type()->expect<TensorType>()->sizes();
  }
  void addAsOutput() {
    v->owningGraph()->registerOutput(v);
  }
  static std::vector<SymbolicVariable> create(Symbol kind, ArrayRef<SymbolicVariable> inputs,
                                 int num_outputs = 1,
                                 Node** created_node = nullptr,
                                 Graph * g = nullptr) {
      if(g == nullptr) {
        g = inputs.at(0).value()->owningGraph();
      }
      Node * n = g->insertNode(g->create(kind, num_outputs));
      for(auto i : inputs) {
        n->addInput(i.value());
      }
      if(created_node) {
        *created_node = n;
      }
      std::vector<SymbolicVariable> out;
      for(auto v : n->outputs()) {
        out.emplace_back(v);
      }
      return out;
  }
  static bool isConstInt(at::Scalar s, int32_t i) {
    // int32_t is safely convertible to both double and int64_t
    if(s.isFloatingPoint()) {
      return (double) i == s.toDouble();
    } else {
      return (int64_t) i == s.toLong();
    }
  }
  SymbolicVariable operator*(const SymbolicVariable rhs) const {
    return create(aten::mul, {*this, rhs})[0].typeLike(*this);
  }
  SymbolicVariable operator*(at::Scalar rhs) const {
    if(isConstInt(rhs, 1))
      return *this;
    Node * n;
    auto r = create(aten::mul, {*this}, 1, &n)[0];
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator>(at::Scalar rhs) const {
    Node * n;
    auto r = create(aten::gt, {*this}, 1, &n)[0].typeLikeWithScalarType(*this, at::kByte);
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator<(at::Scalar rhs) const {
    Node * n;
    auto r = create(aten::lt, {*this}, 1, &n)[0].typeLikeWithScalarType(*this, at::kByte);
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator>=(at::Scalar rhs) const {
    Node * n;
    auto r = create(aten::ge, {*this}, 1, &n)[0].typeLikeWithScalarType(*this, at::kByte);
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator<=(at::Scalar rhs) const {
    Node * n;
    auto r = create(aten::le, {*this}, 1, &n)[0].typeLikeWithScalarType(*this, at::kByte);
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator==(at::Scalar rhs) const {
    Node * n;
    auto r = create(aten::eq, {*this}, 1, &n)[0].typeLikeWithScalarType(*this, at::kByte);
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator!=(at::Scalar rhs) const {
    Node * n;
    auto r = create(aten::ne, {*this}, 1, &n)[0].typeLikeWithScalarType(*this, at::kByte);
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator+(const SymbolicVariable rhs) const {
    Node * n;
    auto r = create(aten::add, {*this, rhs}, 1, &n)[0].typeLike(*this);
    n->t_(attr::alpha, at::Scalar(1).toTensor());
    return r;
  }
  SymbolicVariable operator+(at::Scalar rhs) const {
    Node * n;
    auto r = create(aten::add, {*this}, 1, &n)[0].typeLike(*this);
    n->t_(attr::alpha, at::Scalar(1).toTensor());
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator-() const {
    return create(aten::neg, {*this})[0].typeLike(*this);
  }
  SymbolicVariable operator-(const SymbolicVariable rhs) const {
    Node *n;
    auto r = create(aten::sub, {*this, rhs}, 1, &n)[0].typeLike(*this);
    n->t_(attr::alpha, at::Scalar(1).toTensor());
    return r;
  }
  SymbolicVariable operator/(at::Scalar rhs) const {
    Node *n;
    auto r = create(aten::div, {*this}, 1, &n)[0].typeLike(*this);
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator%(at::Scalar rhs) const {
    Node *n;
    auto r = create(aten::remainder, {*this}, 1, &n)[0].typeLike(*this);
    n->t_(attr::other, rhs.toTensor());
    return r;
  }
  SymbolicVariable mm(const SymbolicVariable rhs) const {
    auto r = create(t("mm"), {*this, rhs})[0];
    return r;
  }
  SymbolicVariable t() const {
    auto r = create(t("t"), {*this})[0];
    return r;
  }
  SymbolicVariable sigmoid() const {
    return create(aten::sigmoid, {*this})[0].typeLike(*this);
  }
  SymbolicVariable tanh() const {
    return create(aten::tanh, {*this})[0].typeLike(*this);
  }
  std::vector<SymbolicVariable> chunk(int32_t chunks, uint32_t dim) const {
    Node * n;
    auto r = create(t("chunk"), { *this }, chunks, &n);
    n->i_(a("chunks"), chunks)
     ->i_(a("dim"), dim);
    return r;
  }
  SymbolicVariable type_as(const SymbolicVariable rhs) const {
    return create(aten::type_as, {*this, rhs})[0].typeLikeWithRhsScalarType(*this, rhs);
  }
  SymbolicVariable narrow(int dim, int64_t start, int64_t length) const {
    Node * n;
    auto r = create(t("narrow"), { *this }, 1, &n)[0];
    n->i_(a("dim"), dim)
     ->i_(a("start"), start)
     ->i_(a("length"), length);
    return r;
  }
  static SymbolicVariable cat(ArrayRef<SymbolicVariable> inputs, int32_t dim) {
    Node* n;
    auto r = create(aten::cat, inputs, 1, &n)[0];
    n->i_(attr::dim, dim);
    return r;
  }
  static SymbolicVariable stack(ArrayRef<SymbolicVariable> inputs, int32_t dim) {
    Node* n;
    auto r = create(aten::stack, inputs, 1, &n)[0];
    n->i_(attr::dim, dim);
    return r;
  }
  SymbolicVariable sum() const {
    auto r = create(t("sum"), {*this})[0];
    return r;
  }
  SymbolicVariable sum(int dim, bool keepdim) const {
    Node * n;
    auto r = create(t("sum"), {*this}, 1, &n)[0];
    n->is_(a("dim"), {dim})
     ->i_(a("keepdim"), keepdim);
    return r;
  }
  SymbolicVariable squeeze(int dim) const {
    Node * n;
    auto r = create(t("squeeze"), {*this}, 1, &n)[0];
    n->i_(a("dim"), dim);
    return r;
  }
  SymbolicVariable transpose(int dim0, int dim1) const {
    Node *n;
    auto r =  create(t("transpose"), {*this}, 1, &n)[0];
    n->i_(a("dim0"), dim0)
     ->i_(a("dim1"), dim1);
    return r;
  }
  SymbolicVariable unsqueeze(int dim) const {
    Node * n;
    auto r = create(t("unsqueeze"), {*this}, 1, &n)[0];
    n->i_(a("dim"), dim);
    return r;
  }
  SymbolicVariable view(std::vector<std::int64_t> sizes) const {
    Node *n;
    auto r =  create(aten::view, {*this}, 1, &n)[0];
    n->is_(a("size"), std::move(sizes));
    return r;
  }
  SymbolicVariable addmm(SymbolicVariable mat1, SymbolicVariable mat2) const {
    Node *n;
    auto r = create(aten::addmm, {*this, mat1, mat2}, 1, &n)[0];
    n->t_(a("alpha"), at::CPU(at::kFloat).scalarTensor(1.0));
    n->t_(a("beta"), at::CPU(at::kFloat).scalarTensor(1.0));
    return r;
  }
  static SymbolicVariable max_pool2d_backward(const SymbolicVariable grad,
                                              const SymbolicVariable input,
                                              const SymbolicVariable indices,
                                              const std::vector<int64_t>& kernel_size,
                                              const std::vector<int64_t>& stride,
                                              const std::vector<int64_t>& padding,
                                              const std::vector<int64_t>& dilation,
                                              const bool ceil_mode) {
    Node *n;
    auto r = create(aten::max_pool2d_backward, {grad, input, indices}, 1, &n)[0];
    n->is_(attr::kernel_size, kernel_size);
    n->is_(attr::stride, stride);
    n->is_(attr::padding, padding);
    n->is_(attr::dilation, dilation);
    n->i_(attr::ceil_mode, ceil_mode);
    return r;
  }
  Value * value() const {
    return v;
  }
  static SymbolicVariable avg_pool2d_backward(const SymbolicVariable grad,
                                              const SymbolicVariable input,
                                              const std::vector<int64_t>& kernel_size,
                                              const std::vector<int64_t>& stride,
                                              const std::vector<int64_t>& padding,
                                              const bool ceil_mode,
                                              const bool count_include_pad) {
    Node *n;
    auto r = create(aten::avg_pool2d_backward, {grad, input}, 1, &n)[0];
    n->is_(attr::kernel_size, kernel_size);
    n->is_(attr::stride, stride);
    n->is_(attr::padding, padding);
    n->i_(attr::ceil_mode, ceil_mode);
    n->i_(attr::count_include_pad, count_include_pad);
    return r;
  }
  static SymbolicVariable convolution(const SymbolicVariable input,
				      const SymbolicVariable weight,
				      const SymbolicVariable bias,
				      const std::vector<int64_t>& stride,
				      const std::vector<int64_t>& padding,
				      const std::vector<int64_t>& dilation,
				      int transposed,
				      const std::vector<int64_t>& output_padding,
				      int groups) {
    Node *n;
    auto r = create(aten::convolution, {input, weight, bias}, 1, &n)[0];
    n->is_(attr::stride, stride);
    n->is_(attr::padding, padding);
    n->is_(attr::dilation, dilation);
    n->i_(attr::transposed, transposed);
    n->is_(attr::output_padding, output_padding);
    n->i_(attr::groups, groups);
    return r;
  }
private:
  SymbolicVariable typeLike(SymbolicVariable other) {
    if (auto other_type = other.v->type()->cast<TensorType>())
      v->setType(other_type->contiguous());
    return *this;
  }
  SymbolicVariable typeLikeWithScalarType(SymbolicVariable other, at::ScalarType type) {
    if (auto other_type = other.v->type()->cast<TensorType>()){
      auto new_type = other_type->toScalarType(type)->contiguous();
      v->setType(new_type);
    }
    return *this;
  }
  SymbolicVariable typeLikeWithRhsScalarType(SymbolicVariable other, SymbolicVariable rhs) {
    auto other_type = other.v->type()->cast<TensorType>();
    auto rhs_type = rhs.v->type()->cast<TensorType>();
    if (other_type && rhs_type){
      auto new_type = other_type->toScalarType(rhs_type->scalarType())->contiguous();
      v->setType(new_type);
    }
    return *this;
  }
  static Symbol a(const char * s_) {
    return Symbol::attr(s_);
  }
  static Symbol t(const char * s_) {
    return Symbol::aten(s_);
  }
  Value * v;
};

// shorter method so that toVar(v) + toVar(c) is short.
static inline SymbolicVariable toVar(Value * v) {
  return SymbolicVariable(v);
}

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
inline SymbolicVariable operator+(T lhs, SymbolicVariable rhs) {
  return rhs + at::Scalar(lhs);
}

inline SymbolicVariable operator+(at::Scalar lhs, SymbolicVariable rhs) {
  return rhs + lhs;
}

inline SymbolicVariable operator-(at::Scalar lhs, SymbolicVariable rhs) {
  return (lhs + (-rhs));
}

}}
