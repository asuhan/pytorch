#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/jit.h>
#include <torch/torch.h>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/softmax.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/ts_softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"
#include "lazy_tensors/permutation_util.h"

namespace torch_lazy_tensors {
namespace compiler {

class TSNodeLowering : public NodeLowering {
 public:
  TSNodeLowering(const std::string& name, ts_backend::TSLoweringContext* loctx)
      : NodeLowering(loctx),
        function_(loctx ? std::make_shared<torch::jit::GraphFunction>(
                              name, loctx->graph(), nullptr)
                        : nullptr) {}

  bool Lower(const ir::Node* node) override {
    TSOpVector ops = LowerToTS(node);
    if (ops.empty()) {
      return false;
    }
    LTC_CHECK_EQ(node->num_outputs(), ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
      loctx()->AssignOutputOp(ir::Output(node, i), ops[i]);
    }
    return true;
  }

  lazy_tensors::Shape Infer(const ir::Node* node) override {
    switch (node->op().op) {
      case at::aten::expand: {
        auto expand =
            ir::NodeCast<ir::ops::Expand>(node, ir::OpKind(at::aten::expand));
        const ir::Output& argument = node->operand(0);
        return lazy_tensors::Shape(argument.shape().element_type(),
                                   expand->size());
      }
      case at::aten::matmul: {
        // Only used from bmm currently.
        return InferBmm(node);
      }
      case at::aten::mm: {
        return InferMm(node);
      }
      case at::aten::native_batch_norm: {
        return InferBatchNorm(node);
      }
      case at::aten::native_batch_norm_backward: {
        return InferBatchNormBackward(node);
      }
      case at::aten::permute: {
        auto permute =
            ir::NodeCast<ir::ops::Permute>(node, ir::OpKind(at::aten::permute));
        const ir::Output& argument = node->operand(0);
        return ir::ops::Permute::MakePermuteShape(argument.shape(),
                                                  permute->dims());
      }
      case at::aten::constant_pad_nd: {
        auto constant_pad_nd = ir::NodeCast<ir::ops::ConstantPadNd>(
            node, ir::OpKind(at::aten::constant_pad_nd));
        const ir::Output& argument = node->operand(0);
        const lazy_tensors::Shape& argument_shape = argument.shape();
        const auto argument_dimensions = argument_shape.dimensions();
        const auto& pad = constant_pad_nd->pad();
        LTC_CHECK_EQ(argument_dimensions.size() * 2, pad.size());
        std::vector<lazy_tensors::int64> padded_dimensions(
            argument_dimensions.begin(), argument_dimensions.end());
        size_t i = 0;
        for (auto rit = pad.rbegin(); rit != pad.rend(); rit += 2, ++i) {
          padded_dimensions[i] += (*rit + *(rit + 1));
        }
        return lazy_tensors::Shape(argument_shape.element_type(),
                                   padded_dimensions);
      }
      case at::aten::eq:
      case at::aten::ge:
      case at::aten::gt:
      case at::aten::le:
      case at::aten::lt:
      case at::aten::ne: {
        return InferComparison(node);
      }
      default:
        LTC_LOG(FATAL) << *node << "Not implemented yet.";
    }
  }

  TSOpVector LowerToTS(const ir::Node* node) {
    if (node->op().op == at::aten::as_strided) {
      return LowerAsStrided(ir::NodeCast<ir::ops::AsStrided>(
          node, ir::OpKind(at::aten::as_strided)));
    }
    if (node->op() == *ir::ops::ltc_as_strided_view_update) {
      return LowerAsStridedViewUpdate(
          ir::NodeCast<ir::ops::AsStridedViewUpdate>(
              node, *ir::ops::ltc_as_strided_view_update));
    }
    if (node->op() == *ir::ops::ltc_cast) {
      return LowerCast(ir::NodeCast<ir::ops::Cast>(node, *ir::ops::ltc_cast));
    }
    if (node->op().op == at::prim::Constant) {
      auto scalar_node = dynamic_cast<const ir::ops::Scalar*>(node);
      if (scalar_node) {
        return LowerScalar(scalar_node);
      }
      return LowerConstant(ir::NodeCast<ir::ops::Constant>(
          node, ir::OpKind(at::prim::Constant)));
    }
    if (node->op().op == at::aten::native_batch_norm) {
      return LowerBatchNorm(ir::NodeCast<ir::ops::TSNativeBatchNormForward>(
          node, ir::OpKind(at::aten::native_batch_norm)));
    }
    if (node->op().op == at::aten::native_batch_norm_backward) {
      return LowerBatchNormBackward(
          ir::NodeCast<ir::ops::TSNativeBatchNormBackward>(
              node, ir::OpKind(at::aten::native_batch_norm_backward)));
    }
    if (node->op().op == at::aten::constant_pad_nd) {
      return LowerConstantPad(ir::NodeCast<ir::ops::ConstantPadNd>(
          node, ir::OpKind(at::aten::constant_pad_nd)));
    }
    if (node->op().op == at::aten::expand) {
      return LowerExpand(
          ir::NodeCast<ir::ops::Expand>(node, ir::OpKind(at::aten::expand)));
    }
    if (node->op().op == at::aten::permute) {
      return LowerPermute(
          ir::NodeCast<ir::ops::Permute>(node, ir::OpKind(at::aten::permute)));
    }
    if (node->op().op == at::aten::softmax) {
      return LowerSoftmax(
          ir::NodeCast<ir::ops::Softmax>(node, ir::OpKind(at::aten::softmax)));
    }
    if (node->op().op == at::aten::_softmax_backward_data) {
      return LowerSoftmaxBackward(ir::NodeCast<ir::ops::TSSoftmaxBackward>(
          node, ir::OpKind(at::aten::_softmax_backward_data)));
    }
    if (node->op().op == at::aten::unsqueeze) {
      return LowerUnsqueeze(ir::NodeCast<ir::ops::Unsqueeze>(
          node, ir::OpKind(at::aten::unsqueeze)));
    }
    if (node->op().op == at::aten::view) {
      return LowerView(
          ir::NodeCast<ir::ops::View>(node, ir::OpKind(at::aten::view)));
    }
    if (node->op() == *ir::ops::ltc_device_data) {
      ir::ops::DeviceData* device_data_node =
          ir::NodeCast<ir::ops::DeviceData>(node, *ir::ops::ltc_device_data);
      return {loctx()->GetParameter(device_data_node->data())};
    }
    std::vector<torch::jit::NamedValue> arguments;
    for (const ir::Output& output : node->operands()) {
      arguments.emplace_back(loctx()->GetOutputOp(output));
    }
    return LowerBuiltin(node, arguments);
  }

 private:
  static lazy_tensors::Shape InferComparison(const ir::Node* node) {
    const ir::Output& lhs = node->operand(0);
    const ir::Output& rhs = node->operand(1);
    return lazy_tensors::Shape(
        lazy_tensors::PrimitiveType::PRED,
        Helpers::GetPromotedShape(lhs.shape().dimensions(),
                                  rhs.shape().dimensions()));
  }

  static lazy_tensors::Shape InferBatchNorm(const ir::Node* node) {
    const ir::Output& input = node->operand(0);
    const ir::Output& running_mean = node->operand(3);
    const ir::Output& running_var = node->operand(4);
    return lazy_tensors::ShapeUtil::MakeTupleShape(
        {input.shape(), running_mean.shape(), running_var.shape()});
  }

  static lazy_tensors::Shape InferBatchNormBackward(const ir::Node* node) {
    const ir::Output& input = node->operand(1);
    const ir::Output& weight = node->operand(2);
    return lazy_tensors::ShapeUtil::MakeTupleShape(
        {input.shape(), weight.shape(), weight.shape()});
  }

  static lazy_tensors::Shape InferBmm(const ir::Node* node) {
    const ir::Output& tensor1 = node->operand(0);
    const ir::Output& tensor2 = node->operand(1);
    const lazy_tensors::Shape& tensor1_shape = tensor1.shape();
    const lazy_tensors::Shape& tensor2_shape = tensor2.shape();
    LTC_CHECK_EQ(tensor1_shape.rank(), 3);
    LTC_CHECK_EQ(tensor2_shape.rank(), 3);
    lazy_tensors::int64 b = tensor1_shape.dimensions(0);
    lazy_tensors::int64 n = tensor1_shape.dimensions(1);
    lazy_tensors::int64 m1 = tensor1_shape.dimensions(2);
    LTC_CHECK_EQ(tensor2_shape.dimensions(0), b);
    LTC_CHECK_EQ(tensor2_shape.dimensions(1), m1);
    lazy_tensors::int64 p = tensor2_shape.dimensions(2);
    return lazy_tensors::Shape(tensor1_shape.element_type(), {b, n, p});
  }

  static lazy_tensors::Shape InferMm(const ir::Node* node) {
    const ir::Output& tensor1 = node->operand(0);
    const ir::Output& tensor2 = node->operand(1);
    const lazy_tensors::Shape& tensor1_shape = tensor1.shape();
    const lazy_tensors::Shape& tensor2_shape = tensor2.shape();
    LTC_CHECK_EQ(tensor1_shape.rank(), 2);
    LTC_CHECK_EQ(tensor2_shape.rank(), 2);
    lazy_tensors::int64 n = tensor1_shape.dimensions(0);
    lazy_tensors::int64 m = tensor1_shape.dimensions(1);
    LTC_CHECK_EQ(tensor2_shape.dimensions(0), m);
    lazy_tensors::int64 p = tensor2_shape.dimensions(1);
    return lazy_tensors::Shape(tensor1_shape.element_type(), {n, p});
  }

  TSOpVector LowerBuiltin(
      const ir::Node* node,
      const std::vector<torch::jit::NamedValue>& arguments) {
    return LowerBuiltin(node->op().op, arguments);
  }

  TSOpVector LowerBuiltin(
      c10::Symbol sym, const std::vector<torch::jit::NamedValue>& arguments) {
    auto builtin =
        std::make_shared<torch::jit::BuiltinFunction>(sym, at::nullopt);
    auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
    auto ret = magic_method->call({}, *function_, arguments, {}, 0);
    auto sv = dynamic_cast<torch::jit::SimpleValue*>(ret.get());
    LTC_CHECK(sv);
    if (sv->getValue()->type()->kind() == c10::TypeKind::TupleType) {
      const auto tuple_call_result = sv->asTuple({}, *function_);
      TSOpVector tuple_result;
      for (const auto& tuple_component : tuple_call_result) {
        auto tuple_component_sv =
            dynamic_cast<torch::jit::SimpleValue*>(tuple_component.get());
        tuple_result.push_back(tuple_component_sv->getValue());
      }
      return tuple_result;
    }
    return {sv->getValue()};
  }

  TSOpVector LowerAsStrided(const ir::ops::AsStrided* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->size());
    arguments.emplace_back(node->stride());
    arguments.emplace_back(node->storage_offset());
    TSOpVector as_strided_out = LowerBuiltin(node, arguments);
    LTC_CHECK_EQ(as_strided_out.size(), 1);
    std::vector<torch::jit::NamedValue> clone_arguments;
    clone_arguments.emplace_back(as_strided_out.front());
    return LowerBuiltin(at::aten::clone, clone_arguments);
  }

  TSOpVector LowerAsStridedViewUpdate(
      const ir::ops::AsStridedViewUpdate* node) {
    std::vector<torch::jit::NamedValue> clone_arguments;
    clone_arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    TSOpVector clone_out = LowerBuiltin(at::aten::clone, clone_arguments);
    LTC_CHECK_EQ(clone_out.size(), 1);
    torch::jit::Value* destination = clone_out.front();
    const ir::Output& input_op = node->operand(1);
    const lazy_tensors::Shape& input_shape = input_op.shape();
    const auto input_dimensions = input_shape.dimensions();
    std::vector<torch::jit::NamedValue> dest_arguments;
    dest_arguments.emplace_back(destination);
    dest_arguments.emplace_back(std::vector<lazy_tensors::int64>(
        input_dimensions.begin(), input_dimensions.end()));
    dest_arguments.emplace_back(node->stride());
    dest_arguments.emplace_back(node->storage_offset());
    TSOpVector as_strided_out =
        LowerBuiltin(at::aten::as_strided, dest_arguments);
    LTC_CHECK_EQ(as_strided_out.size(), 1);
    torch::jit::Value* as_strided = as_strided_out.front();
    std::vector<torch::jit::NamedValue> copy_from_arguments;
    copy_from_arguments.emplace_back(as_strided);
    copy_from_arguments.emplace_back(loctx()->GetOutputOp(input_op));
    LowerBuiltin(at::aten::copy_, copy_from_arguments);
    return {destination};
  }

  TSOpVector LowerBatchNorm(const ir::ops::TSNativeBatchNormForward* node) {
    std::vector<torch::jit::NamedValue> arguments;
    for (size_t i = 0; i < 5; ++i) {
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(i)));
    }
    arguments.emplace_back(node->training());
    arguments.emplace_back(node->momentum());
    arguments.emplace_back(node->eps());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerBatchNormBackward(
      const ir::ops::TSNativeBatchNormBackward* node) {
    std::vector<torch::jit::NamedValue> arguments;
    for (size_t i = 0; i < 7; ++i) {
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(i)));
    }
    arguments.emplace_back(node->training());
    arguments.emplace_back(node->eps());
    arguments.emplace_back(node->output_mask());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerCast(const ir::ops::Cast* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->dtype());
    return LowerBuiltin(at::aten::to, arguments);
  }

  TSOpVector LowerConstant(const ir::ops::Constant* node) {
    return {loctx()->graph()->insertConstant(node->value().value())};
  }

  TSOpVector LowerConstantPad(const ir::ops::ConstantPadNd* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->pad());
    arguments.emplace_back(node->value());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerExpand(const ir::ops::Expand* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->size());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerPermute(const ir::ops::Permute* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->dims());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerScalar(const ir::ops::Scalar* node) {
    const at::Scalar& value = node->value();
    const lazy_tensors::Shape& shape = node->shape();
    return {loctx()->graph()->insertConstant(at::scalar_tensor(
        value, lazy_tensors::PrimitiveToScalarType(shape.element_type())))};
  }

  TSOpVector LowerSoftmax(const ir::ops::Softmax* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->dim());
    arguments.emplace_back(node->dtype());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerSoftmaxBackward(const ir::ops::TSSoftmaxBackward* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(1)));
    arguments.emplace_back(node->dim());
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(2)));
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerUnsqueeze(const ir::ops::Unsqueeze* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->dim());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerView(const ir::ops::View* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->output_size());
    return LowerBuiltin(at::aten::reshape, arguments);
  }

  ts_backend::TSLoweringContext* loctx() {
    return static_cast<ts_backend::TSLoweringContext*>(loctx_);
  }

  std::shared_ptr<torch::jit::GraphFunction> function_;
};

NodeLowering* GetTSNodeLowering() {
  static TSNodeLowering* ts_node_lowering =
      new TSNodeLowering("ltc-ts", nullptr);
  return ts_node_lowering;
}

std::unique_ptr<NodeLowering> CreateTSNodeLowering(ir::LoweringContext* loctx) {
  return std::make_unique<TSNodeLowering>(
      "ltc-ts", static_cast<ts_backend::TSLoweringContext*>(loctx));
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
