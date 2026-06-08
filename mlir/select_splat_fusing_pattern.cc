#include "mlir/select_splat_fusing_pattern.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace libtt::mlir_frontend {
namespace {

std::optional<uint32_t> packedSplatConstant(mlir::Value value) {
  while (auto broadcast =
             value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    value = broadcast.getOperand();
  }

  auto constant = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!constant) {
    return std::nullopt;
  }
  auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat()) {
    return std::nullopt;
  }

  mlir::Type elementType = dense.getElementType();
  if (elementType.isBF16() || elementType.isF16()) {
    auto bits = dense.getSplatValue<llvm::APFloat>().bitcastToAPInt();
    uint32_t value16 = bits.extractBitsAsZExtValue(16, 0);
    return value16 | (value16 << 16);
  }
  if (elementType.isF32()) {
    auto bits = dense.getSplatValue<llvm::APFloat>().bitcastToAPInt();
    return bits.extractBitsAsZExtValue(32, 0);
  }
  if (auto integer = llvm::dyn_cast<mlir::IntegerType>(elementType);
      integer && integer.getWidth() <= 32) {
    return static_cast<uint32_t>(dense.getSplatValue<llvm::APInt>().getZExtValue());
  }
  return std::nullopt;
}

std::optional<uint32_t> packedSelectArm(mlir::Value value,
                                        mlir::Type resultType) {
  if (value.getType() != resultType) {
    return std::nullopt;
  }
  return packedSplatConstant(value);
}

void eraseDeadSplatProducers(mlir::PatternRewriter& rewriter,
                             mlir::Value value) {
  auto broadcast = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (broadcast) {
    value = broadcast.getOperand();
    if (broadcast->use_empty()) {
      rewriter.eraseOp(broadcast);
    }
  }

  auto constant = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (constant && constant->use_empty()) {
    rewriter.eraseOp(constant);
  }
}

} // namespace

mlir::LogicalResult
SelectSplatFusing::matchAndRewrite(mlir::stablehlo::SelectOp selectOp,
                                   mlir::PatternRewriter& rewriter) const {
  if (selectOp->getNumResults() != 1) {
    return mlir::failure();
  }

  mlir::Type resultType = selectOp.getResult().getType();
  mlir::Value trueValue = selectOp.getOnTrue();
  mlir::Value falseValue = selectOp.getOnFalse();
  std::optional<uint32_t> onTrue =
      packedSelectArm(trueValue, resultType);
  std::optional<uint32_t> onFalse =
      packedSelectArm(falseValue, resultType);
  if (!onTrue && !onFalse) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> inputs;
  inputs.push_back(selectOp.getPred());
  if (!onTrue) {
    inputs.push_back(trueValue);
  }
  if (!onFalse) {
    inputs.push_back(falseValue);
  }

  auto backendConfig = rewriter.getDictionaryAttr({
      rewriter.getNamedAttr("on_true_is_constant",
                            rewriter.getBoolAttr(onTrue.has_value())),
      rewriter.getNamedAttr(
          "on_true_packed_value",
          rewriter.getI64IntegerAttr(static_cast<int64_t>(onTrue.value_or(0)))),
      rewriter.getNamedAttr("on_false_is_constant",
                            rewriter.getBoolAttr(onFalse.has_value())),
      rewriter.getNamedAttr(
          "on_false_packed_value",
          rewriter.getI64IntegerAttr(static_cast<int64_t>(onFalse.value_or(0)))),
  });

  rewriter.setInsertionPoint(selectOp);
  auto customCall = rewriter.create<mlir::stablehlo::CustomCallOp>(
      selectOp.getLoc(), selectOp->getResultTypes(), inputs, kSelectSplatTarget,
      /*hasSideEffect=*/false, backendConfig,
      mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI,
      rewriter.getArrayAttr({}),
      /*calledComputations=*/nullptr,
      /*operandLayouts=*/nullptr,
      /*resultLayouts=*/nullptr);
  rewriter.replaceOp(selectOp, customCall.getResults());

  if (onTrue) {
    eraseDeadSplatProducers(rewriter, trueValue);
  }
  if (onFalse) {
    eraseDeadSplatProducers(rewriter, falseValue);
  }
  return mlir::success();
}

mlir::LogicalResult runSelectSplatFusing(mlir::MLIRContext& context,
                                         mlir::ModuleOp module) {
  mlir::RewritePatternSet patterns(&context);
  patterns.add<SelectSplatFusing>(&context);
  mlir::GreedyRewriteConfig config;
  config.enableFolding(false).enableConstantCSE(false);
  return mlir::applyPatternsGreedily(module, std::move(patterns), config);
}

} // namespace libtt::mlir_frontend
