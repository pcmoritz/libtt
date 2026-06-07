#include "mlir/rms_norm_fusing_pattern.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/stablehlo_utils.h"

namespace libtt::mlir_frontend {
namespace {

struct RmsNormComponents {
  mlir::Value input;
  mlir::Value weight;
  uint32_t scaleBits = 0;
  uint32_t biasBits = 0;
};

struct NormalizedInput {
  mlir::Value f32;
  mlir::Value bf16;
};

struct ScaleAndWeight {
  mlir::stablehlo::RsqrtOp rsqrt;
  mlir::Value weight;
};

std::optional<mlir::RankedTensorType> staticRankedTensor(mlir::Value value) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensor || !tensor.hasStaticShape()) {
    return std::nullopt;
  }
  return tensor;
}

bool isStaticBf16Tensor(mlir::Value value) {
  auto tensor = staticRankedTensor(value);
  return tensor && tensor->getElementType().isBF16();
}

bool isStaticF32Tensor(mlir::Value value) {
  auto tensor = staticRankedTensor(value);
  return tensor && tensor->getElementType().isF32();
}

int64_t elementCount(mlir::RankedTensorType type) {
  int64_t count = 1;
  for (int64_t dim : type.getShape()) {
    count *= dim;
  }
  return count;
}

mlir::Value peelIdentityAndReshape(mlir::Value value) {
  while (true) {
    value = peelIdentityCustomCalls(value);
    auto reshape = value.getDefiningOp<mlir::stablehlo::ReshapeOp>();
    if (!reshape) {
      return value;
    }
    auto inputType = staticRankedTensor(reshape.getOperand());
    auto outputType = staticRankedTensor(reshape.getResult());
    if (!inputType || !outputType ||
        inputType->getElementType() != outputType->getElementType() ||
        elementCount(*inputType) != elementCount(*outputType)) {
      return value;
    }
    value = reshape.getOperand();
  }
}

mlir::Value peelScaleShapeOps(mlir::Value value) {
  while (true) {
    value = peelIdentityAndReshape(value);
    auto broadcast = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
    if (!broadcast) {
      return value;
    }
    value = broadcast.getOperand();
  }
}

std::optional<uint32_t> f32SplatBits(mlir::Value value) {
  value = peelScaleShapeOps(value);
  auto constant = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!constant) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat() || !dense.getElementType().isF32()) {
    return std::nullopt;
  }
  auto bits = dense.getSplatValue<llvm::APFloat>().bitcastToAPInt();
  return bits.extractBitsAsZExtValue(32, 0);
}

std::optional<float> f32SplatValue(mlir::Value value) {
  value = peelScaleShapeOps(value);
  auto constant = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!constant) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat() || !dense.getElementType().isF32()) {
    return std::nullopt;
  }
  return dense.getSplatValue<llvm::APFloat>().convertToFloat();
}

uint32_t f32Bits(float value) {
  return llvm::APFloat(value).bitcastToAPInt().getZExtValue();
}

bool isF32Splat(mlir::Value value, float expected) {
  auto bits = f32SplatBits(value);
  if (!bits) {
    return false;
  }
  return *bits == llvm::APFloat(expected).bitcastToAPInt().getZExtValue();
}

bool reduceBodyIsAdd(mlir::stablehlo::ReduceOp reduce) {
  auto &body = reduce.getBody();
  if (body.empty() || body.getBlocks().size() != 1) {
    return false;
  }
  mlir::Operation *reducer = nullptr;
  mlir::Operation *terminator = nullptr;
  for (mlir::Operation &op : body.front()) {
    if (mlir::isa<mlir::stablehlo::ReturnOp>(op)) {
      terminator = &op;
      continue;
    }
    if (reducer) {
      return false;
    }
    reducer = &op;
  }
  auto ret = mlir::dyn_cast_or_null<mlir::stablehlo::ReturnOp>(terminator);
  return reducer && mlir::isa<mlir::stablehlo::AddOp>(reducer) && ret &&
         ret.getNumOperands() == 1 &&
         ret.getOperand(0).getDefiningOp() == reducer;
}

std::optional<mlir::Value> matchBf16ToF32(mlir::Value value) {
  value = peelIdentityCustomCalls(value);
  auto convert = value.getDefiningOp<mlir::stablehlo::ConvertOp>();
  if (!convert || !isStaticF32Tensor(convert.getResult()) ||
      !isStaticBf16Tensor(convert.getOperand())) {
    return std::nullopt;
  }
  return convert.getOperand();
}

std::optional<NormalizedInput> matchNormalizedInput(mlir::Value value) {
  value = peelIdentityAndReshape(value);
  if (auto input = matchBf16ToF32(value)) {
    return NormalizedInput{value, *input};
  }

  auto subtract = value.getDefiningOp<mlir::stablehlo::SubtractOp>();
  if (!subtract || !isF32Splat(subtract.getRhs(), 0.0f)) {
    return std::nullopt;
  }
  auto input = matchBf16ToF32(subtract.getLhs());
  if (!input) {
    return std::nullopt;
  }
  return NormalizedInput{peelIdentityAndReshape(subtract.getLhs()), *input};
}

bool sameValueAfterShapePeeling(mlir::Value lhs, mlir::Value rhs) {
  return peelIdentityAndReshape(lhs) == peelIdentityAndReshape(rhs);
}

bool sameF32Value(mlir::Value lhs, mlir::Value rhs) {
  if (sameValueAfterShapePeeling(lhs, rhs)) {
    return true;
  }
  auto lhsInput = matchBf16ToF32(lhs);
  auto rhsInput = matchBf16ToF32(rhs);
  return lhsInput && rhsInput &&
         sameValueAfterShapePeeling(*lhsInput, *rhsInput);
}

std::optional<mlir::Value> matchSquareInput(mlir::Value value) {
  value = peelIdentityAndReshape(value);
  if (auto mul = value.getDefiningOp<mlir::stablehlo::MulOp>()) {
    mlir::Value lhs = peelIdentityAndReshape(mul.getLhs());
    mlir::Value rhs = peelIdentityAndReshape(mul.getRhs());
    if (sameF32Value(lhs, rhs)) {
      return lhs;
    }
  }
  if (auto power = value.getDefiningOp<mlir::stablehlo::PowOp>()) {
    if (isF32Splat(power.getRhs(), 2.0f)) {
      return peelIdentityAndReshape(power.getLhs());
    }
  }
  return std::nullopt;
}

std::optional<mlir::stablehlo::ReduceOp> definingReduce(mlir::Value value) {
  value = peelScaleShapeOps(value);
  return value.getDefiningOp<mlir::stablehlo::ReduceOp>();
}

std::optional<mlir::stablehlo::RsqrtOp> definingRsqrt(mlir::Value value) {
  value = peelScaleShapeOps(value);
  return value.getDefiningOp<mlir::stablehlo::RsqrtOp>();
}

std::optional<std::pair<uint32_t, uint32_t>>
matchRmsNormScaleAndBias(mlir::stablehlo::RsqrtOp rsqrt, mlir::Value xF32,
                         int64_t hidden) {
  auto add = peelScaleShapeOps(rsqrt.getOperand())
                 .getDefiningOp<mlir::stablehlo::AddOp>();
  if (!add) {
    return std::nullopt;
  }

  mlir::Value scaledMean;
  std::optional<uint32_t> biasBits = f32SplatBits(add.getRhs());
  if (biasBits) {
    scaledMean = add.getLhs();
  } else {
    biasBits = f32SplatBits(add.getLhs());
    if (!biasBits) {
      return std::nullopt;
    }
    scaledMean = add.getRhs();
  }

  mlir::Value reduceValue;
  std::optional<uint32_t> scaleBits;
  if (auto scaleMul = peelScaleShapeOps(scaledMean)
                          .getDefiningOp<mlir::stablehlo::MulOp>()) {
    scaleBits = f32SplatBits(scaleMul.getLhs());
    if (scaleBits) {
      reduceValue = scaleMul.getRhs();
    } else {
      scaleBits = f32SplatBits(scaleMul.getRhs());
      if (!scaleBits) {
        return std::nullopt;
      }
      reduceValue = scaleMul.getLhs();
    }
  } else if (auto scaleDiv = peelScaleShapeOps(scaledMean)
                                 .getDefiningOp<mlir::stablehlo::DivOp>()) {
    auto denominator = f32SplatValue(scaleDiv.getRhs());
    if (!denominator || *denominator == 0.0f) {
      return std::nullopt;
    }
    scaleBits = f32Bits(1.0f / *denominator);
    reduceValue = scaleDiv.getLhs();
  } else {
    return std::nullopt;
  }
  if (!scaleBits) {
    return std::nullopt;
  }

  auto reduce = definingReduce(reduceValue);
  if (!reduce || reduce->getInputs().size() != 1 ||
      reduce->getInitValues().size() != 1 || reduce->getNumResults() != 1 ||
      reduce->getDimensions().size() != 1 || !reduceBodyIsAdd(*reduce)) {
    return std::nullopt;
  }
  auto reduceInputType = staticRankedTensor(reduce->getInputs()[0]);
  if (!reduceInputType || reduceInputType->getRank() < 1 ||
      reduce->getDimensions()[0] != reduceInputType->getRank() - 1 ||
      reduceInputType->getDimSize(reduceInputType->getRank() - 1) != hidden) {
    return std::nullopt;
  }
  auto squareInput = matchSquareInput(reduce->getInputs()[0]);
  if (!squareInput || !sameF32Value(*squareInput, xF32)) {
    return std::nullopt;
  }

  return std::pair{*scaleBits, *biasBits};
}

std::optional<mlir::Value> matchGamma(mlir::Value value, int64_t hidden) {
  value = peelScaleShapeOps(value);
  auto type = staticRankedTensor(value);
  if (!type || !type->getElementType().isBF16() || type->getRank() != 1 ||
      type->getDimSize(0) != hidden) {
    return std::nullopt;
  }
  return value;
}

std::optional<mlir::Value> matchF32Gamma(mlir::Value value, int64_t hidden) {
  value = peelScaleShapeOps(value);
  if (auto convert = value.getDefiningOp<mlir::stablehlo::ConvertOp>()) {
    if (!isStaticF32Tensor(convert.getResult()) ||
        !isStaticBf16Tensor(convert.getOperand())) {
      return std::nullopt;
    }
    return matchGamma(convert.getOperand(), hidden);
  }
  return matchGamma(value, hidden);
}

std::optional<ScaleAndWeight> matchScaleAndWeight(mlir::Value value,
                                                  int64_t hidden) {
  value = peelScaleShapeOps(value);
  auto mul = value.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!mul) {
    return std::nullopt;
  }
  for (auto [candidateScale, candidateWeight] :
       {std::pair{mul.getLhs(), mul.getRhs()},
        std::pair{mul.getRhs(), mul.getLhs()}}) {
    auto rsqrt = definingRsqrt(candidateScale);
    auto weight = matchF32Gamma(candidateWeight, hidden);
    if (rsqrt && weight) {
      return ScaleAndWeight{*rsqrt, *weight};
    }
  }
  return std::nullopt;
}

std::optional<RmsNormComponents>
matchRmsNorm(mlir::stablehlo::ConvertOp rootConvert) {
  if (rootConvert->getParentOfType<mlir::stablehlo::CaseOp>()) {
    return std::nullopt;
  }
  auto outputType = staticRankedTensor(rootConvert.getResult());
  if (!outputType || !outputType->getElementType().isBF16() ||
      outputType->getRank() < 2) {
    return std::nullopt;
  }
  if (!isStaticF32Tensor(rootConvert.getOperand())) {
    return std::nullopt;
  }
  int64_t rows = outputType->getDimSize(outputType->getRank() - 2);
  if (rows != 1) {
    return std::nullopt;
  }
  int64_t hidden = outputType->getDimSize(outputType->getRank() - 1);

  auto rootMul = peelIdentityCustomCalls(rootConvert.getOperand())
                     .getDefiningOp<mlir::stablehlo::MulOp>();
  if (!rootMul) {
    return std::nullopt;
  }
  for (auto [candidateInput, candidateScale] :
       {std::pair{rootMul.getLhs(), rootMul.getRhs()},
        std::pair{rootMul.getRhs(), rootMul.getLhs()}}) {
    auto input = matchNormalizedInput(candidateInput);
    auto scaleAndWeight = matchScaleAndWeight(candidateScale, hidden);
    if (!input || !scaleAndWeight) {
      continue;
    }
    auto scaleAndBias =
        matchRmsNormScaleAndBias(scaleAndWeight->rsqrt, input->f32, hidden);
    if (!scaleAndBias) {
      continue;
    }
    auto inputType = staticRankedTensor(input->bf16);
    if (!inputType || inputType->getShape() != outputType->getShape()) {
      continue;
    }
    return RmsNormComponents{
        input->bf16,
        scaleAndWeight->weight,
        scaleAndBias->first,
        scaleAndBias->second,
    };
  }
  return std::nullopt;
}

mlir::LogicalResult createRmsNormOp(mlir::PatternRewriter &rewriter,
                                    mlir::stablehlo::ConvertOp root,
                                    const RmsNormComponents &components) {
  rewriter.setInsertionPoint(root);
  auto backendConfig = rewriter.getDictionaryAttr({
      rewriter.getNamedAttr(
          "scale_bits",
          rewriter.getI64IntegerAttr(static_cast<int64_t>(components.scaleBits))),
      rewriter.getNamedAttr(
          "bias_bits",
          rewriter.getI64IntegerAttr(static_cast<int64_t>(components.biasBits))),
  });
  auto customCall = rewriter.create<mlir::stablehlo::CustomCallOp>(
      root.getLoc(), root->getResultTypes(),
      mlir::ValueRange{components.input, components.weight}, kRmsNormTarget,
      /*hasSideEffect=*/false, backendConfig,
      mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI,
      rewriter.getArrayAttr({}),
      /*calledComputations=*/nullptr,
      /*operandLayouts=*/nullptr,
      /*resultLayouts=*/nullptr);
  rewriter.replaceOp(root, customCall.getResults());
  return mlir::success();
}

} // namespace

mlir::LogicalResult
RMSNormFusing::matchAndRewrite(mlir::stablehlo::ConvertOp convertOp,
                               mlir::PatternRewriter &rewriter) const {
  auto components = matchRmsNorm(convertOp);
  if (!components) {
    return mlir::failure();
  }
  return createRmsNormOp(rewriter, convertOp, *components);
}

} // namespace libtt::mlir_frontend
