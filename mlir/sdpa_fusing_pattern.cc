#include "mlir/sdpa_fusing_pattern.h"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/stablehlo_utils.h"

namespace libtt::mlir_frontend {
namespace {

struct RepeatedCacheMatch {
  mlir::Value cache;
  mlir::Value loc;
  int64_t cacheTokens = 0;
  int64_t kvHeads = 0;
  int64_t headDim = 0;
  int64_t keyTokens = 0;
};

struct ScorePathMatch {
  mlir::Value q;
  mlir::Value k;
  mlir::Value seqLens;
  mlir::Value loc;
  uint32_t scaleBf16Packed = 0;
};

struct Components {
  mlir::Value q;
  mlir::Value k;
  mlir::Value v;
  mlir::Value seqLens;
  mlir::Value loc;
  uint32_t scaleBf16Packed = 0;
};

std::optional<mlir::RankedTensorType> getStaticRankedTensor(mlir::Value value) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensor || !tensor.hasStaticShape()) {
    return std::nullopt;
  }
  return tensor;
}

bool isStaticBf16Tensor(mlir::Value value) {
  auto tensor = getStaticRankedTensor(value);
  return tensor && tensor->getElementType().isBF16();
}

bool isS32TensorWithLength(mlir::Value value, int64_t length) {
  auto tensor = getStaticRankedTensor(value);
  if (!tensor || tensor->getRank() != 1 || tensor->getDimSize(0) != length) {
    return false;
  }
  auto integer = mlir::dyn_cast<mlir::IntegerType>(tensor->getElementType());
  return integer && integer.getWidth() == 32 && !integer.isUnsigned();
}

std::optional<uint32_t> bf16PackedConstant(mlir::Value value) {
  value = peelIdentityCustomCalls(value);
  while (auto broadcastOp =
             value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    value = peelIdentityCustomCalls(broadcastOp.getOperand());
  }
  if (!isStaticBf16Tensor(value)) {
    return std::nullopt;
  }

  auto constantOp = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!constantOp) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constantOp.getValue());
  if (!dense || !dense.isSplat() || !dense.getElementType().isBF16()) {
    return std::nullopt;
  }

  auto bits = dense.getSplatValue<llvm::APFloat>().bitcastToAPInt();
  uint32_t value16 = bits.extractBitsAsZExtValue(16, 0);
  return value16 | (value16 << 16);
}

bool findUniqueS32TensorWithLength(mlir::Value value, int64_t length,
                                   llvm::DenseSet<mlir::Value> &visited,
                                   std::optional<mlir::Value> &match) {
  if (!visited.insert(value).second) {
    return true;
  }
  if (mlir::isa<mlir::BlockArgument>(value) &&
      isS32TensorWithLength(value, length)) {
    if (match && *match != value) {
      return false;
    }
    match = value;
    return true;
  }

  mlir::Operation *op = value.getDefiningOp();
  if (!op) {
    return true;
  }
  for (mlir::Value operand : op->getOperands()) {
    if (!findUniqueS32TensorWithLength(operand, length, visited, match)) {
      return false;
    }
  }
  return true;
}

std::optional<mlir::Value> findS32TensorWithLength(
    mlir::Value value, int64_t length, llvm::DenseSet<mlir::Value> &visited) {
  if (!visited.insert(value).second) {
    return std::nullopt;
  }
  if (mlir::isa<mlir::BlockArgument>(value) &&
      isS32TensorWithLength(value, length)) {
    return value;
  }

  mlir::Operation *op = value.getDefiningOp();
  if (!op) {
    return std::nullopt;
  }
  for (mlir::Value operand : op->getOperands()) {
    if (auto match = findS32TensorWithLength(operand, length, visited)) {
      return match;
    }
  }
  return std::nullopt;
}

std::optional<mlir::Value> findS32TensorWithLength(mlir::Value value,
                                                   int64_t length) {
  llvm::DenseSet<mlir::Value> visited;
  return findS32TensorWithLength(value, length, visited);
}

std::optional<mlir::Value> findUniqueS32TensorWithLength(mlir::Value value,
                                                         int64_t length) {
  llvm::DenseSet<mlir::Value> visited;
  std::optional<mlir::Value> match;
  if (!findUniqueS32TensorWithLength(value, length, visited, match)) {
    return std::nullopt;
  }
  return match;
}

std::optional<mlir::stablehlo::GatherOp> gatherFromCacheValue(
    mlir::Value value) {
  value = peelIdentityCustomCalls(value);
  if (auto gatherOp = value.getDefiningOp<mlir::stablehlo::GatherOp>()) {
    return gatherOp;
  }
  if (auto selectOp = value.getDefiningOp<mlir::stablehlo::SelectOp>()) {
    mlir::Value onTrue = peelIdentityCustomCalls(selectOp.getOnTrue());
    if (auto gatherOp = onTrue.getDefiningOp<mlir::stablehlo::GatherOp>()) {
      return gatherOp;
    }
  }
  return std::nullopt;
}

std::optional<RepeatedCacheMatch> matchRepeatedCache(mlir::Value value,
                                                     int64_t qHeads) {
  value = peelIdentityCustomCalls(value);
  auto reshapeOp = value.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshapeOp) {
    return std::nullopt;
  }
  auto reshapeType = getStaticRankedTensor(reshapeOp.getResult());
  if (!reshapeType || reshapeType->getRank() != 3 ||
      reshapeType->getDimSize(1) != qHeads) {
    return std::nullopt;
  }
  int64_t keyTokens = reshapeType->getDimSize(0);
  int64_t headDim = reshapeType->getDimSize(2);

  auto broadcastOp =
      definingOpSkippingIdentityCustomCalls<mlir::stablehlo::BroadcastInDimOp>(
          reshapeOp.getOperand());
  if (!broadcastOp) {
    return std::nullopt;
  }
  auto broadcastType = getStaticRankedTensor(broadcastOp.getResult());
  if (!broadcastType || broadcastType->getRank() != 4 ||
      broadcastType->getDimSize(0) != keyTokens ||
      broadcastType->getDimSize(3) != headDim) {
    return std::nullopt;
  }

  mlir::Value gatheredValue = peelIdentityCustomCalls(broadcastOp.getOperand());
  if (broadcastOp.getBroadcastDimensions() ==
      llvm::ArrayRef<int64_t>{0, 1, 2, 3}) {
    auto expandOp =
        definingOpSkippingIdentityCustomCalls<mlir::stablehlo::BroadcastInDimOp>(
            gatheredValue);
    auto expandType = getStaticRankedTensor(gatheredValue);
    if (!expandOp || !expandType || expandType->getRank() != 4 ||
        expandType->getDimSize(0) != keyTokens ||
        expandType->getDimSize(2) != 1 ||
        expandType->getDimSize(3) != headDim ||
        expandOp.getBroadcastDimensions() != llvm::ArrayRef<int64_t>{0, 1, 3}) {
      return std::nullopt;
    }
    gatheredValue = peelIdentityCustomCalls(expandOp.getOperand());
  } else if (broadcastOp.getBroadcastDimensions() !=
             llvm::ArrayRef<int64_t>{0, 1, 3}) {
    return std::nullopt;
  }

  int64_t kvHeads = broadcastType->getDimSize(1);
  int64_t repeat = broadcastType->getDimSize(2);
  if (kvHeads <= 0 || repeat <= 0 || kvHeads * repeat != qHeads) {
    return std::nullopt;
  }

  auto gatherMatch = gatherFromCacheValue(gatheredValue);
  if (!gatherMatch) {
    return std::nullopt;
  }
  auto gatherOp = *gatherMatch;
  auto gatherType = getStaticRankedTensor(gatherOp.getResult());
  auto cacheType = getStaticRankedTensor(gatherOp.getOperand());
  if (!gatherType || !cacheType || gatherType->getRank() != 3 ||
      cacheType->getRank() != 3 || gatherType->getDimSize(0) != keyTokens ||
      gatherType->getDimSize(1) != kvHeads ||
      gatherType->getDimSize(2) != headDim ||
      cacheType->getDimSize(1) != kvHeads ||
      cacheType->getDimSize(2) != headDim ||
      !isStaticBf16Tensor(gatherOp.getOperand())) {
    return std::nullopt;
  }

  auto loc = findS32TensorWithLength(gatherOp.getStartIndices(), keyTokens);
  if (!loc) {
    return std::nullopt;
  }
  return RepeatedCacheMatch{
      gatherOp.getOperand(), *loc, cacheType->getDimSize(0), kvHeads, headDim,
      keyTokens};
}

std::optional<mlir::Value> peelSoftmaxInput(mlir::Value probabilities) {
  auto divOp =
      definingOpSkippingIdentityCustomCalls<mlir::stablehlo::DivOp>(
          probabilities);
  if (!divOp) {
    return std::nullopt;
  }
  auto expOp =
      definingOpSkippingIdentityCustomCalls<mlir::stablehlo::ExpOp>(
          divOp.getLhs());
  if (!expOp) {
    return std::nullopt;
  }
  auto subtractOp =
      definingOpSkippingIdentityCustomCalls<mlir::stablehlo::SubtractOp>(
          expOp.getOperand());
  if (!subtractOp) {
    return std::nullopt;
  }
  return peelIdentityCustomCalls(subtractOp.getLhs());
}

std::optional<ScorePathMatch> matchScorePath(mlir::Value maskedScores,
                                             int64_t qHeads,
                                             int64_t keyTokens) {
  mlir::Value scaledScores = peelIdentityCustomCalls(maskedScores);
  auto outerMask = scaledScores.getDefiningOp<mlir::stablehlo::SelectOp>();
  if (!outerMask) {
    return std::nullopt;
  }

  auto seqLens = findUniqueS32TensorWithLength(outerMask.getPred(), 1);
  if (!seqLens) {
    return std::nullopt;
  }

  for (unsigned depth = 0; depth < 4; ++depth) {
    auto selectOp = scaledScores.getDefiningOp<mlir::stablehlo::SelectOp>();
    if (!selectOp) {
      break;
    }
    scaledScores = peelIdentityCustomCalls(selectOp.getOnTrue());
  }

  auto multiplyOp = scaledScores.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!multiplyOp) {
    return std::nullopt;
  }

  mlir::Value scoreTiles;
  std::optional<uint32_t> scaleBf16Packed =
      bf16PackedConstant(multiplyOp.getLhs());
  if (scaleBf16Packed) {
    scoreTiles = multiplyOp.getRhs();
  } else {
    scaleBf16Packed = bf16PackedConstant(multiplyOp.getRhs());
    if (!scaleBf16Packed) {
      return std::nullopt;
    }
    scoreTiles = multiplyOp.getLhs();
  }
  scoreTiles = peelIdentityCustomCalls(scoreTiles);

  auto transposeOp = scoreTiles.getDefiningOp<mlir::stablehlo::TransposeOp>();
  auto scoreType = getStaticRankedTensor(scoreTiles);
  if (!transposeOp || !scoreType || scoreType->getRank() != 3 ||
      scoreType->getDimSize(0) != 1 || scoreType->getDimSize(1) != qHeads ||
      scoreType->getDimSize(2) != keyTokens) {
    return std::nullopt;
  }

  auto dotOp =
      definingOpSkippingIdentityCustomCalls<mlir::stablehlo::DotGeneralOp>(
          transposeOp.getOperand());
  if (!dotOp) {
    return std::nullopt;
  }
  auto dims = dotOp.getDotDimensionNumbers();
  if (dims.getLhsBatchingDimensions() != llvm::ArrayRef<int64_t>{1} ||
      dims.getRhsBatchingDimensions() != llvm::ArrayRef<int64_t>{1} ||
      dims.getLhsContractingDimensions() != llvm::ArrayRef<int64_t>{2} ||
      dims.getRhsContractingDimensions() != llvm::ArrayRef<int64_t>{2}) {
    return std::nullopt;
  }

  auto lhsKMatch = matchRepeatedCache(dotOp.getLhs(), qHeads);
  auto rhsKMatch = matchRepeatedCache(dotOp.getRhs(), qHeads);
  mlir::Value qValue;
  std::optional<RepeatedCacheMatch> kMatch;
  if (lhsKMatch && !rhsKMatch) {
    qValue = dotOp.getRhs();
    kMatch = *lhsKMatch;
  } else if (rhsKMatch && !lhsKMatch) {
    qValue = dotOp.getLhs();
    kMatch = *rhsKMatch;
  } else {
    return std::nullopt;
  }

  auto qType = getStaticRankedTensor(qValue);
  if (!kMatch || kMatch->keyTokens != keyTokens || !qType ||
      qType->getRank() != 3 || qType->getDimSize(0) != 1 ||
      qType->getDimSize(1) != qHeads ||
      qType->getDimSize(2) != kMatch->headDim ||
      !isStaticBf16Tensor(qValue)) {
    return std::nullopt;
  }

  return ScorePathMatch{qValue, kMatch->cache, *seqLens, kMatch->loc,
                        *scaleBf16Packed};
}

std::optional<Components> matchSdpaDecode(
    mlir::stablehlo::TransposeOp transposeOp) {
  if (transposeOp->getParentOfType<mlir::stablehlo::CaseOp>()) {
    return std::nullopt;
  }

  auto outputType = getStaticRankedTensor(transposeOp.getResult());
  if (!outputType || outputType->getRank() != 3 ||
      outputType->getDimSize(0) != 1 ||
      !outputType->getElementType().isBF16()) {
    return std::nullopt;
  }
  int64_t qHeads = outputType->getDimSize(1);
  int64_t headDim = outputType->getDimSize(2);

  auto valueDot =
      definingOpSkippingIdentityCustomCalls<mlir::stablehlo::DotGeneralOp>(
          transposeOp.getOperand());
  if (!valueDot) {
    return std::nullopt;
  }
  auto dims = valueDot.getDotDimensionNumbers();
  if (dims.getLhsBatchingDimensions() != llvm::ArrayRef<int64_t>{1} ||
      dims.getRhsBatchingDimensions() != llvm::ArrayRef<int64_t>{1}) {
    return std::nullopt;
  }

  mlir::Value probabilities;
  auto lhsVMatch = matchRepeatedCache(valueDot.getLhs(), qHeads);
  auto rhsVMatch = matchRepeatedCache(valueDot.getRhs(), qHeads);
  std::optional<RepeatedCacheMatch> vMatch;
  if (lhsVMatch && !rhsVMatch &&
      dims.getLhsContractingDimensions() == llvm::ArrayRef<int64_t>{0} &&
      dims.getRhsContractingDimensions() == llvm::ArrayRef<int64_t>{2}) {
    probabilities = valueDot.getRhs();
    vMatch = *lhsVMatch;
  } else if (rhsVMatch && !lhsVMatch &&
             dims.getLhsContractingDimensions() == llvm::ArrayRef<int64_t>{2} &&
             dims.getRhsContractingDimensions() == llvm::ArrayRef<int64_t>{0}) {
    probabilities = valueDot.getLhs();
    vMatch = *rhsVMatch;
  } else {
    return std::nullopt;
  }

  if (!vMatch || vMatch->headDim != headDim) {
    return std::nullopt;
  }
  auto probabilitiesType = getStaticRankedTensor(probabilities);
  if (!probabilitiesType || probabilitiesType->getRank() != 3 ||
      probabilitiesType->getDimSize(0) != 1 ||
      probabilitiesType->getDimSize(1) != qHeads ||
      probabilitiesType->getDimSize(2) != vMatch->keyTokens) {
    return std::nullopt;
  }

  auto maskedScores = peelSoftmaxInput(probabilities);
  if (!maskedScores) {
    return std::nullopt;
  }
  auto scoreMatch = matchScorePath(*maskedScores, qHeads, vMatch->keyTokens);
  if (!scoreMatch || scoreMatch->loc != vMatch->loc) {
    return std::nullopt;
  }

  auto qType = getStaticRankedTensor(scoreMatch->q);
  if (!qType || qType->getRank() != 3 || qType->getDimSize(0) != 1 ||
      qType->getDimSize(1) != qHeads || qType->getDimSize(2) != headDim) {
    return std::nullopt;
  }

  return Components{scoreMatch->q,
                    scoreMatch->k,
                    vMatch->cache,
                    scoreMatch->seqLens,
                    vMatch->loc,
                    scoreMatch->scaleBf16Packed};
}

mlir::LogicalResult createSdpaDecodeOp(mlir::PatternRewriter &rewriter,
                                       mlir::stablehlo::TransposeOp root,
                                       const Components &components) {
  rewriter.setInsertionPoint(root);
  auto backendConfig =
      rewriter.getStringAttr(std::to_string(components.scaleBf16Packed));
  auto customCall = rewriter.create<mlir::stablehlo::CustomCallOp>(
      root.getLoc(), root->getResultTypes(),
      mlir::ValueRange{components.q, components.k, components.v,
                       components.seqLens, components.loc},
      kSdpaDecodeTarget,
      /*hasSideEffect=*/false, backendConfig,
      mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr({}),
      /*calledComputations=*/nullptr,
      /*operandLayouts=*/nullptr,
      /*resultLayouts=*/nullptr);
  rewriter.replaceOp(root, customCall.getResults());
  return mlir::success();
}

} // namespace

mlir::LogicalResult SDPADecodeFusing::matchAndRewrite(
    mlir::stablehlo::TransposeOp transposeOp,
    mlir::PatternRewriter &rewriter) const {
  auto components = matchSdpaDecode(transposeOp);
  if (!components) {
    return mlir::failure();
  }
  return createSdpaDecodeOp(rewriter, transposeOp, *components);
}

} // namespace libtt::mlir_frontend
