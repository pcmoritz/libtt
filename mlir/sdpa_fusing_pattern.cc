#include "mlir/sdpa_fusing_pattern.h"

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
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

struct MaskSelectMatch {
  mlir::Value pred;
  mlir::Value unmaskedValue;
};

struct DecodeMaskMatch {
  mlir::Value scaledScores;
  mlir::Value seqLens;
  mlir::Value loc;
};

struct Components {
  mlir::Value q;
  mlir::Value k;
  mlir::Value v;
  mlir::Value fusedKvCache;
  mlir::Value seqLens;
  mlir::Value loc;
  uint32_t scaleBf16Packed = 0;
  bool useFusedKvCache = false;
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

bool isNegativeBf16Splat(mlir::Value value) {
  auto packed = bf16PackedConstant(value);
  return packed && ((*packed & 0x8000u) != 0);
}

std::optional<mlir::Value> peelSingleUseIdentityCustomCalls(mlir::Value value) {
  while (auto customCall =
             value.getDefiningOp<mlir::stablehlo::CustomCallOp>()) {
    if (!isIdentityCustomCall(customCall)) {
      break;
    }
    if (!value.hasOneUse()) {
      return std::nullopt;
    }
    value = customCall.getInputs().front();
  }
  if (!value.hasOneUse()) {
    return std::nullopt;
  }
  return value;
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

std::optional<int64_t> i32SplatConstant(mlir::Value value) {
  value = peelIdentityCustomCalls(value);
  while (auto broadcastOp =
             value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
    value = peelIdentityCustomCalls(broadcastOp.getOperand());
  }
  auto constantOp = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!constantOp) {
    return std::nullopt;
  }
  auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constantOp.getValue());
  if (!dense || !dense.isSplat()) {
    return std::nullopt;
  }
  auto integer = mlir::dyn_cast<mlir::IntegerType>(dense.getElementType());
  if (!integer || integer.getWidth() != 32 || integer.isUnsigned()) {
    return std::nullopt;
  }
  return dense.getSplatValue<llvm::APInt>().getSExtValue();
}

bool isIotaDim0(mlir::Value value, int64_t length) {
  value = peelIdentityCustomCalls(value);
  auto iotaOp = value.getDefiningOp<mlir::stablehlo::IotaOp>();
  auto type = getStaticRankedTensor(value);
  return iotaOp && type && type->getRank() == 1 &&
         type->getDimSize(0) == length && iotaOp.getIotaDimension() == 0;
}

bool isIotaTimesTwo(mlir::Value value, int64_t length) {
  value = peelIdentityCustomCalls(value);
  auto multiplyOp = value.getDefiningOp<mlir::stablehlo::MulOp>();
  if (!multiplyOp) {
    return false;
  }
  auto lhsConstant = i32SplatConstant(multiplyOp.getLhs());
  if (lhsConstant && *lhsConstant == 2 &&
      isIotaDim0(multiplyOp.getRhs(), length)) {
    return true;
  }
  auto rhsConstant = i32SplatConstant(multiplyOp.getRhs());
  return rhsConstant && *rhsConstant == 2 &&
         isIotaDim0(multiplyOp.getLhs(), length);
}

std::optional<int64_t> strideTwoGatherOffset(mlir::Value startIndices,
                                             int64_t length) {
  startIndices = peelIdentityCustomCalls(startIndices);
  auto broadcastOp =
      startIndices.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  auto broadcastType = getStaticRankedTensor(startIndices);
  if (!broadcastOp || !broadcastType || broadcastType->getRank() != 2 ||
      broadcastType->getDimSize(0) != length ||
      broadcastType->getDimSize(1) != 1 ||
      broadcastOp.getBroadcastDimensions() != llvm::ArrayRef<int64_t>{0}) {
    return std::nullopt;
  }

  mlir::Value vector = peelIdentityCustomCalls(broadcastOp.getOperand());
  if (isIotaTimesTwo(vector, length)) {
    return 0;
  }
  auto addOp = vector.getDefiningOp<mlir::stablehlo::AddOp>();
  if (!addOp) {
    return std::nullopt;
  }
  auto lhsConstant = i32SplatConstant(addOp.getLhs());
  if (lhsConstant && isIotaTimesTwo(addOp.getRhs(), length)) {
    return *lhsConstant;
  }
  auto rhsConstant = i32SplatConstant(addOp.getRhs());
  if (rhsConstant && isIotaTimesTwo(addOp.getLhs(), length)) {
    return *rhsConstant;
  }
  return std::nullopt;
}

bool isInterleavedKvSplitGather(mlir::stablehlo::GatherOp gatherOp,
                                int64_t expectedOffset) {
  auto resultType = getStaticRankedTensor(gatherOp.getResult());
  auto operandType = getStaticRankedTensor(gatherOp.getOperand());
  if (!resultType || !operandType || resultType->getRank() != 3 ||
      operandType->getRank() != 3 ||
      !resultType->getElementType().isBF16() ||
      !operandType->getElementType().isBF16()) {
    return false;
  }
  int64_t cacheTokens = resultType->getDimSize(0);
  int64_t kvHeads = resultType->getDimSize(1);
  int64_t headDim = resultType->getDimSize(2);
  if (cacheTokens <= 0 || kvHeads <= 0 || headDim <= 0 ||
      operandType->getDimSize(0) != cacheTokens ||
      operandType->getDimSize(1) != kvHeads * 2 ||
      operandType->getDimSize(2) != headDim) {
    return false;
  }

  auto dims = gatherOp.getDimensionNumbers();
  if (dims.getOffsetDims() != llvm::ArrayRef<int64_t>{0, 2} ||
      dims.getCollapsedSliceDims() != llvm::ArrayRef<int64_t>{1} ||
      dims.getStartIndexMap() != llvm::ArrayRef<int64_t>{1} ||
      dims.getIndexVectorDim() != 1) {
    return false;
  }
  auto sliceSizes = gatherOp.getSliceSizes();
  if (sliceSizes.size() != 3 || sliceSizes[0] != cacheTokens ||
      sliceSizes[1] != 1 || sliceSizes[2] != headDim) {
    return false;
  }
  auto offset = strideTwoGatherOffset(gatherOp.getStartIndices(), kvHeads);
  return offset && *offset == expectedOffset;
}

std::optional<mlir::Value> fusedCacheFromInterleavedKvSplits(
    mlir::Value kCache,
    mlir::Value vCache) {
  kCache = peelIdentityCustomCalls(kCache);
  vCache = peelIdentityCustomCalls(vCache);
  auto kGather = kCache.getDefiningOp<mlir::stablehlo::GatherOp>();
  auto vGather = vCache.getDefiningOp<mlir::stablehlo::GatherOp>();
  if (!kGather || !vGather ||
      !isInterleavedKvSplitGather(kGather, /*expectedOffset=*/0) ||
      !isInterleavedKvSplitGather(vGather, /*expectedOffset=*/1)) {
    return std::nullopt;
  }

  mlir::Value kOperand = peelIdentityCustomCalls(kGather.getOperand());
  mlir::Value vOperand = peelIdentityCustomCalls(vGather.getOperand());
  if (kOperand != vOperand) {
    return std::nullopt;
  }
  auto reshapeOp = kOperand.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshapeOp) {
    return std::nullopt;
  }
  auto splitType = getStaticRankedTensor(kOperand);
  auto fusedType = getStaticRankedTensor(reshapeOp.getOperand());
  if (!splitType || !fusedType || fusedType->getRank() != 5 ||
      !fusedType->getElementType().isBF16()) {
    return std::nullopt;
  }
  int64_t cacheTokens = splitType->getDimSize(0);
  int64_t kvHeads = splitType->getDimSize(1) / 2;
  int64_t headDim = splitType->getDimSize(2);
  if (fusedType->getDimSize(0) * fusedType->getDimSize(1) != cacheTokens ||
      fusedType->getDimSize(2) != kvHeads ||
      fusedType->getDimSize(3) != 2 ||
      fusedType->getDimSize(4) != headDim) {
    return std::nullopt;
  }
  return reshapeOp.getOperand();
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

std::optional<MaskSelectMatch> matchScoreMaskSelect(mlir::Value value) {
  value = peelIdentityCustomCalls(value);
  auto selectOp = value.getDefiningOp<mlir::stablehlo::SelectOp>();
  if (!selectOp || !isNegativeBf16Splat(selectOp.getOnFalse())) {
    return std::nullopt;
  }
  auto unmaskedValue = peelSingleUseIdentityCustomCalls(selectOp.getOnTrue());
  if (!unmaskedValue) {
    return std::nullopt;
  }
  return MaskSelectMatch{selectOp.getPred(), *unmaskedValue};
}

std::optional<DecodeMaskMatch> matchDecodeScoreMasks(mlir::Value maskedScores,
                                                     int64_t keyTokens) {
  // Decode attention masks are nested as:
  //   select(seq_len_mask, select(loc_mask, scaled_scores, -large), -large)
  auto sequenceMask = matchScoreMaskSelect(maskedScores);
  if (!sequenceMask) {
    return std::nullopt;
  }
  auto seqLens = findUniqueS32TensorWithLength(sequenceMask->pred, 1);
  if (!seqLens) {
    return std::nullopt;
  }

  auto locationMask = matchScoreMaskSelect(sequenceMask->unmaskedValue);
  if (!locationMask) {
    return std::nullopt;
  }
  auto loc = findUniqueS32TensorWithLength(locationMask->pred, keyTokens);
  if (!loc) {
    return std::nullopt;
  }

  return DecodeMaskMatch{locationMask->unmaskedValue, *seqLens, *loc};
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
  // This only peels the softmax arithmetic shell. The surrounding dot/gather
  // matcher validates that the exposed value is really an attention score path.
  auto divValue = peelSingleUseIdentityCustomCalls(probabilities);
  if (!divValue) {
    return std::nullopt;
  }
  auto divOp =
      divValue->getDefiningOp<mlir::stablehlo::DivOp>();
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
  auto masks = matchDecodeScoreMasks(maskedScores, keyTokens);
  if (!masks) {
    return std::nullopt;
  }

  auto multiplyOp =
      masks->scaledScores.getDefiningOp<mlir::stablehlo::MulOp>();
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
  auto singleUseScoreTiles = peelSingleUseIdentityCustomCalls(scoreTiles);
  if (!singleUseScoreTiles) {
    return std::nullopt;
  }
  scoreTiles = *singleUseScoreTiles;

  auto transposeOp = scoreTiles.getDefiningOp<mlir::stablehlo::TransposeOp>();
  auto scoreType = getStaticRankedTensor(scoreTiles);
  if (!transposeOp || !scoreType || scoreType->getRank() != 3 ||
      scoreType->getDimSize(0) != 1 || scoreType->getDimSize(1) != qHeads ||
      scoreType->getDimSize(2) != keyTokens) {
    return std::nullopt;
  }

  auto dotValue = peelSingleUseIdentityCustomCalls(transposeOp.getOperand());
  if (!dotValue) {
    return std::nullopt;
  }
  auto dotOp = dotValue->getDefiningOp<mlir::stablehlo::DotGeneralOp>();
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
      !isStaticBf16Tensor(qValue) || kMatch->loc != masks->loc) {
    return std::nullopt;
  }

  return ScorePathMatch{qValue, kMatch->cache, masks->seqLens, masks->loc,
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

  auto valueDotValue =
      peelSingleUseIdentityCustomCalls(transposeOp.getOperand());
  if (!valueDotValue) {
    return std::nullopt;
  }
  auto valueDot =
      valueDotValue->getDefiningOp<mlir::stablehlo::DotGeneralOp>();
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
  auto fusedKvCache =
      fusedCacheFromInterleavedKvSplits(scoreMatch->k, vMatch->cache);

  auto qType = getStaticRankedTensor(scoreMatch->q);
  if (!qType || qType->getRank() != 3 || qType->getDimSize(0) != 1 ||
      qType->getDimSize(1) != qHeads || qType->getDimSize(2) != headDim) {
    return std::nullopt;
  }

  return Components{scoreMatch->q,
                    scoreMatch->k,
                    vMatch->cache,
                    fusedKvCache.value_or(mlir::Value{}),
                    scoreMatch->seqLens,
                    vMatch->loc,
                    scoreMatch->scaleBf16Packed,
                    fusedKvCache.has_value()};
}

mlir::LogicalResult createSdpaDecodeOp(mlir::PatternRewriter &rewriter,
                                       mlir::stablehlo::TransposeOp root,
                                       const Components &components) {
  rewriter.setInsertionPoint(root);
  auto backendConfig =
      rewriter.getStringAttr(std::to_string(components.scaleBf16Packed));
  llvm::SmallVector<mlir::Value, 5> inputs;
  if (components.useFusedKvCache) {
    inputs.push_back(components.q);
    inputs.push_back(components.fusedKvCache);
    inputs.push_back(components.seqLens);
    inputs.push_back(components.loc);
  } else {
    inputs.push_back(components.q);
    inputs.push_back(components.k);
    inputs.push_back(components.v);
    inputs.push_back(components.seqLens);
    inputs.push_back(components.loc);
  }
  auto customCall = rewriter.create<mlir::stablehlo::CustomCallOp>(
      root.getLoc(), root->getResultTypes(),
      inputs,
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
