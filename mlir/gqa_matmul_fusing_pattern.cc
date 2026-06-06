#include "mlir/gqa_matmul_fusing_pattern.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/stablehlo_utils.h"

namespace libtt::mlir_frontend {
namespace {

constexpr llvm::StringLiteral kGroupedHeadDimensionAttr =
    "libtt.grouped_dimension";
constexpr llvm::StringLiteral kGroupedHeadGroupSizeAttr = "libtt.group_size";

std::optional<mlir::RankedTensorType> getStaticRankedTensor(mlir::Value value) {
  auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensor || !tensor.hasStaticShape()) {
    return std::nullopt;
  }
  return tensor;
}

std::optional<GroupedHeadExpansion> matchGroupedHeadExpansion(mlir::Value value) {
  mlir::Value current = value;
  while (auto customCall =
             current.getDefiningOp<mlir::stablehlo::CustomCallOp>()) {
    if (!current.hasOneUse() || !isIdentityCustomCall(customCall)) {
      return std::nullopt;
    }
    current = customCall.getInputs().front();
  }

  auto reshapeOp = current.getDefiningOp<mlir::stablehlo::ReshapeOp>();
  if (!reshapeOp || !current.hasOneUse()) {
    return std::nullopt;
  }

  auto broadcastOp =
      reshapeOp.getOperand().getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!broadcastOp || !broadcastOp.getResult().hasOneUse()) {
    return std::nullopt;
  }

  auto sourceType = getStaticRankedTensor(broadcastOp.getOperand());
  auto broadcastType = getStaticRankedTensor(broadcastOp.getResult());
  auto logicalType = getStaticRankedTensor(reshapeOp.getResult());
  if (!sourceType || !broadcastType || !logicalType ||
      sourceType->getElementType() != broadcastType->getElementType() ||
      sourceType->getElementType() != logicalType->getElementType()) {
    return std::nullopt;
  }

  auto sourceShape = sourceType->getShape();
  auto broadcastShape = broadcastType->getShape();
  auto logicalShape = logicalType->getShape();
  int64_t sourceRank = sourceType->getRank();
  if (sourceRank <= 0 || broadcastType->getRank() != sourceRank + 1 ||
      logicalType->getRank() != sourceRank) {
    return std::nullopt;
  }

  auto broadcastDims = broadcastOp.getBroadcastDimensions();
  if (static_cast<int64_t>(broadcastDims.size()) != sourceRank) {
    return std::nullopt;
  }

  std::vector<bool> mapped(sourceRank + 1, false);
  for (int64_t dim : broadcastDims) {
    if (dim < 0 || dim >= sourceRank + 1 || mapped[dim]) {
      return std::nullopt;
    }
    mapped[dim] = true;
  }

  int64_t insertedDim = -1;
  for (int64_t dim = 0; dim < sourceRank + 1; ++dim) {
    if (!mapped[dim]) {
      insertedDim = dim;
      break;
    }
  }
  if (insertedDim <= 0) {
    return std::nullopt;
  }

  int64_t groupedDim = insertedDim - 1;
  int64_t groupSize = broadcastShape[insertedDim];
  if (groupSize <= 1 || groupSize > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }

  for (int64_t sourceDim = 0; sourceDim < sourceRank; ++sourceDim) {
    int64_t expectedBroadcastDim =
        sourceDim <= groupedDim ? sourceDim : sourceDim + 1;
    if (broadcastDims[sourceDim] != expectedBroadcastDim ||
        broadcastShape[expectedBroadcastDim] != sourceShape[sourceDim]) {
      return std::nullopt;
    }

    int64_t expectedLogical =
        sourceDim == groupedDim ? sourceShape[sourceDim] * groupSize
                                : sourceShape[sourceDim];
    if (logicalShape[sourceDim] != expectedLogical) {
      return std::nullopt;
    }
  }

  return GroupedHeadExpansion{
      .source = broadcastOp.getOperand(),
      .logicalShape =
          llvm::SmallVector<int64_t>(logicalShape.begin(), logicalShape.end()),
      .groupedDimension = static_cast<uint32_t>(groupedDim),
      .groupSize = static_cast<uint32_t>(groupSize),
  };
}

llvm::SmallVector<int64_t> dotFreeDimensions(
    int64_t rank, llvm::ArrayRef<int64_t> batchingDimensions,
    llvm::ArrayRef<int64_t> contractingDimensions) {
  llvm::SmallVector<int64_t> freeDimensions;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (!llvm::is_contained(batchingDimensions, dim) &&
        !llvm::is_contained(contractingDimensions, dim)) {
      freeDimensions.push_back(dim);
    }
  }
  return freeDimensions;
}

bool isPackedGqaLhsExpansion(mlir::stablehlo::DotGeneralOp dotOp,
                             const GroupedHeadExpansion &expansion) {
  auto lhsType = getStaticRankedTensor(dotOp.getLhs());
  auto lhsSourceType = getStaticRankedTensor(expansion.source);
  auto rhsType = getStaticRankedTensor(dotOp.getRhs());
  auto outputType = getStaticRankedTensor(dotOp.getResult());
  if (!lhsType || !lhsSourceType || !rhsType || !outputType) {
    return false;
  }

  auto dims = dotOp.getDotDimensionNumbers();
  auto lhsBatch = dims.getLhsBatchingDimensions();
  auto rhsBatch = dims.getRhsBatchingDimensions();
  auto lhsContract = dims.getLhsContractingDimensions();
  auto rhsContract = dims.getRhsContractingDimensions();
  auto lhsFree =
      dotFreeDimensions(lhsType->getRank(), lhsBatch, lhsContract);
  auto rhsFree =
      dotFreeDimensions(rhsType->getRank(), rhsBatch, rhsContract);

  auto lhsShape = lhsType->getShape();
  auto lhsSourceShape = lhsSourceType->getShape();
  auto rhsShape = rhsType->getShape();
  auto outputShape = outputType->getShape();
  uint32_t groupedDim = expansion.groupedDimension;
  int64_t groupSize = expansion.groupSize;
  if (lhsSourceShape.size() != lhsShape.size() || rhsType->getRank() != 3 ||
      outputType->getRank() != 3 || lhsBatch.size() != 1 ||
      rhsBatch.size() != 1 || lhsFree.size() != 1 || rhsFree.size() != 1 ||
      lhsContract.size() != rhsContract.size() ||
      groupedDim >= lhsShape.size() ||
      groupedDim != static_cast<uint32_t>(lhsBatch[0]) ||
      lhsFree[0] >= static_cast<int64_t>(groupedDim) || rhsBatch[0] != 1 ||
      rhsContract.size() != 1 || rhsContract[0] != 2 ||
      outputShape[2] != 1) {
    return false;
  }
  if (lhsSourceShape[groupedDim] * groupSize != lhsShape[groupedDim] ||
      rhsShape[rhsBatch[0]] != lhsShape[groupedDim] ||
      outputShape[0] != lhsShape[groupedDim] ||
      outputShape[1] != lhsShape[lhsFree[0]]) {
    return false;
  }

  int64_t rhsFreeProduct = 1;
  for (int64_t dim : rhsFree) {
    rhsFreeProduct *= rhsShape[dim];
  }
  if (rhsFreeProduct != 1) {
    return false;
  }
  for (auto [lhsDim, rhsDim] : llvm::zip(lhsBatch, rhsBatch)) {
    if (lhsShape[lhsDim] != rhsShape[rhsDim]) {
      return false;
    }
  }
  for (auto [lhsDim, rhsDim] : llvm::zip(lhsContract, rhsContract)) {
    if (lhsShape[lhsDim] != rhsShape[rhsDim]) {
      return false;
    }
  }
  return true;
}

std::optional<GroupedHeadExpansion> markerGroupedHeadExpansion(
    mlir::Value value) {
  auto customCall = value.getDefiningOp<mlir::stablehlo::CustomCallOp>();
  if (!customCall || customCall.getCallTargetName() !=
                         kGroupedHeadExpansionTarget ||
      customCall.getHasSideEffect() || customCall->getNumResults() != 1 ||
      customCall.getInputs().size() != 1) {
    return std::nullopt;
  }

  auto resultType = getStaticRankedTensor(customCall.getResult(0));
  auto sourceType = getStaticRankedTensor(customCall.getInputs().front());
  auto groupedDimAttr =
      customCall->getAttrOfType<mlir::IntegerAttr>(
          kGroupedHeadDimensionAttr);
  auto groupSizeAttr =
      customCall->getAttrOfType<mlir::IntegerAttr>(kGroupedHeadGroupSizeAttr);
  if (!resultType || !sourceType || !groupedDimAttr || !groupSizeAttr ||
      resultType->getRank() != sourceType->getRank()) {
    return std::nullopt;
  }

  int64_t groupedDim = groupedDimAttr.getInt();
  int64_t groupSize = groupSizeAttr.getInt();
  if (groupedDim < 0 || groupedDim >= resultType->getRank() ||
      groupedDim > std::numeric_limits<uint32_t>::max() || groupSize <= 1 ||
      groupSize > std::numeric_limits<uint32_t>::max()) {
    return std::nullopt;
  }

  return GroupedHeadExpansion{
      .source = customCall.getInputs().front(),
      .logicalShape = llvm::SmallVector<int64_t>(
          resultType->getShape().begin(), resultType->getShape().end()),
      .groupedDimension = static_cast<uint32_t>(groupedDim),
      .groupSize = static_cast<uint32_t>(groupSize),
  };
}

std::optional<GroupedHeadExpansion> matchGroupedHeadMatmulExpansion(
    mlir::stablehlo::DotGeneralOp dotOp) {
  if (dotOp->getParentOfType<mlir::stablehlo::CaseOp>()) {
    return std::nullopt;
  }
  auto expansion = matchGroupedHeadExpansion(dotOp.getLhs());
  if (!expansion || !isPackedGqaLhsExpansion(dotOp, *expansion)) {
    return std::nullopt;
  }
  return expansion;
}

mlir::LogicalResult createGroupedHeadExpansionMarker(
    mlir::PatternRewriter &rewriter, mlir::stablehlo::DotGeneralOp dotOp,
    const GroupedHeadExpansion &expansion) {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "call_target_name", rewriter.getStringAttr(kGroupedHeadExpansionTarget)));
  attrs.push_back(
      rewriter.getNamedAttr("has_side_effect", rewriter.getBoolAttr(false)));
  attrs.push_back(rewriter.getNamedAttr(
      kGroupedHeadDimensionAttr,
      rewriter.getI64IntegerAttr(expansion.groupedDimension)));
  attrs.push_back(rewriter.getNamedAttr(
      kGroupedHeadGroupSizeAttr,
      rewriter.getI64IntegerAttr(expansion.groupSize)));

  rewriter.setInsertionPoint(dotOp);
  auto marker = rewriter.create<mlir::stablehlo::CustomCallOp>(
      dotOp.getLoc(), mlir::TypeRange{dotOp.getLhs().getType()},
      mlir::ValueRange{expansion.source}, attrs);
  auto replacement = rewriter.create<mlir::stablehlo::DotGeneralOp>(
      dotOp.getLoc(), dotOp.getResult().getType(), marker.getResult(0),
      dotOp.getRhs(), dotOp.getDotDimensionNumbers(),
      dotOp.getPrecisionConfigAttr(), dotOp.getAlgorithmAttr());
  rewriter.replaceOp(dotOp, replacement.getResult());
  return mlir::success();
}

} // namespace

mlir::LogicalResult
GQAMatmulFusing::matchAndRewrite(mlir::stablehlo::DotGeneralOp dotOp,
                                 mlir::PatternRewriter &rewriter) const {
  auto expansion = matchGroupedHeadMatmulExpansion(dotOp);
  if (!expansion) {
    return mlir::failure();
  }
  return createGroupedHeadExpansionMarker(rewriter, dotOp, *expansion);
}

std::optional<GroupedHeadExpansion>
matchPackedGqaMatmulLhsMarker(mlir::stablehlo::DotGeneralOp dotOp,
                              mlir::Value value) {
  if (dotOp.getLhs() != value) {
    return std::nullopt;
  }
  auto expansion = markerGroupedHeadExpansion(value);
  if (!expansion || !isPackedGqaLhsExpansion(dotOp, *expansion)) {
    return std::nullopt;
  }
  return expansion;
}

bool isGroupedHeadExpansionMarker(mlir::stablehlo::CustomCallOp customCallOp) {
  return customCallOp &&
         customCallOp.getCallTargetName() == kGroupedHeadExpansionTarget &&
         markerGroupedHeadExpansion(customCallOp.getResult(0)).has_value();
}

} // namespace libtt::mlir_frontend
