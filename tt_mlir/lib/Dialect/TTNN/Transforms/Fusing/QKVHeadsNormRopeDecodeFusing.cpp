// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/QKVHeadsNormRopeDecodeFusing.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttnn::fusing {

namespace {

// The single user of a value that is not a deallocation, or nullptr.
Operation *getSingleNonDeallocUser(Value value) {
  Operation *found = nullptr;
  for (Operation *user : value.getUsers()) {
    if (isa<DeallocateOp>(user)) {
      continue;
    }
    if (found) {
      return nullptr;
    }
    found = user;
  }
  return found;
}

// The rms_norm -> rotary_embedding chain on a q or k result of a decode
// heads op: a norm with a weight and no bias, whose only user is a rotary
// embedding with token_index 0.
struct NormRopeChain {
  RMSNormOp norm;
  RotaryEmbeddingOp rope;
};

std::optional<NormRopeChain> matchNormRope(Value heads) {
  auto norm = dyn_cast_or_null<RMSNormOp>(getSingleNonDeallocUser(heads));
  if (!norm || norm.getInput() != heads || !norm.getWeight() ||
      norm.getBias()) {
    return std::nullopt;
  }
  auto rope =
      dyn_cast_or_null<RotaryEmbeddingOp>(getSingleNonDeallocUser(norm.getResult()));
  if (!rope || rope.getInput() != norm.getResult() || !rope.getTokenIndex() ||
      *rope.getTokenIndex() != 0) {
    return std::nullopt;
  }
  return NormRopeChain{norm, rope};
}

// A tiled, interleaved DRAM tensor of BF16 or FP32, as the fused op reads.
bool isInterleavedDRAMTile(Value value) {
  auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(value.getType()).getEncoding());
  return layout && layout.isTiled() &&
         layout.hasInterleavedDRAMTensorMemoryLayout() &&
         (layout.getDataType() == ttcore::DataType::BFloat16 ||
          layout.getDataType() == ttcore::DataType::Float32);
}

} // namespace

mlir::LogicalResult NLPCreateQKVHeadsDecodeNormRopeFusing::matchAndRewrite(
    NLPCreateQKVHeadsDecodeOp decodeOp, mlir::PatternRewriter &rewriter) const {
  if (decodeOp.getBatchOffset() || decodeOp.getSliceSize() ||
      (decodeOp.getOverlapQkCoregrid() && !*decodeOp.getOverlapQkCoregrid())) {
    return mlir::failure();
  }
  std::optional<NormRopeChain> q = matchNormRope(decodeOp.getQuery());
  std::optional<NormRopeChain> k = matchNormRope(decodeOp.getKey());
  if (!q || !k || q->rope.getCosCache() != k->rope.getCosCache() ||
      q->rope.getSinCache() != k->rope.getSinCache() ||
      q->norm.getEpsilonAttr() != k->norm.getEpsilonAttr()) {
    return mlir::failure();
  }

  // One tile row of users and of heads per core, heads whose halves are whole
  // tiles, and rows of cos, sin and the weights of one head each.
  uint32_t numHeads = decodeOp.getNumHeads();
  uint32_t numKVHeads = decodeOp.getNumKvHeads().value_or(numHeads);
  ArrayRef<int64_t> inputShape = decodeOp.getInput().getType().getShape();
  if (inputShape.size() != 4 || inputShape[0] != 1 || inputShape[1] != 1 ||
      inputShape[2] > 32 || numKVHeads > 32 ||
      inputShape[3] % (numHeads + 2 * numKVHeads) != 0) {
    return mlir::failure();
  }
  int64_t headDim = inputShape[3] / (numHeads + 2 * numKVHeads);
  if (headDim % 64 != 0) {
    return mlir::failure();
  }
  Value qWeight = q->norm.getWeight();
  Value kWeight = k->norm.getWeight();
  Value cos = q->rope.getCosCache();
  Value sin = q->rope.getSinCache();
  for (Value row : {qWeight, kWeight, cos, sin}) {
    auto type = mlir::cast<RankedTensorType>(row.getType());
    if (type.getNumElements() != headDim || type.getShape().back() != headDim ||
        !isInterleavedDRAMTile(row)) {
      return mlir::failure();
    }
  }
  if (!isInterleavedDRAMTile(decodeOp.getInput())) {
    return mlir::failure();
  }

  // Insert after the last of the two chains and every operand; every user of
  // the results must come after that point.
  Operation *insertAfter =
      q->rope->isBeforeInBlock(k->rope) ? k->rope.getOperation()
                                        : q->rope.getOperation();
  Block *block = insertAfter->getBlock();
  for (Value operand : {qWeight, kWeight, cos, sin}) {
    Operation *def = operand.getDefiningOp();
    if (def && (def->getBlock() != block || insertAfter->isBeforeInBlock(def))) {
      return mlir::failure();
    }
  }
  for (Value result :
       {q->rope.getResult(), k->rope.getResult(), decodeOp.getValue()}) {
    for (Operation *user : result.getUsers()) {
      if (user->getBlock() != block || !insertAfter->isBeforeInBlock(user)) {
        return mlir::failure();
      }
    }
  }

  rewriter.setInsertionPointAfter(insertAfter);
  Location loc = decodeOp.getLoc();
  // The weight rows: q's for the q heads, then k's for the k heads.
  auto weightRows = [&](Value weight, int64_t rows) -> Value {
    auto type = mlir::cast<RankedTensorType>(weight.getType());
    Value row = ttir_to_ttnn::utils::generateReshape(
        mlir::cast<TypedValue<RankedTensorType>>(weight), SmallVector<int64_t>{1, headDim},
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_weight_row"));
    if (rows == 1) {
      return row;
    }
    return rewriter.create<RepeatOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_weight_rows"),
        utils::RankedTensorTypeFactory::create(type, SmallVector<int64_t>{rows, headDim}), row,
        ShapeAttr::get(rewriter.getContext(), {rows, 1}));
  };
  Value qRows = weightRows(qWeight, numHeads);
  Value kRows = weightRows(kWeight, numKVHeads);
  auto normWeight = rewriter.create<ConcatOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_norm_weight"),
      utils::RankedTensorTypeFactory::create(
          mlir::cast<RankedTensorType>(qWeight.getType()),
          SmallVector<int64_t>{static_cast<int64_t>(numHeads + numKVHeads), headDim}),
      ValueRange{qRows, kRows}, /*dim=*/0);
  auto fused = rewriter.create<NLPCreateQKVHeadsDecodeNormRopeOp>(
      loc,
      TypeRange{q->rope.getType(), k->rope.getType(),
                decodeOp.getValue().getType()},
      decodeOp.getInput(), normWeight.getResult(), cos, sin,
      rewriter.getUI32IntegerAttr(numHeads),
      rewriter.getUI32IntegerAttr(numKVHeads), q->norm.getEpsilonAttr());
  rewriter.replaceOp(q->rope, fused.getQuery());
  rewriter.replaceOp(k->rope, fused.getKey());
  rewriter.eraseOp(q->norm);
  rewriter.eraseOp(k->norm);
  rewriter.replaceAllUsesWith(decodeOp.getValue(), fused.getValue());
  rewriter.eraseOp(decodeOp);
  return mlir::success();
}

} // namespace mlir::tt::ttnn::fusing
