// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/QKVHeadsNormRopeDecodeFusing.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttnn::fusing {

namespace {

// The non-deallocation users of a value.
SmallVector<Operation *> getNonDeallocUsers(Value value) {
  SmallVector<Operation *> users;
  for (Operation *user : value.getUsers()) {
    if (!isa<DeallocateOp>(user)) {
      users.push_back(user);
    }
  }
  return users;
}

// The single user of a value that is not a deallocation, or nullptr.
Operation *getSingleNonDeallocUser(Value value) {
  SmallVector<Operation *> users = getNonDeallocUsers(value);
  return users.size() == 1 ? users.front() : nullptr;
}

// The single user of `value` and the operand it reads: `value` itself, or a
// reshape of `value` that is its single user and keeps the last dimension.
// Such a reshape regroups whole rows: a norm or rotary embedding over the
// last dimension sees the same rows. With `splitRows`, the reshape may also
// split each row into several, as a projection [B, heads * D] into heads
// [B, heads, D].
struct OperandUse {
  Value operand;
  Operation *user;
};

OperandUse getUseThroughRowReshape(Value value, bool splitRows = false) {
  Operation *user = getSingleNonDeallocUser(value);
  if (auto reshape = dyn_cast_or_null<ReshapeOp>(user)) {
    int64_t from =
        mlir::cast<RankedTensorType>(value.getType()).getShape().back();
    int64_t to = reshape.getType().getShape().back();
    if (to == from || (splitRows && to > 0 && from % to == 0)) {
      return {reshape.getResult(),
              getSingleNonDeallocUser(reshape.getResult())};
    }
  }
  return {value, user};
}

// The [begin, end) of a static slice with step 1 that only slices the last
// dimension.
std::optional<std::pair<int64_t, int64_t>>
getLastDimSlice(SliceStaticOp slice) {
  ArrayRef<int64_t> shape = slice.getInput().getType().getShape();
  auto at = [](ArrayAttr values, size_t i) {
    return mlir::cast<IntegerAttr>(values[i]).getInt();
  };
  if (slice.getBegins().size() != shape.size() || shape.empty()) {
    return std::nullopt;
  }
  size_t last = shape.size() - 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (at(slice.getStep(), i) != 1 ||
        (i != last && (at(slice.getBegins(), i) != 0 ||
                       at(slice.getEnds(), i) != shape[i]))) {
      return std::nullopt;
    }
  }
  return std::make_pair(at(slice.getBegins(), last), at(slice.getEnds(), last));
}

// A rotary embedding with token_index 0 of `input` (not of its cos or sin).
RotaryEmbeddingOp getDecodeRope(OperandUse use) {
  auto rope = dyn_cast_or_null<RotaryEmbeddingOp>(use.user);
  if (!rope || rope.getInput() != use.operand || !rope.getTokenIndex() ||
      *rope.getTokenIndex() != 0) {
    return nullptr;
  }
  return rope;
}

// The rotary embedding of normalized heads [.., D]: of the whole head, or of
// its first rotary_dim elements, sliced off, rotated and concatenated back
// onto the rest. `result` is the rotated heads.
struct RopeMatch {
  Value result;
  Value cos;
  Value sin;
  int64_t rotaryDim;
};

std::optional<RopeMatch> matchRope(Value normed) {
  int64_t headDim =
      mlir::cast<RankedTensorType>(normed.getType()).getShape().back();
  OperandUse use = getUseThroughRowReshape(normed);
  if (RotaryEmbeddingOp rope = getDecodeRope(use)) {
    return RopeMatch{rope.getResult(), rope.getCosCache(), rope.getSinCache(),
                     headDim};
  }

  // The partial rotary slices of the normalized heads, or of a reshape that
  // regroups their rows.
  SmallVector<Operation *> users = getNonDeallocUsers(use.operand);
  if (users.size() != 2) {
    return std::nullopt;
  }
  auto first = dyn_cast<SliceStaticOp>(users[0]);
  auto second = dyn_cast<SliceStaticOp>(users[1]);
  if (!first || !second) {
    return std::nullopt;
  }
  auto firstBounds = getLastDimSlice(first);
  auto secondBounds = getLastDimSlice(second);
  if (!firstBounds || !secondBounds) {
    return std::nullopt;
  }
  if (firstBounds->first != 0) {
    std::swap(first, second);
    std::swap(firstBounds, secondBounds);
  }
  int64_t rotaryDim = firstBounds->second;
  if (firstBounds->first != 0 || secondBounds->first != rotaryDim ||
      secondBounds->second != headDim) {
    return std::nullopt;
  }
  RotaryEmbeddingOp rope =
      getDecodeRope(getUseThroughRowReshape(first.getResult()));
  if (!rope) {
    return std::nullopt;
  }
  // The pass-through part, reshaped like the rotated part.
  Value rest = second.getResult();
  if (auto reshape =
          dyn_cast_or_null<ReshapeOp>(getSingleNonDeallocUser(rest))) {
    rest = reshape.getResult();
  }
  auto concat =
      dyn_cast_or_null<ConcatOp>(getSingleNonDeallocUser(rope.getResult()));
  int64_t rank = concat ? concat.getType().getRank() : 0;
  if (!concat || concat.getInputs().size() != 2 ||
      concat.getInputs()[0] != rope.getResult() ||
      concat.getInputs()[1] != rest ||
      (concat.getDim() != rank - 1 && concat.getDim() != -1)) {
    return std::nullopt;
  }
  return RopeMatch{concat.getResult(), rope.getCosCache(), rope.getSinCache(),
                   rotaryDim};
}

// An rms_norm with a weight and no bias over heads of headDim, directly or
// after a reshape that splits rows into heads, then its rotary embedding.
struct NormRope {
  RMSNormOp norm;
  RopeMatch rope;
};

std::optional<NormRope> matchNormRope(Value heads, int64_t headDim) {
  OperandUse use = getUseThroughRowReshape(heads, /*splitRows=*/true);
  auto norm = dyn_cast_or_null<RMSNormOp>(use.user);
  if (!norm || norm.getInput() != use.operand || !norm.getWeight() ||
      norm.getBias() ||
      norm.getInput().getType().getShape().back() != headDim) {
    return std::nullopt;
  }
  std::optional<RopeMatch> rope = matchRope(norm.getResult());
  if (!rope) {
    return std::nullopt;
  }
  return NormRope{norm, *rope};
}

// The q and k chains of one prologue: the same cos, sin and epsilon.
bool areCompatible(NormRope q, NormRope k) {
  return q.rope.cos == k.rope.cos && q.rope.sin == k.rope.sin &&
         q.rope.rotaryDim == k.rope.rotaryDim &&
         q.norm.getEpsilonAttr() == k.norm.getEpsilonAttr();
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

// What the fused op computes, and the values it replaces.
struct FusedPrologue {
  Location loc;
  // [1, 1, B, W], or a value to reshape to it.
  Value input;
  Value qWeight;
  Value kWeight;
  Value cos;
  Value sin;
  uint32_t numHeads;
  uint32_t numKVHeads;
  int64_t headDim;
  int64_t rotaryDim;
  bool gatedQuery;
  FloatAttr epsilon;
  Value query;
  Value key;
  Value value;
};

// Checks the shapes and placement of a matched prologue and replaces it with
// one nlp_create_qkv_heads_decode_norm_rope; fails without changing the IR.
mlir::LogicalResult createFusedOp(const FusedPrologue &p,
                                  mlir::PatternRewriter &rewriter) {
  // One tile row of users and of heads per core, heads of whole tiles,
  // rotated halves of whole tiles, and rows of cos, sin and the weights of
  // one head each.
  auto inputType = mlir::cast<RankedTensorType>(p.input.getType());
  int64_t batch = inputType.getShape()[inputType.getRank() - 2];
  int64_t width = inputType.getShape().back();
  if (batch > 32 || p.numKVHeads > 32 || p.headDim % 32 != 0 ||
      p.rotaryDim % 64 != 0 || p.rotaryDim > p.headDim ||
      inputType.getNumElements() != batch * width ||
      width != ((p.gatedQuery ? 2 : 1) * p.numHeads + 2 * p.numKVHeads) *
                   p.headDim ||
      !isInterleavedDRAMTile(p.input)) {
    return mlir::failure();
  }
  for (auto [row, size] : {std::pair{p.qWeight, p.headDim},
                           {p.kWeight, p.headDim},
                           {p.cos, p.rotaryDim},
                           {p.sin, p.rotaryDim}}) {
    auto type = mlir::cast<RankedTensorType>(row.getType());
    if (type.getNumElements() != size || type.getShape().back() != size ||
        !isInterleavedDRAMTile(row)) {
      return mlir::failure();
    }
  }
  for (auto [result, heads] : {std::pair{p.query, p.numHeads},
                               {p.key, p.numKVHeads},
                               {p.value, p.numKVHeads}}) {
    if (mlir::cast<RankedTensorType>(result.getType()).getShape() !=
        ArrayRef<int64_t>{1, batch, heads, p.headDim}) {
      return mlir::failure();
    }
  }

  // Insert after the last of the replaced results and every operand; every
  // user of the results must come after that point.
  Operation *insertAfter = nullptr;
  for (Value result : {p.query, p.key, p.value}) {
    Operation *def = result.getDefiningOp();
    if (!def || (insertAfter && def->getBlock() != insertAfter->getBlock())) {
      return mlir::failure();
    }
    if (!insertAfter || insertAfter->isBeforeInBlock(def)) {
      insertAfter = def;
    }
  }
  Block *block = insertAfter->getBlock();
  for (Value operand : {p.input, p.qWeight, p.kWeight, p.cos, p.sin}) {
    Operation *def = operand.getDefiningOp();
    if (def &&
        (def->getBlock() != block || insertAfter->isBeforeInBlock(def))) {
      return mlir::failure();
    }
  }
  for (Value result : {p.query, p.key, p.value}) {
    for (Operation *user : result.getUsers()) {
      if (user->getBlock() != block || !insertAfter->isBeforeInBlock(user)) {
        return mlir::failure();
      }
    }
  }

  rewriter.setInsertionPointAfter(insertAfter);
  Value input = p.input;
  if (inputType.getShape() != ArrayRef<int64_t>{1, 1, batch, width}) {
    input = ttir_to_ttnn::utils::generateReshape(
        mlir::cast<TypedValue<RankedTensorType>>(input),
        SmallVector<int64_t>{1, 1, batch, width}, rewriter,
        ttmlir::utils::appendLocationSuffix(p.loc, "_input"));
  }
  // The weight rows: q's for the q heads, then k's for the k heads.
  auto weightRows = [&](Value weight, int64_t rows) -> Value {
    auto type = mlir::cast<RankedTensorType>(weight.getType());
    Value row = ttir_to_ttnn::utils::generateReshape(
        mlir::cast<TypedValue<RankedTensorType>>(weight),
        SmallVector<int64_t>{1, p.headDim}, rewriter,
        ttmlir::utils::appendLocationSuffix(p.loc, "_weight_row"));
    if (rows == 1) {
      return row;
    }
    return rewriter.create<RepeatOp>(
        ttmlir::utils::appendLocationSuffix(p.loc, "_weight_rows"),
        utils::RankedTensorTypeFactory::create(
            type, SmallVector<int64_t>{rows, p.headDim}),
        row, ShapeAttr::get(rewriter.getContext(), {rows, 1}));
  };
  Value qRows = weightRows(p.qWeight, p.numHeads);
  Value kRows = weightRows(p.kWeight, p.numKVHeads);
  auto normWeight = rewriter.create<ConcatOp>(
      ttmlir::utils::appendLocationSuffix(p.loc, "_norm_weight"),
      utils::RankedTensorTypeFactory::create(
          mlir::cast<RankedTensorType>(p.qWeight.getType()),
          SmallVector<int64_t>{static_cast<int64_t>(p.numHeads + p.numKVHeads),
                               p.headDim}),
      ValueRange{qRows, kRows}, /*dim=*/0);
  auto fused = rewriter.create<NLPCreateQKVHeadsDecodeNormRopeOp>(
      p.loc, TypeRange{p.query.getType(), p.key.getType(), p.value.getType()},
      input, normWeight.getResult(), p.cos, p.sin,
      rewriter.getUI32IntegerAttr(p.numHeads),
      rewriter.getUI32IntegerAttr(p.numKVHeads), p.epsilon,
      rewriter.getBoolAttr(p.gatedQuery));
  // The replaced chains are left without users and erased as dead.
  rewriter.replaceAllUsesWith(p.query, fused.getQuery());
  rewriter.replaceAllUsesWith(p.key, fused.getKey());
  rewriter.replaceAllUsesWith(p.value, fused.getValue());
  return mlir::success();
}

} // namespace

mlir::LogicalResult NLPCreateQKVHeadsDecodeNormRopeFusing::matchAndRewrite(
    NLPCreateQKVHeadsDecodeOp decodeOp, mlir::PatternRewriter &rewriter) const {
  if (decodeOp.getBatchOffset() || decodeOp.getSliceSize() ||
      (decodeOp.getOverlapQkCoregrid() && !*decodeOp.getOverlapQkCoregrid())) {
    return mlir::failure();
  }
  ArrayRef<int64_t> inputShape = decodeOp.getInput().getType().getShape();
  uint32_t numHeads = decodeOp.getNumHeads();
  uint32_t numKVHeads = decodeOp.getNumKvHeads().value_or(numHeads);
  if (inputShape.size() != 4 || inputShape[0] != 1 || inputShape[1] != 1 ||
      inputShape[3] % (numHeads + 2 * numKVHeads) != 0) {
    return mlir::failure();
  }
  int64_t headDim = inputShape[3] / (numHeads + 2 * numKVHeads);
  std::optional<NormRope> q = matchNormRope(decodeOp.getQuery(), headDim);
  std::optional<NormRope> k = matchNormRope(decodeOp.getKey(), headDim);
  if (!q || !k || !areCompatible(*q, *k)) {
    return mlir::failure();
  }
  return createFusedOp(
      FusedPrologue{decodeOp.getLoc(), decodeOp.getInput(), q->norm.getWeight(),
                    k->norm.getWeight(), q->rope.cos, q->rope.sin, numHeads,
                    numKVHeads, headDim, q->rope.rotaryDim,
                    /*gatedQuery=*/false, q->norm.getEpsilonAttr(),
                    q->rope.result, k->rope.result, decodeOp.getValue()},
      rewriter);
}

mlir::LogicalResult GatedQKVHeadsDecodeNormRopeFusing::matchAndRewrite(
    MatmulOp matmulOp, mlir::PatternRewriter &rewriter) const {
  // A [B, W] projection split into three slices: the gated q heads, k and v.
  ArrayRef<int64_t> shape = matmulOp.getType().getShape();
  SmallVector<Operation *> users = getNonDeallocUsers(matmulOp.getResult());
  if (shape.size() != 2 || users.size() != 3) {
    return mlir::failure();
  }
  SmallVector<std::pair<std::pair<int64_t, int64_t>, SliceStaticOp>, 3> slices;
  for (Operation *user : users) {
    auto slice = dyn_cast<SliceStaticOp>(user);
    std::optional<std::pair<int64_t, int64_t>> bounds =
        slice ? getLastDimSlice(slice) : std::nullopt;
    if (!bounds) {
      return mlir::failure();
    }
    slices.push_back({*bounds, slice});
  }
  llvm::sort(slices,
             [](const auto &a, const auto &b) { return a.first < b.first; });
  if (slices[0].first.first != 0 ||
      slices[1].first.first != slices[0].first.second ||
      slices[2].first.first != slices[1].first.second ||
      slices[2].first.second != shape[1]) {
    return mlir::failure();
  }
  auto [qGateSlice, kSlice, vSlice] =
      std::tuple{slices[0].second, slices[1].second, slices[2].second};

  // The q and gate heads, [.., 2 * D], each split in half.
  auto qGateHeads = dyn_cast_or_null<ReshapeOp>(
      getSingleNonDeallocUser(qGateSlice.getResult()));
  if (!qGateHeads) {
    return mlir::failure();
  }
  int64_t pairWidth = qGateHeads.getType().getShape().back();
  int64_t headDim = pairWidth / 2;
  SliceStaticOp qSlice;
  for (Operation *user : getNonDeallocUsers(qGateHeads.getResult())) {
    auto slice = dyn_cast<SliceStaticOp>(user);
    auto bounds = slice ? getLastDimSlice(slice) : std::nullopt;
    if (bounds && bounds->first == 0 && bounds->second == headDim && !qSlice) {
      qSlice = slice;
    }
  }
  if (pairWidth % 2 != 0 || !qSlice ||
      qGateSlice.getType().getShape()[1] % pairWidth != 0) {
    return mlir::failure();
  }
  auto numHeads =
      static_cast<uint32_t>(qGateSlice.getType().getShape()[1] / pairWidth);
  int64_t kWidth = kSlice.getType().getShape()[1];
  if (kWidth % headDim != 0 || vSlice.getType().getShape()[1] != kWidth) {
    return mlir::failure();
  }
  auto numKVHeads = static_cast<uint32_t>(kWidth / headDim);

  std::optional<NormRope> q = matchNormRope(qSlice.getResult(), headDim);
  std::optional<NormRope> k = matchNormRope(kSlice.getResult(), headDim);
  if (!q || !k || !areCompatible(*q, *k)) {
    return mlir::failure();
  }
  auto value =
      dyn_cast_or_null<ReshapeOp>(getSingleNonDeallocUser(vSlice.getResult()));
  if (!value) {
    return mlir::failure();
  }
  return createFusedOp(FusedPrologue{matmulOp.getLoc(), matmulOp.getResult(),
                                     q->norm.getWeight(), k->norm.getWeight(),
                                     q->rope.cos, q->rope.sin, numHeads,
                                     numKVHeads, headDim, q->rope.rotaryDim,
                                     /*gatedQuery=*/true,
                                     q->norm.getEpsilonAttr(), q->rope.result,
                                     k->rope.result, value.getResult()},
                       rewriter);
}

} // namespace mlir::tt::ttnn::fusing
