// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SplitQKVNormRopeDecodeFusing.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttnn::fusing {

namespace {

// The single user of a value, or nullptr.
Operation *getSingleUser(Value value) {
  return value.hasOneUse() ? *value.getUsers().begin() : nullptr;
}

Value throughReshapes(Value value) {
  while (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    value = reshape.getInput();
  }
  return value;
}

// The end of the chain of reshapes that are each the single user of the
// previous value, from `value`.
Value alongReshapes(Value value) {
  while (auto reshape = dyn_cast_or_null<ReshapeOp>(getSingleUser(value))) {
    value = reshape.getResult();
  }
  return value;
}

ArrayRef<int64_t> shapeOf(Value value) {
  return mlir::cast<RankedTensorType>(value.getType()).getShape();
}

// The value that reshapes and static slices of `value` read, and the flat
// (row-major) index there of `value`'s element `flat`.
std::pair<Value, int64_t> sourceOf(Value value, int64_t flat) {
  for (;;) {
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      value = reshape.getInput();
      continue;
    }
    auto slice = value.getDefiningOp<SliceStaticOp>();
    if (!slice) {
      return {value, flat};
    }
    ArrayRef<int64_t> outShape = shapeOf(slice.getResult());
    ArrayRef<int64_t> inShape = shapeOf(slice.getInput());
    int64_t inFlat = 0;
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(outShape.size()) - 1; i >= 0; --i) {
      const int64_t index = flat % outShape[i];
      flat /= outShape[i];
      const int64_t begin =
          mlir::cast<IntegerAttr>(slice.getBegins()[i]).getInt();
      const int64_t step = mlir::cast<IntegerAttr>(slice.getStep()[i]).getInt();
      inFlat += (begin + index * step) * stride;
      stride *= inShape[i];
    }
    value = slice.getInput();
    flat = inFlat;
  }
}

// Whether `value` holds `heads` heads of every row of `qkv` [tokens, width],
// head_dim wide, the first at `offset` and each `stride` after the last, in
// (token, head, element) order.
bool isHeadView(Value value, Value qkv, int64_t tokens, int64_t width,
                int64_t offset, int64_t heads, int64_t stride,
                int64_t headDim) {
  const int64_t rowElements = heads * headDim;
  if (mlir::cast<RankedTensorType>(value.getType()).getNumElements() !=
      tokens * rowElements) {
    return false;
  }
  for (int64_t i = 0; i < tokens * rowElements; ++i) {
    const int64_t head = i % rowElements / headDim;
    auto [source, flat] = sourceOf(value, i);
    if (source != qkv || flat != i / rowElements * width + offset +
                                     head * stride + i % headDim) {
      return false;
    }
  }
  return true;
}

// A tiled tensor of BF16 (or FP32 if `float32`); interleaved in DRAM if
// `dram`.
bool isTiled(Value value, bool dram, bool float32 = false) {
  auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(value.getType()).getEncoding());
  return layout && layout.isTiled() &&
         (layout.getDataType() == ttcore::DataType::BFloat16 ||
          (float32 && layout.getDataType() == ttcore::DataType::Float32)) &&
         (!dram || layout.hasInterleavedDRAMTensorMemoryLayout());
}

// Whether `value`, through reshapes, is a static slice of `source` taking
// [begin, end) of its last dimension and all of the others.
bool isLastDimSlice(Value value, Value source, int64_t begin, int64_t end) {
  auto slice = throughReshapes(value).getDefiningOp<SliceStaticOp>();
  if (!slice || slice.getInput() != source) {
    return false;
  }
  ArrayRef<int64_t> shape = shapeOf(source);
  for (size_t i = 0; i < shape.size(); ++i) {
    const bool last = i + 1 == shape.size();
    auto attr = [&](ArrayAttr array) {
      return mlir::cast<IntegerAttr>(array[i]).getInt();
    };
    if (attr(slice.getStep()) != 1 ||
        attr(slice.getBegins()) != (last ? begin : 0) ||
        attr(slice.getEnds()) != (last ? end : shape[i])) {
      return false;
    }
  }
  return true;
}

// The heads `rope` rotates: all of their dims, or, in a partial rotary, a
// slice of their leading ones.
Value rotatedHeads(RotaryEmbeddingOp rope, int64_t headDim) {
  const int64_t rotated = shapeOf(rope.getInput()).back();
  Value x = throughReshapes(rope.getInput());
  if (rotated == headDim) {
    return x;
  }
  auto slice = x.getDefiningOp<SliceStaticOp>();
  return slice && isLastDimSlice(x, slice.getInput(), 0, rotated)
             ? slice.getInput()
             : Value();
}

// The rms_norm whose heads `rope` rotates.
RMSNormOp normOf(RotaryEmbeddingOp rope, int64_t headDim) {
  Value heads = rotatedHeads(rope, headDim);
  return heads ? throughReshapes(heads).getDefiningOp<RMSNormOp>() : nullptr;
}

// Whether `parts`, side by side, are the heads of `norm` with the dims `rope`
// rotates rotated: its result, then the rest of the heads.
bool rotates(RotaryEmbeddingOp rope, RMSNormOp norm, ValueRange parts,
             int64_t headDim) {
  const int64_t rotated = shapeOf(rope.getInput()).back();
  return normOf(rope, headDim) == norm && !parts.empty() &&
         throughReshapes(parts[0]) == rope.getResult() &&
         (rotated == headDim
              ? parts.size() == 1
              : parts.size() == 2 &&
                    isLastDimSlice(parts[1], rotatedHeads(rope, headDim),
                                   rotated, headDim));
}

bool isLastDimConcat(ConcatOp concat) {
  const int64_t rank = shapeOf(concat.getResult()).size();
  return concat.getDim() == rank - 1 || concat.getDim() == -1;
}

} // namespace

mlir::LogicalResult SplitQKVNormRopeDecodeFusing::matchAndRewrite(
    ConcatOp concat, mlir::PatternRewriter &rewriter) const {
  // concat(k, v) [B, Hkv, 2 D] along the last dim, k as rotated parts.
  ArrayRef<int64_t> kvShape = shapeOf(concat.getResult());
  if (kvShape.size() != 3 || !isLastDimConcat(concat) ||
      concat.getInputs().size() < 2 || kvShape[2] % 2 != 0) {
    return mlir::failure();
  }
  const int64_t tokens = kvShape[0];
  const int64_t numKVHeads = kvShape[1];
  const int64_t headDim = kvShape[2] / 2;
  ValueRange kParts = concat.getInputs().drop_back();
  Value vHeads = concat.getInputs().back();
  if (tokens > 32 || numKVHeads > 32 || headDim % 32 != 0) {
    return mlir::failure();
  }

  // k's rotary embedding and norm, and the projection the norm reads.
  auto kRope = throughReshapes(kParts[0]).getDefiningOp<RotaryEmbeddingOp>();
  RMSNormOp kNorm = kRope ? normOf(kRope, headDim) : nullptr;
  if (!kNorm || !rotates(kRope, kNorm, kParts, headDim)) {
    return mlir::failure();
  }
  Value qkv = sourceOf(kNorm.getInput(), 0).first;
  // q's: the rotary embedding of the same cos and sin (which every layer
  // shares) whose norm reads the same projection.
  RotaryEmbeddingOp qRope;
  RMSNormOp qNorm;
  for (Operation *user : kRope.getCosCache().getUsers()) {
    auto rope = dyn_cast<RotaryEmbeddingOp>(user);
    if (!rope || rope == kRope || rope.getSinCache() != kRope.getSinCache() ||
        rope.getTokenIndex() != kRope.getTokenIndex()) {
      continue;
    }
    RMSNormOp norm = normOf(rope, headDim);
    if (!norm || sourceOf(norm.getInput(), 0).first != qkv) {
      continue;
    }
    if (qRope) {
      return mlir::failure();
    }
    qRope = rope;
    qNorm = norm;
  }
  if (!qRope || !qNorm.getWeight() || !kNorm.getWeight() || qNorm.getBias() ||
      kNorm.getBias() || qNorm.getEpsilonAttr() != kNorm.getEpsilonAttr()) {
    return mlir::failure();
  }

  // The heads are views of one projection [B, (s Hq + 2 Hkv) D], each q head
  // followed by its gate head if s is 2.
  auto qkvType = mlir::cast<RankedTensorType>(qkv.getType());
  const int64_t width = qkvType.getShape().back();
  const int64_t qElements =
      mlir::cast<RankedTensorType>(qNorm.getInput().getType()).getNumElements();
  if (qkvType.getNumElements() != tokens * width ||
      qElements % (tokens * headDim) != 0) {
    return mlir::failure();
  }
  const int64_t numQHeads = qElements / (tokens * headDim);
  const int64_t qStride = (width / headDim - 2 * numKVHeads) / numQHeads;
  const int64_t kOffset = numQHeads * qStride * headDim;
  if ((qStride != 1 && qStride != 2) ||
      width != (qStride * numQHeads + 2 * numKVHeads) * headDim ||
      !isHeadView(qNorm.getInput(), qkv, tokens, width, 0, numQHeads,
                  qStride * headDim, headDim) ||
      !isHeadView(kNorm.getInput(), qkv, tokens, width, kOffset, numKVHeads,
                  headDim, headDim) ||
      !isHeadView(vHeads, qkv, tokens, width, kOffset + numKVHeads * headDim,
                  numKVHeads, headDim, headDim)) {
    return mlir::failure();
  }

  // The rotary embeddings take [1, B, H, R] with token b's row of cos and
  // sin: row 0 by token_index 0 for one token, one row per token for several.
  Value cos = qRope.getCosCache();
  Value sin = qRope.getSinCache();
  const int64_t rotated = shapeOf(qRope.getInput()).back();
  const int64_t tableRows = tokens == 1 ? 1 : tokens;
  if (rotated % 64 != 0 || shapeOf(kRope.getInput()).back() != rotated ||
      shapeOf(qRope.getInput()) !=
          ArrayRef<int64_t>{1, tokens, numQHeads, rotated} ||
      shapeOf(kRope.getInput()) !=
          ArrayRef<int64_t>{1, tokens, numKVHeads, rotated} ||
      shapeOf(cos) != ArrayRef<int64_t>{1, tableRows, 1, rotated} ||
      shapeOf(sin) != shapeOf(cos) ||
      (tokens == 1 ? qRope.getTokenIndex() != std::optional<uint32_t>(0)
                   : qRope.getTokenIndex().has_value())) {
    return mlir::failure();
  }
  Value qWeight = qNorm.getWeight();
  Value kWeight = kNorm.getWeight();
  for (Value operand : {qkv, cos, sin}) {
    if (!isTiled(operand, /*dram=*/true)) {
      return mlir::failure();
    }
  }
  // The norm weights are BF16 or, as 1 + w of a zero-centered norm, FP32.
  for (Value weight : {qWeight, kWeight}) {
    if (!isTiled(weight, /*dram=*/true, /*float32=*/true) ||
        mlir::cast<RankedTensorType>(weight.getType()).getNumElements() !=
            headDim) {
      return mlir::failure();
    }
  }
  if (mlir::cast<RankedTensorType>(qWeight.getType()).getElementType() !=
      mlir::cast<RankedTensorType>(kWeight.getType()).getElementType()) {
    return mlir::failure();
  }

  // q: its rotated parts, a pad of the last dim with +0 (the fused kernel's
  // zeros), reshapes to [1, B, Hq, q_width].
  Value qHeads = alongReshapes(qRope.getResult());
  SmallVector<Value> qParts{qHeads};
  if (rotated < headDim) {
    auto qConcat = dyn_cast_or_null<ConcatOp>(getSingleUser(qHeads));
    if (!qConcat || !isLastDimConcat(qConcat)) {
      return mlir::failure();
    }
    qParts.assign(qConcat.getInputs().begin(), qConcat.getInputs().end());
    qHeads = alongReshapes(qConcat.getResult());
  }
  if (!rotates(qRope, qNorm, qParts, headDim)) {
    return mlir::failure();
  }
  auto pad = dyn_cast_or_null<PadOp>(getSingleUser(qHeads));
  if (!pad || !pad.getValue().isPosZero()) {
    return mlir::failure();
  }
  ArrayRef<int32_t> padding = pad.getPadding();
  for (size_t i = 0; i + 1 < padding.size(); ++i) {
    if (padding[i] != 0) {
      return mlir::failure();
    }
  }
  Value qOut = alongReshapes(pad.getResult());
  const int64_t qWidth = shapeOf(qOut).back();
  if (shapeOf(pad.getInput()).back() != headDim || qWidth % 32 != 0 ||
      shapeOf(qOut) != ArrayRef<int64_t>{1, tokens, numQHeads, qWidth} ||
      !isTiled(qOut, /*dram=*/false)) {
    return mlir::failure();
  }
  // k and v of all tokens: the concat reshaped to [1, B, Hkv, 2 D].
  auto kvReshape = dyn_cast_or_null<ReshapeOp>(getSingleUser(concat.getResult()));
  Value kvOut = kvReshape ? alongReshapes(kvReshape.getResult()) : Value();
  if (!kvOut ||
      shapeOf(kvOut) !=
          ArrayRef<int64_t>{1, tokens, numKVHeads, 2 * headDim} ||
      !isTiled(kvOut, /*dram=*/false)) {
    return mlir::failure();
  }
  // Build the fused op before the first user of its results, after its
  // operands.
  Operation *first = nullptr;
  for (Value out : {qOut, kvOut}) {
    for (Operation *user : out.getUsers()) {
      if (user->getBlock() != concat->getBlock()) {
        return mlir::failure();
      }
      if (!first || user->isBeforeInBlock(first)) {
        first = user;
      }
    }
  }
  if (!first) {
    return mlir::failure();
  }
  for (Value operand : {qkv, qWeight, kWeight, cos, sin}) {
    Operation *def = operand.getDefiningOp();
    if (def && (def->getBlock() != first->getBlock() ||
                !def->isBeforeInBlock(first))) {
      return mlir::failure();
    }
  }
  rewriter.setInsertionPoint(first);
  Value input = qkv;
  if (qkvType.getShape() != ArrayRef<int64_t>{tokens, width}) {
    input = ttir_to_ttnn::utils::generateReshape(
        mlir::cast<TypedValue<RankedTensorType>>(qkv),
        SmallVector<int64_t>{tokens, width}, rewriter,
        ttmlir::utils::appendLocationSuffix(qRope.getLoc(), "_qkv_rows"));
  }
  auto fused = rewriter.create<SplitQKVNormRopeDecodeOp>(
      qRope.getLoc(), qOut.getType(), kvOut.getType(), input, qWeight, kWeight,
      cos, sin, rewriter.getUI32IntegerAttr(numQHeads),
      rewriter.getUI32IntegerAttr(numKVHeads), qNorm.getEpsilonAttr());
  rewriter.replaceAllUsesWith(qOut, fused.getQuery());
  rewriter.replaceAllUsesWith(kvOut, fused.getKeyValues());
  // The separate ops are now unused, and the rewrite driver erases them (TTNN
  // ops are pure); any that still feed other ops stay.
  return mlir::success();
}

} // namespace mlir::tt::ttnn::fusing

namespace mlir::tt::ttnn {

namespace {

// Writes the k and v of every token where the op that reads them wants them:
// the workarounds, after fusing, move them into the paged cache update's L1
// shard.
class WriteKeyValuesInPlace
    : public mlir::OpRewritePattern<SplitQKVNormRopeDecodeOp> {
public:
  using OpRewritePattern<SplitQKVNormRopeDecodeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SplitQKVNormRopeDecodeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Value keyValues = op.getKeyValues();
    auto move = keyValues.hasOneUse() ? dyn_cast<ToMemoryConfigOp>(
                                            *keyValues.getUsers().begin())
                                      : nullptr;
    if (!move || move.getType() == keyValues.getType()) {
      return mlir::failure();
    }
    auto layout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(move.getType().getEncoding());
    if (!layout || !layout.isTiled() ||
        layout.getDataType() != ttcore::DataType::BFloat16) {
      return mlir::failure();
    }
    rewriter.modifyOpInPlace(op, [&] { keyValues.setType(move.getType()); });
    rewriter.replaceOp(move, keyValues);
    return mlir::success();
  }
};

} // namespace

void SplitQKVNormRopeDecodeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<WriteKeyValuesInPlace>(context);
}

} // namespace mlir::tt::ttnn
