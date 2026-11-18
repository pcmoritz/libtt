// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/SplitQKVNormRopeDecodeFusing.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir::fusing {

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

RankedTensorType typeOf(Value value) {
  return mlir::cast<RankedTensorType>(value.getType());
}

ArrayRef<int64_t> shapeOf(Value value) { return typeOf(value).getShape(); }

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
  if (typeOf(value).getNumElements() != tokens * rowElements) {
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

// A rotary embedding, as the RoPE fusions make it: a composite of (x, cos,
// sin).
ttcore::CompositeOp asRotary(Operation *op) {
  auto composite = dyn_cast_or_null<ttcore::CompositeOp>(op);
  return composite && composite.getCompositeName() == "rotary_embedding" &&
                 composite.getInputs().size() == 3 &&
                 composite->getNumResults() == 1
             ? composite
             : nullptr;
}

Value rotaryInput(ttcore::CompositeOp rope) { return rope.getInputs()[0]; }
Value rotaryCos(ttcore::CompositeOp rope) { return rope.getInputs()[1]; }
Value rotarySin(ttcore::CompositeOp rope) { return rope.getInputs()[2]; }

// The row of cos and sin a decode step's rotary embedding reads, if any.
IntegerAttr tokenIndexOf(ttcore::CompositeOp rope) {
  auto attributes = rope.getCompositeAttributes();
  return attributes ? attributes->getAs<IntegerAttr>("token_index") : nullptr;
}

// The heads `rope` rotates: all of their dims, or, in a partial rotary, a
// slice of their leading ones.
Value rotatedHeads(ttcore::CompositeOp rope, int64_t headDim) {
  const int64_t rotated = shapeOf(rotaryInput(rope)).back();
  Value x = throughReshapes(rotaryInput(rope));
  if (rotated == headDim) {
    return x;
  }
  auto slice = x.getDefiningOp<SliceStaticOp>();
  return slice && isLastDimSlice(x, slice.getInput(), 0, rotated)
             ? slice.getInput()
             : Value();
}

// The rms_norm whose heads `rope` rotates.
RMSNormOp normOf(ttcore::CompositeOp rope, int64_t headDim) {
  Value heads = rotatedHeads(rope, headDim);
  return heads ? throughReshapes(heads).getDefiningOp<RMSNormOp>() : nullptr;
}

// Whether `parts`, side by side, are the heads of `norm` with the dims `rope`
// rotates rotated: its result, then the rest of the heads. Each part has one
// head per row, so that concatenating them keeps each head's dims together:
// all heads' rotated dims, then all heads' others, would mix the heads.
bool rotates(ttcore::CompositeOp rope, RMSNormOp norm, ValueRange parts,
             int64_t headDim) {
  const int64_t rotated = shapeOf(rotaryInput(rope)).back();
  return normOf(rope, headDim) == norm && !parts.empty() &&
         shapeOf(parts[0]).back() == rotated &&
         throughReshapes(parts[0]) == rope->getResult(0) &&
         (rotated == headDim
              ? parts.size() == 1
              : parts.size() == 2 &&
                    shapeOf(parts[1]).back() == headDim - rotated &&
                    isLastDimSlice(parts[1], rotatedHeads(rope, headDim),
                                   rotated, headDim));
}

bool isLastDimConcat(ConcatOp concat) {
  const int64_t rank = shapeOf(concat.getResult()).size();
  return concat.getDim() == rank - 1 || concat.getDim() == -1;
}

// Whether the rotary embeddings' cos (or sin) tables `a` and `b` are the same:
// the same value, or two of the copies the RoPE fusion makes of one half
// table, its concat with itself along the last dim, of it or of a reshape of
// it. Only those ops are compared, down to the same half table: other ops of
// the same operands, such as two calls of a stateful function, can produce
// different tables.
bool sameTable(Value a, Value b) {
  if (a == b) {
    return true;
  }
  if (a.getType() != b.getType()) {
    return false;
  }
  auto half = [](Value value) -> Value {
    auto concat = value.getDefiningOp<ConcatOp>();
    if (!concat || !isLastDimConcat(concat) || concat.getInputs().size() != 2 ||
        concat.getInputs()[0] != concat.getInputs()[1]) {
      return Value();
    }
    return concat.getInputs()[0];
  };
  Value halfA = half(a);
  Value halfB = half(b);
  if (halfA && halfB) {
    return sameTable(halfA, halfB);
  }
  auto reshapeA = a.getDefiningOp<ReshapeOp>();
  auto reshapeB = b.getDefiningOp<ReshapeOp>();
  return reshapeA && reshapeB && reshapeA.getInput() == reshapeB.getInput();
}

// The rotary embeddings of norms of views of `qkv`.
SmallVector<ttcore::CompositeOp> rotariesOf(Value qkv) {
  SmallVector<ttcore::CompositeOp> rotaries;
  SmallVector<Value> work{qkv};
  llvm::DenseSet<Operation *> seen;
  while (!work.empty()) {
    Value value = work.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (!seen.insert(user).second) {
        continue;
      }
      if (auto rope = asRotary(user)) {
        rotaries.push_back(rope);
      } else if (mlir::isa<ReshapeOp, SliceStaticOp, RMSNormOp>(user)) {
        work.push_back(user->getResult(0));
      }
    }
  }
  return rotaries;
}

SmallVector<int32_t> i32Shape(ArrayRef<int64_t> shape) {
  return SmallVector<int32_t>(shape.begin(), shape.end());
}

} // namespace

mlir::LogicalResult SplitQKVNormRopeDecodeFusing::matchAndRewrite(
    ConcatOp concat, mlir::PatternRewriter &rewriter) const {
  // Not in decompositions, which hold the ops fusions replaced.
  if (utils::isInsideCompositeDecomposition(concat)) {
    return mlir::failure();
  }
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
  ttcore::CompositeOp kRope =
      asRotary(throughReshapes(kParts[0]).getDefiningOp());
  RMSNormOp kNorm = kRope ? normOf(kRope, headDim) : nullptr;
  if (!kNorm || !rotates(kRope, kNorm, kParts, headDim)) {
    return mlir::failure();
  }
  Value qkv = sourceOf(kNorm.getInput(), 0).first;
  // q's: the rotary embedding of the same cos, sin and token whose norm reads
  // the same projection.
  ttcore::CompositeOp qRope;
  RMSNormOp qNorm;
  for (ttcore::CompositeOp rope : rotariesOf(qkv)) {
    if (rope == kRope || tokenIndexOf(rope) != tokenIndexOf(kRope) ||
        !sameTable(rotaryCos(rope), rotaryCos(kRope)) ||
        !sameTable(rotarySin(rope), rotarySin(kRope))) {
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
  const SmallVector<int64_t> normalized{headDim};
  if (!qRope || !qNorm.getWeight() || !kNorm.getWeight() || qNorm.getBias() ||
      kNorm.getBias() || qNorm.getEpsilonAttr() != kNorm.getEpsilonAttr() ||
      qNorm.getNormalizedShape() != ArrayRef<int64_t>(normalized) ||
      kNorm.getNormalizedShape() != ArrayRef<int64_t>(normalized)) {
    return mlir::failure();
  }

  // The heads are views of one projection [B, (s Hq + 2 Hkv) D], each q head
  // followed by its gate head if s is 2.
  RankedTensorType qkvType = typeOf(qkv);
  const int64_t width = qkvType.getShape().back();
  const int64_t qElements = typeOf(qNorm.getInput()).getNumElements();
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
  Value cos = rotaryCos(qRope);
  Value sin = rotarySin(qRope);
  const int64_t rotated = shapeOf(rotaryInput(qRope)).back();
  IntegerAttr tokenIndex = tokenIndexOf(qRope);
  if (rotated % 64 != 0 || shapeOf(rotaryInput(kRope)).back() != rotated ||
      shapeOf(rotaryInput(qRope)) !=
          ArrayRef<int64_t>{1, tokens, numQHeads, rotated} ||
      shapeOf(rotaryInput(kRope)) !=
          ArrayRef<int64_t>{1, tokens, numKVHeads, rotated} ||
      shapeOf(cos) != ArrayRef<int64_t>{1, tokens, 1, rotated} ||
      shapeOf(sin) != shapeOf(cos) ||
      (tokens == 1 ? !tokenIndex || !tokenIndex.getValue().isZero()
                   : !!tokenIndex)) {
    return mlir::failure();
  }
  // The kernel reads and writes BF16, and norm weights of BF16 or, as 1 + w
  // of a zero-centered norm, FP32.
  Value qWeight = qNorm.getWeight();
  Value kWeight = kNorm.getWeight();
  Type bf16 = rewriter.getBF16Type();
  for (Value operand : {qkv, cos, sin}) {
    if (typeOf(operand).getElementType() != bf16) {
      return mlir::failure();
    }
  }
  for (Value weight : {qWeight, kWeight}) {
    Type element = typeOf(weight).getElementType();
    if ((element != bf16 && !element.isF32()) ||
        typeOf(weight).getNumElements() != headDim ||
        element != typeOf(qWeight).getElementType()) {
      return mlir::failure();
    }
  }

  // q: its rotated parts, possibly a pad of the last dim with +0 (the fused
  // kernel's zeros), reshapes to [1, B, Hq, q_width].
  Value qHeads = alongReshapes(qRope->getResult(0));
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
  // Without the pad, the query is as wide as the heads.
  Value qOut = qHeads;
  if (auto pad = dyn_cast_or_null<PadOp>(getSingleUser(qHeads))) {
    if (!pad.getValue().isPosZero()) {
      return mlir::failure();
    }
    ArrayRef<int32_t> padding = pad.getPadding();
    for (size_t i = 0; i + 1 < padding.size(); ++i) {
      if (padding[i] != 0) {
        return mlir::failure();
      }
    }
    qOut = alongReshapes(pad.getResult());
  }
  // The kernel writes q_width >= head_dim columns, a multiple of 32. (TTIR
  // allows a negative pad, a crop, and the first fusing runs before such pads
  // become slices.)
  const int64_t qWidth = shapeOf(qOut).back();
  if (shapeOf(qHeads).back() != headDim || qWidth < headDim ||
      qWidth % 32 != 0 ||
      shapeOf(qOut) != ArrayRef<int64_t>{1, tokens, numQHeads, qWidth} ||
      typeOf(qOut).getElementType() != bf16) {
    return mlir::failure();
  }
  // k and v of all tokens: the concat reshaped to [1, B, Hkv, 2 D].
  auto kvReshape =
      dyn_cast_or_null<ReshapeOp>(getSingleUser(concat.getResult()));
  Value kvOut = kvReshape ? alongReshapes(kvReshape.getResult()) : Value();
  if (!kvOut ||
      shapeOf(kvOut) !=
          ArrayRef<int64_t>{1, tokens, numKVHeads, 2 * headDim} ||
      typeOf(kvOut).getElementType() != bf16) {
    return mlir::failure();
  }

  // The fused op goes before the first user of its results, after its
  // operands.
  Block *block = concat->getBlock();
  Operation *first = nullptr;
  for (Value out : {qOut, kvOut}) {
    for (Operation *user : out.getUsers()) {
      if (user->getBlock() != block) {
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
    if (def && (def->getBlock() != block || !def->isBeforeInBlock(first))) {
      return mlir::failure();
    }
  }

  // The ops it replaces: those from its operands (k's copies of cos and sin
  // among them) to its results, all views, norms, rotary embeddings, concats
  // and pads.
  const llvm::DenseSet<Value> operands{
      qkv, qWeight, kWeight, cos, sin, rotaryCos(kRope), rotarySin(kRope)};
  llvm::SetVector<Operation *> replaced;
  SmallVector<Value> work{qOut, kvOut};
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (operands.contains(value)) {
      continue;
    }
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != block ||
        !(mlir::isa<ReshapeOp, SliceStaticOp, RMSNormOp, ConcatOp, PadOp>(op) ||
          asRotary(op))) {
      return mlir::failure();
    }
    if (replaced.insert(op)) {
      llvm::append_range(work, op->getOperands());
    }
  }
  SmallVector<Operation *> ops(replaced.begin(), replaced.end());
  llvm::sort(ops, [](Operation *a, Operation *b) {
    return a->isBeforeInBlock(b);
  });

  // The kernel's input is the projection's rows.
  Location loc = qRope.getLoc();
  rewriter.setInsertionPoint(first);
  Value rows = qkv;
  const SmallVector<int64_t> rowsShape{tokens, width};
  if (qkvType.getShape() != ArrayRef<int64_t>(rowsShape)) {
    rows = rewriter.create<ReshapeOp>(
        loc, RankedTensorType::get(rowsShape, qkvType.getElementType()), qkv,
        rewriter.getI32ArrayAttr(i32Shape(rowsShape)));
  }
  auto fused = rewriter.create<SplitQKVNormRopeDecodeOp>(
      loc, qOut.getType(), kvOut.getType(), rows, qWeight, kWeight, cos, sin,
      rewriter.getUI32IntegerAttr(numQHeads),
      rewriter.getUI32IntegerAttr(numKVHeads), qNorm.getEpsilonAttr());
  rewriter.replaceAllUsesWith(qOut, fused.getQuery());
  rewriter.replaceAllUsesWith(kvOut, fused.getKeyValues());
  // The replaced ops that feed nothing else now; the rotary embeddings are
  // not pure, so the rewrite driver would keep them.
  for (Operation *op : llvm::reverse(ops)) {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }
  return mlir::success();
}

} // namespace mlir::tt::ttir::fusing
