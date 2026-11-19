// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/SplitQKVNormRopeDecodeFusing.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

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

RankedTensorType typeOf(Value value) {
  return mlir::cast<RankedTensorType>(value.getType());
}

ArrayRef<int64_t> shapeOf(Value value) { return typeOf(value).getShape(); }

// Along the chain of single-user reshapes from `value`: the first value of
// `shape`, or the last value of the chain.
Value alongReshapesTo(Value value, ArrayRef<int64_t> shape) {
  while (shapeOf(value) != shape) {
    auto reshape = dyn_cast_or_null<ReshapeOp>(getSingleUser(value));
    if (!reshape) {
      break;
    }
    value = reshape.getResult();
  }
  return value;
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

// The op of type T that is the single user of a value along the chain of
// single-user reshapes from `value`, if any.
template <typename T>
T alongReshapesUntil(Value value) {
  for (;;) {
    Operation *user = getSingleUser(value);
    if (auto op = dyn_cast_or_null<T>(user)) {
      return op;
    }
    auto reshape = dyn_cast_or_null<ReshapeOp>(user);
    if (!reshape) {
      return nullptr;
    }
    value = reshape.getResult();
  }
}

// A head's rotated parts and where they end: the rotary embedding's result,
// or, in a partial rotary, the parts the concat of the last dim after it puts
// side by side, and that concat's result.
std::optional<std::pair<SmallVector<Value>, Value>>
rotatedParts(ttcore::CompositeOp rope, int64_t rotated, int64_t headDim) {
  Value result = rope->getResult(0);
  if (rotated == headDim) {
    return std::make_pair(SmallVector<Value>{result}, result);
  }
  auto concat = alongReshapesUntil<ConcatOp>(result);
  if (!concat || !isLastDimConcat(concat)) {
    return std::nullopt;
  }
  return std::make_pair(SmallVector<Value>(concat.getInputs().begin(),
                                           concat.getInputs().end()),
                        concat.getResult());
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

// What the two forms of the prologue share: the projection and q's side.
struct Prologue {
  Value qkv;
  RMSNormOp qNorm;
  ttcore::CompositeOp qRope;
  int64_t tokens;
  int64_t numQHeads;
  int64_t numKVHeads;
  int64_t headDim;
  // The projection's width, and where the v heads start in a row of it.
  int64_t width;
  int64_t vOffset;
  // q's output, [1, B, Hq, q_width].
  Value qOut;
};

// The rest of the prologue whose k heads, numKVHeads for each of `tokens`,
// `kNorm` normalizes and `kRope` rotates into `kParts`: the projection the
// norm reads, q's norm and rotary embedding of the same cos, sin and token,
// and q's output.
std::optional<Prologue> matchPrologue(ttcore::CompositeOp kRope,
                                      RMSNormOp kNorm, ValueRange kParts,
                                      int64_t tokens, int64_t numKVHeads,
                                      int64_t headDim) {
  if (tokens > 32 || numKVHeads > 32 || headDim % 32 != 0 ||
      !rotates(kRope, kNorm, kParts, headDim)) {
    return std::nullopt;
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
      return std::nullopt;
    }
    qRope = rope;
    qNorm = norm;
  }
  const SmallVector<int64_t> normalized{headDim};
  if (!qRope || !qNorm.getWeight() || !kNorm.getWeight() || qNorm.getBias() ||
      kNorm.getBias() || qNorm.getEpsilonAttr() != kNorm.getEpsilonAttr() ||
      qNorm.getNormalizedShape() != ArrayRef<int64_t>(normalized) ||
      kNorm.getNormalizedShape() != ArrayRef<int64_t>(normalized)) {
    return std::nullopt;
  }

  // The heads are views of one projection [B, (s Hq + 2 Hkv) D], each q head
  // followed by its gate head if s is 2.
  RankedTensorType qkvType = typeOf(qkv);
  const int64_t width = qkvType.getShape().back();
  const int64_t qElements = typeOf(qNorm.getInput()).getNumElements();
  if (qkvType.getNumElements() != tokens * width ||
      qElements % (tokens * headDim) != 0) {
    return std::nullopt;
  }
  const int64_t numQHeads = qElements / (tokens * headDim);
  const int64_t qStride = (width / headDim - 2 * numKVHeads) / numQHeads;
  const int64_t kOffset = numQHeads * qStride * headDim;
  if ((qStride != 1 && qStride != 2) ||
      width != (qStride * numQHeads + 2 * numKVHeads) * headDim ||
      !isHeadView(qNorm.getInput(), qkv, tokens, width, 0, numQHeads,
                  qStride * headDim, headDim) ||
      !isHeadView(kNorm.getInput(), qkv, tokens, width, kOffset, numKVHeads,
                  headDim, headDim)) {
    return std::nullopt;
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
    return std::nullopt;
  }
  // The kernel reads and writes BF16, and norm weights of BF16 or, as 1 + w
  // of a zero-centered norm, FP32.
  Value qWeight = qNorm.getWeight();
  Value kWeight = kNorm.getWeight();
  Type bf16 = BFloat16Type::get(kRope.getContext());
  for (Value operand : {qkv, cos, sin}) {
    if (typeOf(operand).getElementType() != bf16) {
      return std::nullopt;
    }
  }
  for (Value weight : {qWeight, kWeight}) {
    Type element = typeOf(weight).getElementType();
    if ((element != bf16 && !element.isF32()) ||
        typeOf(weight).getNumElements() != headDim ||
        element != typeOf(qWeight).getElementType()) {
      return std::nullopt;
    }
  }

  // q: its rotated parts, reshaped to [1, B, Hq, D], or padded in the last
  // dim with +0 (the fused kernel's zeros) and reshaped to [1, B, Hq,
  // q_width]. Any reshapes after that read the fused op's query.
  auto qParts = rotatedParts(qRope, rotated, headDim);
  if (!qParts || !rotates(qRope, qNorm, qParts->first, headDim)) {
    return std::nullopt;
  }
  auto pad = alongReshapesUntil<PadOp>(qParts->second);
  Value qHeads = pad ? pad.getInput()
                     : alongReshapesTo(qParts->second,
                                       {1, tokens, numQHeads, headDim});
  // Without the pad, the query is as wide as the heads.
  Value qOut = qHeads;
  if (pad) {
    if (!pad.getValue().isPosZero()) {
      return std::nullopt;
    }
    ArrayRef<int32_t> padding = pad.getPadding();
    for (size_t i = 0; i + 1 < padding.size(); ++i) {
      if (padding[i] != 0) {
        return std::nullopt;
      }
    }
    qOut = alongReshapesTo(
        pad.getResult(),
        {1, tokens, numQHeads, shapeOf(pad.getResult()).back()});
  }
  // The kernel writes q_width >= head_dim columns, a multiple of 32. (TTIR
  // allows a negative pad, a crop, and the first fusing runs before such pads
  // become slices.)
  const int64_t qWidth = shapeOf(qOut).back();
  if (shapeOf(qHeads).back() != headDim || qWidth < headDim ||
      qWidth % 32 != 0 ||
      shapeOf(qOut) != ArrayRef<int64_t>{1, tokens, numQHeads, qWidth} ||
      typeOf(qOut).getElementType() != bf16) {
    return std::nullopt;
  }
  return Prologue{qkv,       qNorm,      qRope,
                  tokens,    numQHeads,  numKVHeads,
                  headDim,   width,      kOffset + numKVHeads * headDim,
                  qOut};
}

// Replaces the prologue with one split_qkv_norm_rope_decode whose results
// replace `outs`: q's output, then the k and v output or the k and the v
// outputs.
mlir::LogicalResult rewritePrologue(Prologue prologue,
                                    ttcore::CompositeOp kRope, RMSNormOp kNorm,
                                    ArrayRef<Value> outs,
                                    mlir::PatternRewriter &rewriter) {
  Value qkv = prologue.qkv;
  Value qWeight = prologue.qNorm.getWeight();
  Value kWeight = kNorm.getWeight();
  Value cos = rotaryCos(prologue.qRope);
  Value sin = rotarySin(prologue.qRope);

  // The fused op goes before the first user of its results, after its
  // operands.
  Block *block = kRope->getBlock();
  Operation *first = nullptr;
  for (Value out : outs) {
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
  SmallVector<Value> work(outs.begin(), outs.end());
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
  Location loc = prologue.qRope.getLoc();
  rewriter.setInsertionPoint(first);
  Value rows = qkv;
  RankedTensorType qkvType = typeOf(qkv);
  const SmallVector<int64_t> rowsShape{prologue.tokens, prologue.width};
  if (qkvType.getShape() != ArrayRef<int64_t>(rowsShape)) {
    rows = rewriter.create<ReshapeOp>(
        loc, RankedTensorType::get(rowsShape, qkvType.getElementType()), qkv,
        rewriter.getI32ArrayAttr(i32Shape(rowsShape)));
  }
  SmallVector<Type> types;
  for (Value out : outs) {
    types.push_back(out.getType());
  }
  auto fused = rewriter.create<SplitQKVNormRopeDecodeOp>(
      loc, types, rows, qWeight, kWeight, cos, sin,
      rewriter.getUI32IntegerAttr(prologue.numQHeads),
      rewriter.getUI32IntegerAttr(prologue.numKVHeads),
      prologue.qNorm.getEpsilonAttr());
  for (auto [out, result] : llvm::zip_equal(outs, fused->getResults())) {
    rewriter.replaceAllUsesWith(out, result);
  }
  // The replaced ops that feed nothing else now; the rotary embeddings are
  // not pure, so the rewrite driver would keep them.
  for (Operation *op : llvm::reverse(ops)) {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }
  return mlir::success();
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

  // k's rotary embedding and norm, and the rest of the prologue.
  ttcore::CompositeOp kRope =
      asRotary(throughReshapes(kParts[0]).getDefiningOp());
  RMSNormOp kNorm = kRope ? normOf(kRope, headDim) : nullptr;
  if (!kNorm) {
    return mlir::failure();
  }
  std::optional<Prologue> prologue =
      matchPrologue(kRope, kNorm, kParts, tokens, numKVHeads, headDim);
  if (!prologue ||
      !isHeadView(vHeads, prologue->qkv, tokens, prologue->width,
                  prologue->vOffset, numKVHeads, headDim, headDim)) {
    return mlir::failure();
  }

  // k and v of all tokens: the concat reshaped to [1, B, Hkv, 2 D].
  const SmallVector<int64_t> outShape{1, tokens, numKVHeads, 2 * headDim};
  Value kvOut = alongReshapesTo(concat.getResult(), outShape);
  if (shapeOf(kvOut) != ArrayRef<int64_t>(outShape) ||
      typeOf(kvOut).getElementType() != rewriter.getBF16Type()) {
    return mlir::failure();
  }
  return rewritePrologue(*prologue, kRope, kNorm, {prologue->qOut, kvOut},
                         rewriter);
}

mlir::LogicalResult SplitQKVNormRopeDecodeSeparateKVFusing::matchAndRewrite(
    ttcore::CompositeOp op, mlir::PatternRewriter &rewriter) const {
  // k's rotary embedding of [1, B, Hkv, R], and its norm.
  ttcore::CompositeOp kRope = asRotary(op);
  if (!kRope || shapeOf(rotaryInput(kRope)).size() != 4 ||
      shapeOf(rotaryInput(kRope))[0] != 1) {
    return mlir::failure();
  }
  const int64_t tokens = shapeOf(rotaryInput(kRope))[1];
  const int64_t numKVHeads = shapeOf(rotaryInput(kRope))[2];
  const int64_t rotated = shapeOf(rotaryInput(kRope))[3];
  // A partial rotary rotates a slice of the normed heads.
  int64_t headDim = rotated;
  if (auto slice =
          throughReshapes(rotaryInput(kRope)).getDefiningOp<SliceStaticOp>()) {
    headDim = shapeOf(slice.getInput()).back();
  }
  RMSNormOp kNorm = normOf(kRope, headDim);
  if (!kNorm) {
    return mlir::failure();
  }

  // k's output: its rotated parts, reshaped to [1, B, Hkv, D]. Any reshapes
  // after that read the fused op's keys.
  Type bf16 = rewriter.getBF16Type();
  auto kParts = rotatedParts(kRope, rotated, headDim);
  if (!kParts) {
    return mlir::failure();
  }
  const SmallVector<int64_t> kvShape{1, tokens, numKVHeads, headDim};
  Value kOut = alongReshapesTo(kParts->second, kvShape);
  if (shapeOf(kOut) != ArrayRef<int64_t>(kvShape) ||
      typeOf(kOut).getElementType() != bf16) {
    return mlir::failure();
  }
  std::optional<Prologue> prologue = matchPrologue(
      kRope, kNorm, kParts->first, tokens, numKVHeads, headDim);
  if (!prologue) {
    return mlir::failure();
  }

  // v's output: the one view of the projection that holds its v heads as
  // [1, B, Hkv, D].
  Value vOut;
  SmallVector<Value> work{prologue->qkv};
  while (!work.empty()) {
    Value value = work.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (!mlir::isa<ReshapeOp, SliceStaticOp>(user)) {
        continue;
      }
      Value view = user->getResult(0);
      if (shapeOf(view) == ArrayRef<int64_t>(kvShape) &&
          isHeadView(view, prologue->qkv, tokens, prologue->width,
                     prologue->vOffset, numKVHeads, headDim, headDim)) {
        if (vOut) {
          return mlir::failure();
        }
        vOut = view;
        continue;
      }
      work.push_back(view);
    }
  }
  if (!vOut) {
    return mlir::failure();
  }
  return rewritePrologue(*prologue, kRope, kNorm,
                         {prologue->qOut, kOut, vOut}, rewriter);
}

} // namespace mlir::tt::ttir::fusing
