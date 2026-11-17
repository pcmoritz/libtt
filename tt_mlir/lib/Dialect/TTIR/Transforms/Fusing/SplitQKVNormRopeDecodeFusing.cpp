// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/SplitQKVNormRopeDecodeFusing.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
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
// rotates rotated: its result, then the rest of the heads.
bool rotates(ttcore::CompositeOp rope, RMSNormOp norm, ValueRange parts,
             int64_t headDim) {
  const int64_t rotated = shapeOf(rotaryInput(rope)).back();
  return normOf(rope, headDim) == norm && !parts.empty() &&
         throughReshapes(parts[0]) == rope->getResult(0) &&
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

// Whether `a` and `b` hold the same tensor: the same value, or results of
// equal ops of the same tensors. Each rotary embedding has its own copy of
// the cos and sin rows.
bool sameValue(Value a, Value b, unsigned depth = 8) {
  if (a == b) {
    return true;
  }
  auto resultA = mlir::dyn_cast<OpResult>(a);
  auto resultB = mlir::dyn_cast<OpResult>(b);
  if (!resultA || !resultB || depth == 0 || a.getType() != b.getType() ||
      resultA.getResultNumber() != resultB.getResultNumber()) {
    return false;
  }
  return OperationEquivalence::isEquivalentTo(
      resultA.getOwner(), resultB.getOwner(),
      [&](Value x, Value y) { return mlir::success(sameValue(x, y, depth - 1)); },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
}

// The rotary embeddings of norms of views of `qkv`.
SmallVector<ttcore::CompositeOp> rotariesOf(Value qkv) {
  SmallVector<ttcore::CompositeOp> rotaries;
  SmallVector<Value> work{qkv};
  llvm::DenseSet<Operation *> seen;
  while (!work.empty() && seen.size() < 256) {
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
        !sameValue(rotaryCos(rope), rotaryCos(kRope)) ||
        !sameValue(rotarySin(rope), rotarySin(kRope))) {
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
  const int64_t tableRows = tokens == 1 ? 1 : tokens;
  IntegerAttr tokenIndex = tokenIndexOf(qRope);
  if (rotated % 64 != 0 || shapeOf(rotaryInput(kRope)).back() != rotated ||
      shapeOf(rotaryInput(qRope)) !=
          ArrayRef<int64_t>{1, tokens, numQHeads, rotated} ||
      shapeOf(rotaryInput(kRope)) !=
          ArrayRef<int64_t>{1, tokens, numKVHeads, rotated} ||
      shapeOf(cos) != ArrayRef<int64_t>{1, tableRows, 1, rotated} ||
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
  const int64_t qWidth = shapeOf(qOut).back();
  if (shapeOf(qHeads).back() != headDim || qWidth % 32 != 0 ||
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

  // The composite goes before the first user of its results, after its
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
  llvm::DenseMap<Value, unsigned> argument{
      {qkv, 0},  {qWeight, 1}, {kWeight, 2}, {cos, 3}, {sin, 4},
      {rotaryCos(kRope), 3},   {rotarySin(kRope), 4}};
  llvm::SetVector<Operation *> replaced;
  SmallVector<Value> work{qOut, kvOut};
  while (!work.empty()) {
    Value value = work.pop_back_val();
    if (argument.count(value)) {
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
  SmallVector<Value> operands{rows, qWeight, kWeight, cos, sin};
  SmallVector<Type> resultTypes{qOut.getType(), kvOut.getType()};

  // The decomposition: the replaced ops, reading the composite's operands.
  auto module = concat->getParentOfType<ModuleOp>();
  auto decomposition = func::FuncOp::create(
      loc, (kSplitQKVNormRopeDecode + "_decomp").str(),
      rewriter.getFunctionType(ValueRange(operands).getTypes(), resultTypes));
  decomposition.setVisibility(SymbolTable::Visibility::Private);
  decomposition->setAttr(utils::kCompositeDecompositionAttr,
                         rewriter.getUnitAttr());
  SymbolTable(module).insert(decomposition);
  Block *body = decomposition.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockBegin(body);
  IRMapping mapping;
  for (auto [value, index] : argument) {
    mapping.map(value, body->getArgument(index));
  }
  if (rows != qkv) {
    mapping.map(qkv, builder.create<ReshapeOp>(
                         loc, qkvType, body->getArgument(0),
                         builder.getI32ArrayAttr(i32Shape(qkvType.getShape()))));
  }
  for (Operation *op : ops) {
    builder.clone(*op, mapping);
  }
  builder.create<func::ReturnOp>(
      loc, ValueRange{mapping.lookup(qOut), mapping.lookup(kvOut)});

  NamedAttrList attributes;
  attributes.append("num_q_heads", rewriter.getUI32IntegerAttr(numQHeads));
  attributes.append("num_kv_heads", rewriter.getUI32IntegerAttr(numKVHeads));
  attributes.append("epsilon", qNorm.getEpsilonAttr());
  // Its fused op needs no op-model validation (at O0, composites are inlined
  // unless they say so).
  attributes.append("promote_unvalidated", rewriter.getUnitAttr());
  auto composite = rewriter.create<ttcore::CompositeOp>(
      loc, resultTypes, operands,
      rewriter.getStringAttr(kSplitQKVNormRopeDecode),
      FlatSymbolRefAttr::get(decomposition.getSymNameAttr()),
      attributes.getDictionary(rewriter.getContext()));
  rewriter.replaceAllUsesWith(qOut, composite.getResult(0));
  rewriter.replaceAllUsesWith(kvOut, composite.getResult(1));
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
