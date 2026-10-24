// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// A sum over the last dimension of a compare against a per-row threshold, or
// of a select on one, becomes a single ttir.masked_reduce:
//
//   sum(where(x <cmp> t, 1, 0), dim = last)   count of the x <cmp> t per row
//   sum(where(x <cmp> t, x, 0), dim = last)   sum of those x per row
//
// with t a [..., H, 1] value broadcast over the row, or a [..., H] value
// reshaped to that. The binary searches behind SGLang-JAX's top-k and top-p
// masks run such a reduction 31 times per step over the vocabulary: the
// compare, the select and the sum each pass over the [batch, vocab] input or
// an intermediate of its size, and each pass is latency-bound on its tiles.
// The fused kernel reads the input once. Its result takes the threshold's
// shape and, for a count summed as i32, that type, so the search's glue
// around it (a reshape in, a reshape and a typecast out) goes too.
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttir {
namespace {

// Whether every element of `value` is the constant `expected`.
bool isSplatOf(Value value, double expected) {
  Operation *op = value.getDefiningOp();
  if (!op) {
    return false;
  }
  if (isa<ZerosOp>(op)) {
    return expected == 0.0;
  }
  if (isa<OnesOp>(op)) {
    return expected == 1.0;
  }
  Attribute element;
  if (auto full = dyn_cast<FullOp>(op)) {
    element = full.getFillValue();
  } else if (auto constant = dyn_cast<ConstantOp>(op)) {
    auto elements = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (!elements || !elements.isSplat()) {
      return false;
    }
    element = elements.getSplatValue<Attribute>();
  } else {
    return false;
  }
  if (auto floatAttr = dyn_cast<FloatAttr>(element)) {
    return floatAttr.getValueAsDouble() == expected;
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(element)) {
    const APInt &bits = intAttr.getValue();
    return bits.getActiveBits() <= 63 &&
           static_cast<double>(bits.getZExtValue()) == expected;
  }
  return false;
}

// The per-row threshold behind a compare operand: a [..., H, 1] value of the
// input's type that reaches the compare as is or through a broadcast over the
// last dimension, or the [..., H] value such a one is a reshape of. Null when
// the operand is something else.
Value rowThreshold(Value operand, RankedTensorType inputType) {
  Value source = operand;
  if (auto broadcast = operand.getDefiningOp<BroadcastOp>()) {
    source = broadcast.getInput();
  }
  auto type = dyn_cast<RankedTensorType>(source.getType());
  if (!type || type.getRank() != inputType.getRank() ||
      type.getShape().back() != 1 ||
      type.getShape().drop_back() != inputType.getShape().drop_back() ||
      type.getElementType() != inputType.getElementType()) {
    return nullptr;
  }
  if (auto reshape = source.getDefiningOp<ReshapeOp>()) {
    auto rowType = cast<RankedTensorType>(reshape.getInput().getType());
    if (rowType.getShape() == inputType.getShape().drop_back()) {
      return reshape.getInput();
    }
  }
  return source;
}

std::optional<ttcore::MaskedCompare> compareOf(Operation *op) {
  if (isa<GreaterThanOp>(op)) {
    return ttcore::MaskedCompare::GT;
  }
  if (isa<GreaterEqualOp>(op)) {
    return ttcore::MaskedCompare::GE;
  }
  if (isa<LessThanOp>(op)) {
    return ttcore::MaskedCompare::LT;
  }
  if (isa<LessEqualOp>(op)) {
    return ttcore::MaskedCompare::LE;
  }
  return std::nullopt;
}

// The compare with its operands swapped: t <cmp> x as x <cmp'> t.
ttcore::MaskedCompare swapped(ttcore::MaskedCompare compare) {
  switch (compare) {
  case ttcore::MaskedCompare::GT:
    return ttcore::MaskedCompare::LT;
  case ttcore::MaskedCompare::GE:
    return ttcore::MaskedCompare::LE;
  case ttcore::MaskedCompare::LT:
    return ttcore::MaskedCompare::GT;
  case ttcore::MaskedCompare::LE:
    return ttcore::MaskedCompare::GE;
  }
  return compare;
}

struct RowCompare {
  Value input;
  Value threshold;
  ttcore::MaskedCompare compare;
};

// A compare of a [..., H, W] FP32 or BF16 tensor against a per-row threshold,
// in either operand order. The compare may have other users (two searches
// over one compare, say): the fused kernel repeats it in passing.
std::optional<RowCompare> matchRowCompare(Value condition) {
  Operation *op = condition.getDefiningOp();
  if (!op || op->getNumOperands() != 2) {
    return std::nullopt;
  }
  std::optional<ttcore::MaskedCompare> compare = compareOf(op);
  if (!compare) {
    return std::nullopt;
  }
  for (bool swap : {false, true}) {
    Value input = op->getOperand(swap ? 1 : 0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() < 2 ||
        !(inputType.getElementType().isF32() ||
          inputType.getElementType().isBF16())) {
      continue;
    }
    if (Value threshold = rowThreshold(op->getOperand(swap ? 0 : 1), inputType)) {
      return RowCompare{input, threshold, swap ? swapped(*compare) : *compare};
    }
  }
  return std::nullopt;
}

class MaskedReduceFusionPattern : public OpRewritePattern<SumOp> {
  using OpRewritePattern<SumOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(SumOp sum,
                                PatternRewriter &rewriter) const final {
    RankedTensorType sumInputType = sum.getInput().getType();
    const int64_t rank = sumInputType.getRank();
    std::optional<ArrayAttr> dims = sum.getDimArg();
    if (rank < 2 || !dims || dims->size() != 1) {
      return failure();
    }
    const int64_t dim = cast<IntegerAttr>((*dims)[0]).getInt();
    if (dim != rank - 1 && dim != -1) {
      return failure();
    }

    // The mask: a select of 1 / 0 or of x / 0 on the compare, or the compare
    // cast to the summed type. A BF16 select reaches the sum widened to FP32,
    // which the fused reduction produces anyway.
    std::optional<RowCompare> rowCompare;
    ttcore::MaskedMode mode = ttcore::MaskedMode::Count;
    Operation *masked = sum.getInput().getDefiningOp();
    if (!masked || !masked->hasOneUse()) {
      return failure();
    }
    if (auto widen = dyn_cast<TypecastOp>(masked)) {
      auto where = widen.getInput().getDefiningOp<WhereOp>();
      if (where && where->hasOneUse() &&
          where.getResult().getType().getElementType().isBF16() &&
          widen.getResult().getType().getElementType().isF32()) {
        masked = where;
      }
    }
    if (auto where = dyn_cast<WhereOp>(masked)) {
      rowCompare = matchRowCompare(where.getFirst());
      if (!rowCompare || !isSplatOf(where.getThird(), 0.0)) {
        return failure();
      }
      if (where.getSecond() == rowCompare->input) {
        mode = ttcore::MaskedMode::Sum;
      } else if (!isSplatOf(where.getSecond(), 1.0)) {
        return failure();
      }
    } else if (auto typecast = dyn_cast<TypecastOp>(masked)) {
      rowCompare = matchRowCompare(typecast.getInput());
      if (!rowCompare) {
        return failure();
      }
    } else {
      return failure();
    }
    auto inputType = cast<RankedTensorType>(rowCompare->input.getType());
    if (inputType.getShape() != sumInputType.getShape() ||
        (mode == ttcore::MaskedMode::Sum &&
         cast<RankedTensorType>(masked->getResult(0).getType())
                 .getElementType() != inputType.getElementType())) {
      return failure();
    }

    // The result is shaped like the threshold, in FP32 or, for a count the
    // program sums as i32, in i32.
    Location loc = sum.getLoc();
    RankedTensorType sumType = sum.getResult().getType();
    ArrayRef<int64_t> reducedShape =
        cast<RankedTensorType>(rowCompare->threshold.getType()).getShape();
    const bool countsAsI32 = mode == ttcore::MaskedMode::Count &&
                             sumType.getElementType().isInteger(32);
    Value result = rewriter.create<MaskedReduceOp>(
        loc,
        RankedTensorType::get(reducedShape, countsAsI32
                                                ? sumType.getElementType()
                                                : rewriter.getF32Type()),
        rowCompare->input, rowCompare->threshold,
        ttcore::MaskedCompareAttr::get(getContext(), rowCompare->compare),
        ttcore::MaskedModeAttr::get(getContext(), mode));
    if (!sumType.getElementType().isF32() && !countsAsI32) {
      result = rewriter.create<TypecastOp>(
          loc, RankedTensorType::get(reducedShape, sumType.getElementType()),
          result);
    }
    if (sumType.getShape() != reducedShape) {
      result = rewriter.create<ReshapeOp>(
          loc, sumType, result,
          rewriter.getI32ArrayAttr(SmallVector<int32_t>(
              sumType.getShape().begin(), sumType.getShape().end())));
    }
    rewriter.replaceOp(sum, result);
    return success();
  }
};

// A threshold that is the bit pattern of 32-bit integers (a binary search's
// candidate) reaches masked_reduce as those integers: the kernel copies the
// threshold's bits as the input's FP32 values anyway, so the bitcast, a
// program of its own, goes. A negated one becomes the integers with their sign
// bit flipped (-x and x ^ 0x80000000 are one bit pattern, NaNs and zeros
// included), which elementwise integer fusion folds into the integers' own
// producer.
class MaskedReduceIntegerThresholdPattern
    : public OpRewritePattern<MaskedReduceOp> {
  using OpRewritePattern<MaskedReduceOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(MaskedReduceOp op,
                                PatternRewriter &rewriter) const final {
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    auto thresholdType = cast<RankedTensorType>(op.getThreshold().getType());
    if (!inputType.getElementType().isF32() ||
        !thresholdType.getElementType().isF32()) {
      return failure();
    }
    Value threshold = op.getThreshold();
    bool negated = false;
    if (auto neg = threshold.getDefiningOp<NegOp>()) {
      threshold = neg.getInput();
      negated = true;
    }
    auto bitcast = threshold.getDefiningOp<BitcastConvertOp>();
    if (!bitcast) {
      return failure();
    }
    Value bits = bitcast.getInput();
    auto bitsType = cast<RankedTensorType>(bits.getType());
    if (!bitsType.getElementType().isInteger(32) ||
        bitsType.getShape() != thresholdType.getShape()) {
      return failure();
    }
    if (negated) {
      Value sign = rewriter.create<FullOp>(
          op.getLoc(), bitsType,
          rewriter.getI32IntegerAttr(static_cast<int32_t>(0x80000000u)));
      bits = rewriter.create<BitwiseXorOp>(op.getLoc(), bitsType, bits, sign);
    }
    rewriter.modifyOpInPlace(op, [&] { op.getThresholdMutable().set(bits); });
    return success();
  }
};

} // namespace

void populateMaskedReduceFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<MaskedReduceFusionPattern>(patterns.getContext());
  patterns.add<MaskedReduceIntegerThresholdPattern>(patterns.getContext());
}

} // namespace mlir::tt::ttir
