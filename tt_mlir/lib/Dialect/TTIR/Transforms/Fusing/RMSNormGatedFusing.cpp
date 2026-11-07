// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/RMSNormGatedFusing.h"

#include <optional>
#include <tuple>
#include <utility>

namespace mlir::tt::ttir::fusing {

namespace {

constexpr int64_t maxGroup = 128;

RankedTensorType typeOf(Value value) {
  return mlir::cast<RankedTensorType>(value.getType());
}

bool isFloat(Value value) {
  Type element = typeOf(value).getElementType();
  return element.isBF16() || element.isF32();
}

// The single user of `value`, or nullptr.
Operation *singleUser(Value value) {
  return value.hasOneUse() ? *value.getUsers().begin() : nullptr;
}

// `value` before an exact widening typecast to FP32, or `value`.
Value throughWidening(Value value) {
  if (auto cast = value.getDefiningOp<TypecastOp>()) {
    if (typeOf(value).getElementType().isF32() && isFloat(cast.getInput())) {
      return cast.getInput();
    }
  }
  return value;
}

// The op whose result `value` is, through single-use reshapes and widening
// typecasts.
Operation *producerOf(Value value) {
  while (value.hasOneUse()) {
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      value = reshape.getInput();
    } else if (Value narrow = throughWidening(value); narrow != value) {
      value = narrow;
    } else {
      break;
    }
  }
  return value.getDefiningOp();
}

// The tensor `value` reshapes or casts between BF16 and FP32: the op reads
// the source's values, also where a typecast narrowed them to BF16 (one
// rounding fewer).
Value sourceOf(Value value) {
  while (true) {
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      value = reshape.getInput();
    } else if (auto cast = value.getDefiningOp<TypecastOp>();
               cast && isFloat(cast.getInput()) && isFloat(value)) {
      value = cast.getInput();
    } else {
      return value;
    }
  }
}

int64_t product(ArrayRef<int64_t> dims) {
  int64_t result = 1;
  for (int64_t dim : dims) {
    result *= dim;
  }
  return result;
}

// The tensor and the first of its last dimension's columns that `value`
// holds (through reshapes), when it is `rows` rows of those `width` columns:
// a slice of the last dimension, or of the heads of a reshape that splits
// the last dimension into heads of D (SGLang's z of its qkvz projection).
std::optional<std::pair<Value, int64_t>>
columnsOf(Value value, int64_t rows, int64_t width) {
  while (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    value = reshape.getInput();
  }
  auto slice = value.getDefiningOp<SliceStaticOp>();
  if (!slice) {
    return std::nullopt;
  }
  ArrayRef<int64_t> shape = typeOf(slice.getInput()).getShape();
  const int64_t rank = shape.size();
  std::optional<int64_t> dim;
  int64_t begin = 0;
  int64_t end = 0;
  for (int64_t d = 0; d < rank; ++d) {
    const int64_t b = mlir::cast<IntegerAttr>(slice.getBegins()[d]).getInt();
    const int64_t e = mlir::cast<IntegerAttr>(slice.getEnds()[d]).getInt();
    if (mlir::cast<IntegerAttr>(slice.getStep()[d]).getInt() != 1) {
      return std::nullopt;
    }
    if (b != 0 || e != shape[d]) {
      if (dim) {
        return std::nullopt;
      }
      dim = d;
      begin = b;
      end = e;
    }
  }
  if (!dim) {
    return std::nullopt;
  }
  if (*dim == rank - 1 && product(shape.drop_back()) == rows &&
      end - begin == width) {
    return std::pair(slice.getInput(), begin);
  }
  auto split = slice.getInput().getDefiningOp<ReshapeOp>();
  if (*dim != rank - 2 || !split) {
    return std::nullopt;
  }
  Value whole = split.getInput();
  ArrayRef<int64_t> wholeShape = typeOf(whole).getShape();
  const int64_t headDim = shape.back();
  if (wholeShape.back() != shape[rank - 2] * headDim ||
      product(wholeShape.drop_back()) != rows ||
      product(shape.drop_back(2)) != rows || (end - begin) * headDim != width) {
    return std::nullopt;
  }
  return std::pair(whole, begin * headDim);
}

// `value` reshaped to `shape` if it has another one.
Value withShape(PatternRewriter &rewriter, Location loc, Value value,
                ArrayRef<int64_t> shape) {
  RankedTensorType type = typeOf(value);
  if (type.getShape() == shape) {
    return value;
  }
  SmallVector<int32_t> dims(shape.begin(), shape.end());
  return rewriter.create<ReshapeOp>(
      loc,
      RankedTensorType::get(shape, type.getElementType(), type.getEncoding()),
      value, rewriter.getI32ArrayAttr(dims));
}

} // namespace

mlir::LogicalResult
RMSNormGatedFusing::matchAndRewrite(MultiplyOp op,
                                    mlir::PatternRewriter &rewriter) const {
  RankedTensorType type = typeOf(op.getResult());
  if (type.getRank() == 0) {
    return failure();
  }
  const int64_t group = type.getShape().back();
  if (group % 32 != 0 || group > maxGroup) {
    return failure();
  }
  for (auto [normed, gated] : {std::pair(op.getLhs(), op.getRhs()),
                               std::pair(op.getRhs(), op.getLhs())}) {
    auto norm = dyn_cast_or_null<RMSNormOp>(producerOf(normed));
    auto silu = dyn_cast_or_null<SiluOp>(producerOf(gated));
    if (!norm || !silu || !norm.getWeight() || norm.getBias() ||
        !norm.getResult().hasOneUse() || !silu.getResult().hasOneUse() ||
        norm.getNormalizedShape().size() != 1 ||
        norm.getNormalizedShape()[0] != group ||
        typeOf(norm.getResult()).getRank() == 0 ||
        typeOf(norm.getResult()).getShape().back() != group ||
        typeOf(silu.getResult()).getRank() == 0 ||
        typeOf(silu.getResult()).getShape().back() != group) {
      continue;
    }
    Value input = sourceOf(norm.getInput());
    Value gate = throughWidening(silu.getInput());
    Value weight = sourceOf(norm.getWeight());
    if (!isFloat(input) || !isFloat(gate) || !isFloat(weight) ||
        typeOf(input).getNumElements() != type.getNumElements() ||
        typeOf(gate).getNumElements() != type.getNumElements() ||
        typeOf(weight).getNumElements() != group) {
      continue;
    }
    // The op makes a typecast result in its type, and a reshape of it to
    // rows of several groups (heads) in that layout.
    Operation *root = op;
    Type element = type.getElementType();
    if (auto cast = dyn_cast_or_null<TypecastOp>(singleUser(op.getResult()));
        cast && isFloat(cast.getResult())) {
      root = cast;
      element = typeOf(cast.getResult()).getElementType();
    }
    while (auto reshape =
               dyn_cast_or_null<ReshapeOp>(singleUser(root->getResult(0)))) {
      if (typeOf(reshape.getResult()).getRank() == 0 ||
          typeOf(reshape.getResult()).getShape().back() % group != 0) {
        break;
      }
      root = reshape;
    }
    ArrayRef<int64_t> shape = typeOf(root->getResult(0)).getShape();
    const int64_t width = shape.back();
    // A gate of whole rows of a wider tensor (with the output's leading
    // dimensions) is read in place from its columns.
    const Location loc = op.getLoc();
    rewriter.setInsertionPoint(root);
    int64_t gateOffset = 0;
    auto columns = columnsOf(gate, product(shape.drop_back()), width);
    if (columns && columns->second % 32 == 0 &&
        typeOf(columns->first).getShape().drop_back() == shape.drop_back()) {
      std::tie(gate, gateOffset) = *columns;
    } else {
      gate = withShape(rewriter, loc, gate, shape);
    }
    auto fused = rewriter.create<RMSNormGatedOp>(
        loc, RankedTensorType::get(shape, element, type.getEncoding()),
        withShape(rewriter, loc, input, shape), gate,
        weight, norm.getEpsilonAttr(),
        rewriter.getI64IntegerAttr(gateOffset));
    rewriter.replaceOp(root, fused.getResult());
    return success();
  }
  return failure();
}

} // namespace mlir::tt::ttir::fusing
