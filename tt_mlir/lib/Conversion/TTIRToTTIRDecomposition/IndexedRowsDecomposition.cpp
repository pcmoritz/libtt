// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// The TTIR ops that move elements by index - torch-style ttir.gather and
// ttir.scatter, ttir.embedding and ttir.embedding_backward - onto
// ttir.rows_gather and ttir.rows_scatter, so that every indexed data movement
// runs on the indexed-rows kernels and ttnn.gather, ttnn.scatter,
// ttnn.embedding and ttnn.embedding_backward are never emitted. StableHLO
// gathers and scatters never produce these ops (IndexedRows.cpp lowers them
// directly); the remaining producers are upstream composites (top-k) and
// select_and_scatter.
//
// An embedding is a row lookup as is. A torch gather or scatter addresses
// elements: the operand is viewed as rows of its last dimension, and each
// element is addressed by its row (from the index tensor and the coordinates
// of the other leading dimensions) and its column (a one-element window).

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <numeric>

namespace mlir::tt {
namespace {

int64_t volume(ArrayRef<int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

Value asInt32(ConversionPatternRewriter &rewriter, Location loc, Value value) {
  auto type = mlir::cast<RankedTensorType>(value.getType());
  if (type.getElementType().isInteger(32)) {
    return value;
  }
  return rewriter.create<ttir::TypecastOp>(
      loc, RankedTensorType::get(type.getShape(), rewriter.getI32Type()),
      value);
}

// The row and column, in the operand viewed as [rows, last dim], of every
// element of `index`, whose value is the coordinate along `dim`; the other
// coordinates are the element's own. Returned as an int32 [1, positions, 2]
// index tensor of (row, column) components.
Value rowColumnIndices(ConversionPatternRewriter &rewriter, Location loc,
                       Value index, ArrayRef<int64_t> operandShape,
                       int64_t dim) {
  auto indexType = mlir::cast<RankedTensorType>(index.getType());
  const int64_t rank = operandShape.size();
  const int64_t last = rank - 1;
  const int64_t width = operandShape[last];
  auto i32Type =
      RankedTensorType::get(indexType.getShape(), rewriter.getI32Type());
  SmallVector<int64_t> strides = mlir::computeStrides(operandShape);
  Value index32 = asInt32(rewriter, loc, index);

  auto coordinate = [&](int64_t d) -> Value {
    return d == dim ? index32
                    : rewriter.create<ttir::ArangeOp>(
                          loc, i32Type, /*start=*/0,
                          /*end=*/indexType.getDimSize(d), /*step=*/1,
                          /*arange_dimension=*/d);
  };
  Value row;
  for (int64_t d = 0; d < last; ++d) {
    if (d != dim && indexType.getDimSize(d) == 1) {
      continue; // a size-1 coordinate is 0
    }
    Value term = coordinate(d);
    const int64_t rowStride = strides[d] / width;
    if (rowStride != 1) {
      Value stride = rewriter.create<ttir::FullOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(rowStride));
      term = rewriter.create<ttir::MultiplyOp>(loc, i32Type, term, stride);
    }
    row = row ? rewriter.create<ttir::AddOp>(loc, i32Type, row, term) : term;
  }
  if (!row) {
    row = rewriter.create<ttir::FullOp>(loc, i32Type,
                                        rewriter.getI32IntegerAttr(0));
  }
  Value column = coordinate(last);

  const int64_t positions = volume(indexType.getShape());
  auto componentType =
      RankedTensorType::get({1, positions, 1}, rewriter.getI32Type());
  row = ttir::utils::createReshapeOp(rewriter, loc, row,
                                     componentType.getShape());
  column = ttir::utils::createReshapeOp(rewriter, loc, column,
                                        componentType.getShape());
  return rewriter.create<ttir::ConcatOp>(
      loc, componentType.clone({1, positions, 2}), ValueRange{row, column},
      rewriter.getSI32IntegerAttr(2));
}

Value asRows(ConversionPatternRewriter &rewriter, Location loc, Value value,
             int64_t rows, int64_t width) {
  return ttir::utils::createReshapeOp(rewriter, loc, value, {1, rows, width});
}

// The operand as a [1, rows, last dim] table, a view of its pages.
std::pair<int64_t, int64_t> tableDims(RankedTensorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  int64_t width = shape.empty() ? 1 : shape.back();
  return {volume(shape) / width, width};
}

// ttir.gather: result[c] = input[c with c[dim] = index[c]].
class GatherToRowsPattern : public OpConversionPattern<ttir::GatherOp> {
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto indexType = mlir::cast<RankedTensorType>(adaptor.getIndex().getType());
    int64_t dim = op.getDim();
    if (dim < 0) {
      dim += inputType.getRank();
    }
    auto [rows, width] = tableDims(inputType);
    int64_t positions = volume(indexType.getShape());
    Value table = asRows(rewriter, loc, adaptor.getInput(), rows, width);
    Value indices = rowColumnIndices(rewriter, loc, adaptor.getIndex(),
                                     inputType.getShape(), dim);
    Value gathered = rewriter.create<ttir::RowsGatherOp>(
        loc, RankedTensorType::get({1, positions}, inputType.getElementType()),
        table, indices, rewriter.getDenseI64ArrayAttr({rows, width}),
        rewriter.getDenseI64ArrayAttr({1, 1}), rewriter.getBoolAttr(true),
        rewriter.getBoolAttr(false));
    rewriter.replaceOp(op, ttir::utils::createReshapeOp(
                               rewriter, loc, gathered,
                               op.getType().getShape()));
    return success();
  }
};

// ttir.scatter: input[c with c[dim] = index[c]] = source[c], or reduced.
class ScatterToRowsPattern : public OpConversionPattern<ttir::ScatterOp> {
  using OpConversionPattern<ttir::ScatterOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto indexType = mlir::cast<RankedTensorType>(adaptor.getIndex().getType());
    int64_t dim = op.getDim();
    if (dim < 0) {
      dim += inputType.getRank();
    }
    auto [rows, width] = tableDims(inputType);
    int64_t positions = volume(indexType.getShape());
    Value table = asRows(rewriter, loc, adaptor.getInput(), rows, width);
    Value indices = rowColumnIndices(rewriter, loc, adaptor.getIndex(),
                                     inputType.getShape(), dim);
    Value source = ttir::utils::createReshapeOp(rewriter, loc,
                                                adaptor.getSource(),
                                                {1, positions});
    Value scattered = rewriter.create<ttir::RowsScatterOp>(
        loc, table.getType(), table, indices, source,
        rewriter.getDenseI64ArrayAttr({rows, width}),
        rewriter.getDenseI64ArrayAttr({1, 1}), rewriter.getBoolAttr(true),
        op.getScatterReduceTypeAttr());
    rewriter.replaceOp(op, ttir::utils::createReshapeOp(
                               rewriter, loc, scattered,
                               op.getType().getShape()));
    return success();
  }
};

// ttir.embedding: result[i, :] = weight[input[i], :].
class EmbeddingToRowsPattern : public OpConversionPattern<ttir::EmbeddingOp> {
  using OpConversionPattern<ttir::EmbeddingOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::EmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto weightType =
        mlir::cast<RankedTensorType>(adaptor.getWeight().getType());
    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto [rows, width] = tableDims(weightType);
    int64_t positions = volume(inputType.getShape());
    Value table = asRows(rewriter, loc, adaptor.getWeight(), rows, width);
    Value indices =
        asRows(rewriter, loc, asInt32(rewriter, loc, adaptor.getInput()),
               positions, 1);
    Value gathered = rewriter.create<ttir::RowsGatherOp>(
        loc,
        RankedTensorType::get({1, positions, width}, weightType.getElementType()),
        table, indices, rewriter.getDenseI64ArrayAttr({rows}),
        rewriter.getDenseI64ArrayAttr({1}), rewriter.getBoolAttr(false),
        rewriter.getBoolAttr(false));
    rewriter.replaceOp(op, ttir::utils::createReshapeOp(
                               rewriter, loc, gathered,
                               op.getType().getShape()));
    return success();
  }
};

// ttir.embedding_backward: the weight gradient, zero except for the summed
// in_gradient rows at the input's indices.
class EmbeddingBackwardToRowsPattern
    : public OpConversionPattern<ttir::EmbeddingBackwardOp> {
  using OpConversionPattern<ttir::EmbeddingBackwardOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(ttir::EmbeddingBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(op.getType());
    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    auto [rows, width] = tableDims(resultType);
    int64_t positions = volume(inputType.getShape());
    auto tableType =
        RankedTensorType::get({1, rows, width}, resultType.getElementType());
    Attribute zero = mlir::isa<FloatType>(resultType.getElementType())
                         ? Attribute(rewriter.getF32FloatAttr(0.0f))
                         : Attribute(rewriter.getI32IntegerAttr(0));
    Value table = rewriter.create<ttir::FullOp>(loc, tableType, zero);
    Value indices =
        asRows(rewriter, loc, asInt32(rewriter, loc, adaptor.getInput()),
               positions, 1);
    Value source = adaptor.getInGradient();
    if (mlir::cast<RankedTensorType>(source.getType()).getElementType() !=
        resultType.getElementType()) {
      source = rewriter.create<ttir::TypecastOp>(
          loc,
          RankedTensorType::get(
              mlir::cast<RankedTensorType>(source.getType()).getShape(),
              resultType.getElementType()),
          source);
    }
    source = asRows(rewriter, loc, source, positions, width);
    Value gradient = rewriter.create<ttir::RowsScatterOp>(
        loc, tableType, table, indices, source,
        rewriter.getDenseI64ArrayAttr({rows}),
        rewriter.getDenseI64ArrayAttr({1}), rewriter.getBoolAttr(false),
        ttcore::ReduceTypeAttr::get(rewriter.getContext(),
                                    ttcore::ReduceType::Sum));
    rewriter.replaceOp(op, ttir::utils::createReshapeOp(
                               rewriter, loc, gradient, resultType.getShape()));
    return success();
  }
};

} // namespace

void populateIndexedRowsDecompositionPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  patterns.add<GatherToRowsPattern, ScatterToRowsPattern,
               EmbeddingToRowsPattern, EmbeddingBackwardToRowsPattern>(
      typeConverter, ctx);
}

} // namespace mlir::tt
