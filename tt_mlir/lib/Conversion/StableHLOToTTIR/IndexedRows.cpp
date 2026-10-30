// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// StableHLO gather, scatter and dynamic_update_slice onto ttir.rows_gather and
// ttir.rows_scatter.
//
// All three move rows of a table by index. The table is the operand viewed as
// [B, N, S..., R]: B leading dimensions whose index is shared with (or comes
// from) the index tensor's batch, N rows addressed by the index components,
// and the trailing dimensions copied per row, kept as they are so the table has
// the operand's pages (a row-major reshape that changes the last dimension
// copies the tensor). The index components address a contiguous block of
// operand dimensions, each with a window of consecutive positions (StableHLO's
// slice sizes or update window extents); operand dimensions inside that block
// that are not indexed get a constant zero component. When the block reaches
// the last dimension, the last component addresses a window of columns inside
// a row's page instead of flattening the operand to one-element rows. Only when
// the StableHLO batching dimensions do not lead the operand is it permuted, so
// a lookup in a large table never copies it. Everything else - permuting and
// reshaping indices, updates and results - touches the small operands only.
//
// Out-of-bounds handling is the kernel's: gathers clamp each component into
// the operand, scatters drop the update, exactly as StableHLO specifies, so no
// index arithmetic is emitted here beyond casting indices to int32 (uint32
// indices stay unsigned, see rowsIndexType).

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>

namespace mlir::tt {
namespace {

// Set on a stablehlo.gather whose out-of-bounds positions must read as zeros
// rather than clamp (see tt_mlir_sharded_gather.patch).
constexpr llvm::StringLiteral kFillOutOfBoundsAttr = "tt.fill_out_of_bounds";

int64_t product(ArrayRef<int64_t> values) {
  return std::accumulate(values.begin(), values.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

bool isIdentity(ArrayRef<int64_t> permutation) {
  return llvm::equal(permutation, llvm::seq<int64_t>(0, permutation.size()));
}

// The element type the rows ops read indices of `elementType` as: uint32 stays
// unsigned (the kernels compare it as unsigned, so values past INT32_MAX clamp
// high instead of reading as negative), every other integer type is int32.
Type rowsIndexType(Builder &builder, Type elementType) {
  return builder.getIntegerType(
      32, /*isSigned=*/!elementType.isUnsignedInteger(32));
}

// What an operand dimension is to the rows ops.
enum class DimKind {
  Batch,   // a StableHLO batching dimension, paired with an index dimension
  Indexed, // addressed by an index component
  Free,    // every position copied (or, if partial, sliced from zero)
};

struct OperandDim {
  DimKind kind;
  int64_t column = -1; // the index component for Indexed dimensions
  int64_t window = 1;  // positions taken along this dimension
  int64_t size = 1;
};

// The operand's [B, N, R] view.
struct Canonical {
  // Applied to the operand before the view; identity when not needed.
  SmallVector<int64_t> permutation;
  // Positions (of the permuted operand) in each region.
  SmallVector<int64_t> batchDims;
  SmallVector<int64_t> componentDims;
  SmallVector<int64_t> rowDims;
  // Per component: its index column (-1 for a constant zero), size, window.
  SmallVector<int64_t> columns;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> windows;
  // Whether the last component addresses columns inside the row's page.
  bool column = false;
  // Dimensions of the permuted operand.
  SmallVector<OperandDim> dims;
  int64_t B = 1, N = 1, W = 1; // W: over the row components
  int64_t columnWindow = 1;

  // The table's shape: [B, N, row dims...], or [B, N, R] with a column
  // component (R the last dimension's extent).
  SmallVector<int64_t> tableShape() const {
    SmallVector<int64_t> shape{B, N};
    if (column) {
      shape.push_back(sizes.back());
    } else {
      for (int64_t d : rowDims) {
        shape.push_back(dims[d].size);
      }
      if (rowDims.empty()) {
        shape.push_back(1);
      }
    }
    return shape;
  }

  // The shape of `positions` gathered rows: [B, positions * W, row dims...],
  // or [B, positions * W * columnWindow] with a column component.
  SmallVector<int64_t> gatheredShape(int64_t positions) const {
    SmallVector<int64_t> shape{B, positions * W};
    if (column) {
      shape.back() *= columnWindow;
    } else {
      for (int64_t d : rowDims) {
        shape.push_back(dims[d].size);
      }
      if (rowDims.empty()) {
        shape.push_back(1);
      }
    }
    return shape;
  }
};

// Lays the operand out as [batch dims][component dims][row dims]. The
// component region spans the first to the last dimension that is indexed or
// partially covered; a Free dimension inside it becomes a zero component. The
// operand keeps its order when its batching dimensions, if any, are exactly the
// dimensions before that region, apart from size-1 dimensions, which sit
// anywhere without moving data; otherwise it is permuted to [Batch, Indexed
// and partial Free, full Free].
Canonical canonicalize(ArrayRef<OperandDim> operandDims) {
  const int64_t rank = operandDims.size();
  auto needsComponent = [](const OperandDim &dim) {
    return dim.kind == DimKind::Indexed ||
           (dim.kind == DimKind::Free && dim.window != dim.size);
  };
  int64_t regionStart = rank, regionEnd = -1;
  for (int64_t d = 0; d < rank; ++d) {
    if (needsComponent(operandDims[d])) {
      regionStart = std::min(regionStart, d);
      regionEnd = std::max(regionEnd, d);
    }
  }
  bool hasBatch = llvm::any_of(
      operandDims, [](const OperandDim &dim) { return dim.kind == DimKind::Batch; });
  bool natural = true;
  for (int64_t d = 0; d < rank; ++d) {
    bool leading = d < regionStart;
    bool isBatch = operandDims[d].kind == DimKind::Batch;
    bool unit = operandDims[d].kind == DimKind::Free && operandDims[d].size == 1;
    if (isBatch && !leading) {
      natural = false;
    }
    if (hasBatch && leading && !isBatch && !unit) {
      natural = false;
    }
  }

  Canonical c;
  c.permutation = llvm::to_vector(llvm::seq<int64_t>(0, rank));
  if (!natural) {
    c.permutation.clear();
    for (DimKind kind : {DimKind::Batch, DimKind::Indexed, DimKind::Free}) {
      for (int64_t d = 0; d < rank; ++d) {
        bool component = needsComponent(operandDims[d]);
        bool take = kind == DimKind::Batch
                        ? operandDims[d].kind == DimKind::Batch
                        : kind == DimKind::Indexed
                              ? component
                              : operandDims[d].kind == DimKind::Free && !component;
        if (take) {
          c.permutation.push_back(d);
        }
      }
    }
  }
  for (int64_t d : c.permutation) {
    c.dims.push_back(operandDims[d]);
  }
  regionStart = rank;
  regionEnd = -1;
  for (int64_t d = 0; d < rank; ++d) {
    if (needsComponent(c.dims[d])) {
      regionStart = std::min(regionStart, d);
      regionEnd = std::max(regionEnd, d);
    }
  }
  // With nothing indexed, every position is copied: the last dimension is the
  // row and one zero component of size 1 stands in for the (empty) region.
  if (regionEnd < 0) {
    regionStart = rank - 1;
    regionEnd = rank - 2;
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (d < regionStart) {
      c.batchDims.push_back(d);
      c.B *= c.dims[d].size;
    } else if (d <= regionEnd) {
      c.componentDims.push_back(d);
      c.columns.push_back(c.dims[d].kind == DimKind::Indexed ? c.dims[d].column
                                                             : -1);
      c.sizes.push_back(c.dims[d].size);
      c.windows.push_back(c.dims[d].window);
    } else {
      c.rowDims.push_back(d);
    }
  }
  if (c.componentDims.empty()) {
    c.columns.push_back(-1);
    c.sizes.push_back(1);
    c.windows.push_back(1);
  }
  // A component on the last dimension addresses columns of the row's page.
  c.column = !c.componentDims.empty() && c.componentDims.back() == rank - 1;
  const size_t rowComponents = c.sizes.size() - (c.column ? 1 : 0);
  c.N = product(ArrayRef<int64_t>(c.sizes).take_front(rowComponents));
  c.W = product(ArrayRef<int64_t>(c.windows).take_front(rowComponents));
  c.columnWindow = c.column ? c.windows.back() : 1;
  return c;
}

// Emits the operand as the [B, N, R] table.
Value buildTable(ConversionPatternRewriter &rewriter, Location loc,
                 Value operand, const Canonical &c) {
  auto type = mlir::cast<RankedTensorType>(operand.getType());
  if (!isIdentity(c.permutation)) {
    SmallVector<int64_t> shape =
        ttmlir::utils::applyPermutation(type.getShape(), c.permutation);
    operand = rewriter.create<ttir::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_permute_table"),
        type.clone(shape), operand, c.permutation);
  }
  return ttir::utils::createReshapeOp(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_table"), operand,
      c.tableShape());
}

// Moves dimensions by `permutation`: a reshape when only size-1 dimensions
// move (their relative order among the others is immaterial), else a permute.
Value moveDims(ConversionPatternRewriter &rewriter, Location loc, Value value,
               ArrayRef<int64_t> permutation, StringRef suffix) {
  if (isIdentity(permutation)) {
    return value;
  }
  auto type = mlir::cast<RankedTensorType>(value.getType());
  SmallVector<int64_t> shape =
      ttmlir::utils::applyPermutation(type.getShape(), permutation);
  auto nonUnit = [&](auto dims) {
    return llvm::to_vector(llvm::make_filter_range(
        dims, [&](int64_t d) { return type.getDimSize(d) != 1; }));
  };
  if (nonUnit(llvm::seq<int64_t>(0, type.getRank())) == nonUnit(permutation)) {
    return ttir::utils::createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, suffix), value,
        shape);
  }
  return rewriter.create<ttir::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(loc, suffix), type.clone(shape),
      value, permutation);
}

// The operands of the concatenate along `dim` that defines `value`, if any:
// the converted op, or the StableHLO op when it is not converted yet.
SmallVector<Value> concatenatedAlong(ConversionPatternRewriter &rewriter,
                                     Value value, int64_t dim) {
  const int64_t rank = mlir::cast<RankedTensorType>(value.getType()).getRank();
  if (auto concat = value.getDefiningOp<ttir::ConcatOp>()) {
    const int64_t d = concat.getDim() < 0 ? concat.getDim() + rank : concat.getDim();
    if (d == dim) {
      return SmallVector<Value>(concat.getInputs());
    }
  } else if (auto concat = value.getDefiningOp<stablehlo::ConcatenateOp>()) {
    if (static_cast<int64_t>(concat.getDimension()) == dim) {
      // Only pieces that are converted already (so their types are TTIR's).
      SmallVector<Value> pieces;
      for (Value input : concat.getInputs()) {
        Value remapped = rewriter.getRemappedValue(input);
        if (!remapped || remapped == input) {
          return {};
        }
        pieces.push_back(remapped);
      }
      return pieces;
    }
  }
  return {};
}

// The index tensor as [Bi, K, M] int32 (uint32 for uint32 indices, see
// rowsIndexType): for each batch, one page per component (the kernels stream
// positions in chunks, one read per component).
// StableHLO keeps the components innermost, and JAX assembles that index
// vector by concatenating one [..., 1] piece per component; the pages are
// built from those pieces, so no intermediate has K-element rows (which tile
// to 32x32 and page to K elements).
// `indexBatchingDims[i]` is the index dimension paired with the i-th operand
// batching dimension in increasing operand order; the index dimensions other
// than those and `indexVectorDim` are the positions, in increasing order.
struct IndexLayout {
  Value indices;
  SmallVector<int64_t> batchDims; // index dims, in operand batching order
  SmallVector<int64_t> otherDims; // index dims, increasing
  int64_t Bi = 1, M = 1;
};

IndexLayout buildIndices(ConversionPatternRewriter &rewriter, Location loc,
                         Value indices, int64_t indexVectorDim,
                         ArrayRef<int64_t> indexBatchingDims,
                         const Canonical &c) {
  IndexLayout layout;
  auto type = mlir::cast<RankedTensorType>(indices.getType());
  const int64_t rank = type.getRank();
  ArrayRef<int64_t> shape = type.getShape();
  layout.batchDims.assign(indexBatchingDims.begin(), indexBatchingDims.end());
  for (int64_t d = 0; d < rank; ++d) {
    if (d != indexVectorDim && !llvm::is_contained(indexBatchingDims, d)) {
      layout.otherDims.push_back(d);
    }
  }
  for (int64_t d : layout.batchDims) {
    layout.Bi *= shape[d];
  }
  for (int64_t d : layout.otherDims) {
    layout.M *= shape[d];
  }
  const bool hasVector = indexVectorDim < rank;
  const int64_t K = hasVector ? shape[indexVectorDim] : 1;
  Type indexType = rowsIndexType(rewriter, type.getElementType());

  // [batch..., components, positions...], per piece of the component dim.
  SmallVector<int64_t> permutation(layout.batchDims);
  if (hasVector) {
    permutation.push_back(indexVectorDim);
  }
  llvm::append_range(permutation, layout.otherDims);
  SmallVector<Value> pieces;
  if (hasVector) {
    pieces = concatenatedAlong(rewriter, indices, indexVectorDim);
  }
  if (pieces.empty()) {
    pieces.push_back(indices);
  }
  SmallVector<Value> pages;
  for (Value piece : pieces) {
    auto pieceType = mlir::cast<RankedTensorType>(piece.getType());
    const int64_t components = hasVector ? pieceType.getDimSize(indexVectorDim) : 1;
    Value page = moveDims(rewriter, loc, piece, permutation, "_index_components");
    page = ttir::utils::createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_index_pages"),
        page, {layout.Bi, components, layout.M});
    if (pieceType.getElementType() != indexType) {
      page = rewriter.create<ttir::TypecastOp>(
          ttmlir::utils::appendLocationSuffix(loc, "_indices_i32"),
          RankedTensorType::get({layout.Bi, components, layout.M}, indexType),
          page);
    }
    pages.push_back(page);
  }
  auto pageType = RankedTensorType::get({layout.Bi, 1, layout.M}, indexType);
  auto concatPages = [&](ArrayRef<Value> values, StringRef suffix) -> Value {
    if (values.size() == 1) {
      return values.front();
    }
    int64_t total = 0;
    for (Value v : values) {
      total += mlir::cast<RankedTensorType>(v.getType()).getDimSize(1);
    }
    return rewriter.create<ttir::ConcatOp>(
        ttmlir::utils::appendLocationSuffix(loc, suffix),
        pageType.clone({layout.Bi, total, layout.M}), values,
        rewriter.getSI32IntegerAttr(1));
  };
  Value value = concatPages(pages, "_indices");

  // Components in canonical order, with zero pages for unindexed dimensions.
  bool identityColumns =
      static_cast<int64_t>(c.columns.size()) == K &&
      llvm::equal(c.columns, llvm::seq<int64_t>(0, K));
  if (identityColumns) {
    layout.indices = value;
    return layout;
  }
  SmallVector<Value> columns;
  Value zeros;
  for (int64_t column : c.columns) {
    if (column < 0) {
      if (!zeros) {
        zeros = rewriter.create<ttir::FullOp>(
            ttmlir::utils::appendLocationSuffix(loc, "_zero_column"),
            pageType, rewriter.getI32IntegerAttr(0));
      }
      columns.push_back(zeros);
      continue;
    }
    columns.push_back(rewriter.create<ttir::SliceStaticOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_column"), pageType, value,
        rewriter.getI32ArrayAttr({0, static_cast<int32_t>(column), 0}),
        rewriter.getI32ArrayAttr({static_cast<int32_t>(layout.Bi),
                                  static_cast<int32_t>(column + 1),
                                  static_cast<int32_t>(layout.M)}),
        rewriter.getI32ArrayAttr({1, 1, 1})));
  }
  layout.indices = concatPages(columns, "_columns");
  return layout;
}

// Reshapes a rows op result [B, M * W, R] back to a tensor with one dimension
// per table batch dim, index position dim, component window (except collapsed
// ones) and row dim, in that order, and permutes those dimensions to
// `order`, which names each result dimension in that enumeration.
struct ResultDims {
  // Indices into the flat enumeration below for each category.
  SmallVector<int64_t> batch, other, window, row;
  SmallVector<int64_t> shape;
};

ResultDims enumerateResultDims(const Canonical &c, const IndexLayout &layout,
                               ArrayRef<int64_t> indexShape,
                               ArrayRef<bool> componentInResult) {
  ResultDims r;
  for (int64_t d : c.batchDims) {
    r.batch.push_back(r.shape.size());
    r.shape.push_back(c.dims[d].size);
  }
  for (int64_t d : layout.otherDims) {
    r.other.push_back(r.shape.size());
    r.shape.push_back(indexShape[d]);
  }
  for (auto [i, window] : llvm::enumerate(c.windows)) {
    if (componentInResult[i]) {
      r.window.push_back(r.shape.size());
      r.shape.push_back(window);
    } else {
      r.window.push_back(-1);
    }
  }
  for (int64_t d : c.rowDims) {
    r.row.push_back(r.shape.size());
    r.shape.push_back(c.dims[d].size);
  }
  return r;
}

Value reshapeAndPermute(ConversionPatternRewriter &rewriter, Location loc,
                        Value value, ArrayRef<int64_t> shape,
                        ArrayRef<int64_t> permutation,
                        RankedTensorType resultType) {
  value = ttir::utils::createReshapeOp(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_unflatten"), value,
      shape);
  if (!isIdentity(permutation)) {
    value = rewriter.create<ttir::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_permute_result"),
        resultType.clone(ttmlir::utils::applyPermutation(shape, permutation)),
        value, permutation);
  }
  if (mlir::cast<RankedTensorType>(value.getType()).getShape() !=
      resultType.getShape()) {
    value = ttir::utils::createReshapeOp(rewriter, loc, value,
                                         resultType.getShape());
  }
  return value;
}

DenseI64ArrayAttr i64Array(ConversionPatternRewriter &rewriter,
                           ArrayRef<int64_t> values) {
  return rewriter.getDenseI64ArrayAttr(values);
}

Value zerosLike(ConversionPatternRewriter &rewriter, Location loc,
                RankedTensorType type) {
  Attribute zero = mlir::isa<FloatType>(type.getElementType())
                       ? Attribute(rewriter.getF32FloatAttr(0.0f))
                       : Attribute(rewriter.getI32IntegerAttr(0));
  return rewriter.create<ttir::FullOp>(loc, type, zero);
}

//===----------------------------------------------------------------------===//
// Gather
//===----------------------------------------------------------------------===//

class GatherConversionPattern
    : public OpConversionPattern<mlir::stablehlo::GatherOp> {
  using OpConversionPattern<mlir::stablehlo::GatherOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::GatherOp srcOp,
                  mlir::stablehlo::GatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = srcOp.getLoc();
    auto dimensionNumbers = srcOp.getDimensionNumbers();
    ArrayRef<int64_t> offsetDims = dimensionNumbers.getOffsetDims();
    ArrayRef<int64_t> collapsedDims = dimensionNumbers.getCollapsedSliceDims();
    ArrayRef<int64_t> operandBatchingDims =
        dimensionNumbers.getOperandBatchingDims();
    ArrayRef<int64_t> startIndicesBatchingDims =
        dimensionNumbers.getStartIndicesBatchingDims();
    ArrayRef<int64_t> startIndexMap = dimensionNumbers.getStartIndexMap();
    int64_t indexVectorDim = dimensionNumbers.getIndexVectorDim();
    ArrayRef<int64_t> sliceSizes = srcOp.getSliceSizes();

    Value operand = adaptor.getOperand();
    auto operandType = mlir::cast<RankedTensorType>(operand.getType());
    ArrayRef<int64_t> operandShape = operandType.getShape();
    auto indicesType =
        mlir::cast<RankedTensorType>(adaptor.getStartIndices().getType());
    auto resultType = mlir::cast<RankedTensorType>(
        getTypeConverter()->convertType(srcOp.getType()));
    if (!mlir::isa<IntegerType>(indicesType.getElementType())) {
      return rewriter.notifyMatchFailure(srcOp, "indices must be integers");
    }
    if (resultType.getNumElements() == 0) {
      rewriter.replaceOp(srcOp, zerosLike(rewriter, loc, resultType));
      return success();
    }
    // A complex operand arrives as floats with a trailing dimension of 2,
    // which is copied whole: an extra free dimension at the end of the row.
    const int64_t originalRank = srcOp.getOperand().getType().getRank();
    const int64_t extraDims = operandType.getRank() - originalRank;
    if (extraDims < 0 ||
        resultType.getRank() != srcOp.getType().getRank() + extraDims) {
      return rewriter.notifyMatchFailure(srcOp, "unexpected converted ranks");
    }

    // Classify the operand dimensions.
    SmallVector<OperandDim> dims(operandShape.size());
    for (auto [d, size] : llvm::enumerate(operandShape)) {
      OperandDim &dim = dims[d];
      dim.size = size;
      if (static_cast<int64_t>(d) >= originalRank) {
        dim.kind = DimKind::Free;
        dim.window = size;
        continue;
      }
      dim.window = sliceSizes[d];
      if (llvm::is_contained(operandBatchingDims, d)) {
        dim.kind = DimKind::Batch;
      } else if (auto *it = llvm::find(startIndexMap, d);
                 it != startIndexMap.end()) {
        dim.kind = DimKind::Indexed;
        dim.column = it - startIndexMap.begin();
      } else {
        dim.kind = DimKind::Free;
      }
    }
    Canonical c = canonicalize(dims);
    // Index dims paired with the operand batching dims, in increasing operand
    // dim order (the order the batch dims take in the table).
    SmallVector<int64_t> indexBatchingDims;
    for (int64_t d = 0; d < originalRank; ++d) {
      if (auto *it = llvm::find(operandBatchingDims, d);
          it != operandBatchingDims.end()) {
        indexBatchingDims.push_back(
            startIndicesBatchingDims[it - operandBatchingDims.begin()]);
      }
    }

    Value table = buildTable(rewriter, loc, operand, c);
    IndexLayout layout = buildIndices(rewriter, loc, adaptor.getStartIndices(),
                                      indexVectorDim, indexBatchingDims, c);
    if (layout.Bi != 1 && layout.Bi != c.B) {
      return rewriter.notifyMatchFailure(
          srcOp, "index batches do not match the table's");
    }

    auto gatherType = RankedTensorType::get(c.gatheredShape(layout.M),
                                            operandType.getElementType());
    // The sharded-table partitioner (UpdateGlobalToLocalShapes) marks the
    // gathers whose indices are relative to the device's shard: positions in
    // other shards read as zeros and the shards' results are summed.
    bool fillOutOfBounds = srcOp->hasAttr(kFillOutOfBoundsAttr);
    Value gathered = rewriter.create<ttir::RowsGatherOp>(
        loc, gatherType, table, layout.indices, i64Array(rewriter, c.sizes),
        i64Array(rewriter, c.windows), rewriter.getBoolAttr(c.column),
        rewriter.getBoolAttr(fillOutOfBounds));

    // Back to the StableHLO result: offset dims hold the operand's
    // non-collapsed, non-batching dims in operand order; the other positions
    // hold the index dims (without the index vector) in index order.
    SmallVector<bool> componentInResult;
    for (int64_t d : c.componentDims) {
      componentInResult.push_back(!llvm::is_contained(
          collapsedDims, c.permutation[d]));
    }
    if (c.componentDims.empty()) {
      componentInResult.push_back(false);
    }
    ResultDims r = enumerateResultDims(c, layout, indicesType.getShape(),
                                       componentInResult);
    // Where each original operand dim and index dim landed.
    SmallVector<int64_t> permutation(resultType.getRank(), -1);
    SmallVector<int64_t> operandOffsetDims;
    for (int64_t d = 0; d < originalRank; ++d) {
      if (!llvm::is_contained(collapsedDims, d) &&
          !llvm::is_contained(operandBatchingDims, d)) {
        operandOffsetDims.push_back(d);
      }
    }
    auto resultDimOfOperandDim = [&](int64_t original) -> int64_t {
      int64_t p = llvm::find(c.permutation, original) - c.permutation.begin();
      if (auto *it = llvm::find(c.batchDims, p); it != c.batchDims.end()) {
        return r.batch[it - c.batchDims.begin()];
      }
      if (auto *it = llvm::find(c.componentDims, p);
          it != c.componentDims.end()) {
        return r.window[it - c.componentDims.begin()];
      }
      auto *it = llvm::find(c.rowDims, p);
      return r.row[it - c.rowDims.begin()];
    };
    if (offsetDims.size() != operandOffsetDims.size()) {
      return rewriter.notifyMatchFailure(srcOp, "offset dims mismatch");
    }
    for (auto [position, original] : llvm::zip(offsetDims, operandOffsetDims)) {
      permutation[position] = resultDimOfOperandDim(original);
    }
    for (int64_t i = 0; i < extraDims; ++i) {
      permutation[srcOp.getType().getRank() + i] =
          resultDimOfOperandDim(originalRank + i);
    }
    SmallVector<int64_t> indexDims;
    for (int64_t d = 0; d < indicesType.getRank(); ++d) {
      if (d != indexVectorDim) {
        indexDims.push_back(d);
      }
    }
    size_t next = 0;
    for (int64_t position = 0; position < resultType.getRank(); ++position) {
      if (permutation[position] >= 0) {
        continue;
      }
      if (next >= indexDims.size()) {
        return rewriter.notifyMatchFailure(srcOp, "result rank mismatch");
      }
      int64_t d = indexDims[next++];
      if (auto *it = llvm::find(startIndicesBatchingDims, d);
          it != startIndicesBatchingDims.end()) {
        // Where the paired operand batching dim landed (the table's batch
        // dims may also hold size-1 free dims).
        permutation[position] = resultDimOfOperandDim(
            operandBatchingDims[it - startIndicesBatchingDims.begin()]);
      } else {
        auto *other = llvm::find(layout.otherDims, d);
        permutation[position] = r.other[other - layout.otherDims.begin()];
      }
    }
    if (llvm::is_contained(permutation, -1) ||
        static_cast<int64_t>(r.shape.size()) != resultType.getRank()) {
      return rewriter.notifyMatchFailure(srcOp, "result dims mismatch");
    }
    // PermuteOp: result dim i = input dim permutation[i].
    rewriter.replaceOp(srcOp, reshapeAndPermute(rewriter, loc, gathered,
                                                r.shape, permutation,
                                                resultType));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Scatter
//===----------------------------------------------------------------------===//

enum class UnaryApply { None, Negate, Sine, Cosine, Square, Sqrt, Log, Exp, SubtractOne };

struct UpdateComputation {
  ttcore::ReduceType reduce = ttcore::ReduceType::Invalid; // Invalid: overwrite
  bool negateUpdates = false;
  UnaryApply unary = UnaryApply::None;
};

template <typename BinaryOp>
bool matchesBinary(Block &block) {
  if (block.getNumArguments() != 2 || block.getOperations().size() != 2) {
    return false;
  }
  auto binary = mlir::dyn_cast<BinaryOp>(block.front());
  auto returnOp =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  return binary && returnOp && returnOp.getNumOperands() == 1 &&
         binary.getLhs() == block.getArgument(0) &&
         binary.getRhs() == block.getArgument(1) &&
         returnOp.getOperand(0) == binary.getResult();
}

template <typename UnaryOp>
bool matchesUnary(Block &block) {
  if (block.getNumArguments() != 2 || block.getOperations().size() != 2) {
    return false;
  }
  auto unary = mlir::dyn_cast<UnaryOp>(block.front());
  auto returnOp =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  return unary && returnOp && returnOp.getNumOperands() == 1 &&
         unary.getOperand() == block.getArgument(0) &&
         returnOp.getOperand(0) == unary.getResult();
}

std::optional<UpdateComputation> classifyUpdateComputation(Region &region) {
  if (!region.hasOneBlock()) {
    return std::nullopt;
  }
  Block &block = region.front();
  if (block.getNumArguments() != 2) {
    return std::nullopt;
  }
  auto returnOp =
      mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1) {
    return std::nullopt;
  }
  UpdateComputation computation;
  if (block.getOperations().size() == 1) {
    // Overwrite: (operand, update) -> update. Returning the operand keeps it.
    if (returnOp.getOperand(0) == block.getArgument(1)) {
      return computation;
    }
    return std::nullopt;
  }
  if (matchesBinary<mlir::stablehlo::AddOp>(block)) {
    computation.reduce = ttcore::ReduceType::Sum;
  } else if (matchesBinary<mlir::stablehlo::MulOp>(block)) {
    computation.reduce = ttcore::ReduceType::Prod;
  } else if (matchesBinary<mlir::stablehlo::MaxOp>(block)) {
    computation.reduce = ttcore::ReduceType::Max;
  } else if (matchesBinary<mlir::stablehlo::MinOp>(block)) {
    computation.reduce = ttcore::ReduceType::Min;
  } else if (matchesBinary<mlir::stablehlo::SubtractOp>(block)) {
    computation.reduce = ttcore::ReduceType::Sum;
    computation.negateUpdates = true;
  } else if (matchesUnary<mlir::stablehlo::NegOp>(block)) {
    computation.unary = UnaryApply::Negate;
  } else if (matchesUnary<mlir::stablehlo::SineOp>(block)) {
    computation.unary = UnaryApply::Sine;
  } else if (matchesUnary<mlir::stablehlo::CosineOp>(block)) {
    computation.unary = UnaryApply::Cosine;
  } else if (matchesUnary<mlir::stablehlo::SqrtOp>(block)) {
    computation.unary = UnaryApply::Sqrt;
  } else if (matchesUnary<mlir::stablehlo::LogOp>(block)) {
    computation.unary = UnaryApply::Log;
  } else if (matchesUnary<mlir::stablehlo::ExpOp>(block)) {
    computation.unary = UnaryApply::Exp;
  } else {
    Operation *result = returnOp.getOperand(0).getDefiningOp();
    auto multiply = mlir::dyn_cast_or_null<mlir::stablehlo::MulOp>(result);
    if (multiply && block.getOperations().size() == 2 &&
        multiply.getLhs() == block.getArgument(0) &&
        multiply.getRhs() == block.getArgument(0)) {
      computation.unary = UnaryApply::Square;
      return computation;
    }
    auto subtract = mlir::dyn_cast_or_null<mlir::stablehlo::SubtractOp>(result);
    if (!subtract || subtract.getLhs() != block.getArgument(0)) {
      return std::nullopt;
    }
    auto constant =
        subtract.getRhs().getDefiningOp<mlir::stablehlo::ConstantOp>();
    bool constantInBlock = constant && constant->getBlock() == &block;
    if (block.getOperations().size() != (constantInBlock ? 3u : 2u)) {
      return std::nullopt;
    }
    auto value = constant ? mlir::dyn_cast<DenseIntElementsAttr>(
                                constant.getValue())
                          : DenseIntElementsAttr();
    if (!value || !value.isSplat() || !value.getSplatValue<APInt>().isOne()) {
      return std::nullopt;
    }
    computation.unary = UnaryApply::SubtractOne;
  }
  return computation;
}

Value applyUnary(ConversionPatternRewriter &rewriter, Location loc,
                 UnaryApply kind, Value value) {
  auto type = mlir::cast<RankedTensorType>(value.getType());
  switch (kind) {
  case UnaryApply::Negate:
    return rewriter.create<ttir::NegOp>(loc, type, value);
  case UnaryApply::Sine:
    return rewriter.create<ttir::SinOp>(loc, type, value);
  case UnaryApply::Cosine:
    return rewriter.create<ttir::CosOp>(loc, type, value);
  case UnaryApply::Square:
    return rewriter.create<ttir::MultiplyOp>(loc, type, value, value);
  case UnaryApply::Sqrt:
    return rewriter.create<ttir::SqrtOp>(loc, type, value);
  case UnaryApply::Log:
    return rewriter.create<ttir::LogOp>(loc, type, value);
  case UnaryApply::Exp:
    return rewriter.create<ttir::ExpOp>(loc, type, value);
  case UnaryApply::SubtractOne: {
    Attribute one = rewriter.getOneAttr(type.getElementType());
    Value ones = rewriter.create<ttir::ConstantOp>(
        loc, type, DenseElementsAttr::get(type, one));
    return rewriter.create<ttir::SubtractOp>(loc, type, value, ones);
  }
  case UnaryApply::None:
    break;
  }
  llvm_unreachable("not a unary apply");
}

// Restores the operand's shape from a [B, N, R] table result.
Value untable(ConversionPatternRewriter &rewriter, Location loc, Value scattered,
              const Canonical &c, RankedTensorType operandType) {
  SmallVector<int64_t> permutedShape =
      ttmlir::utils::applyPermutation(operandType.getShape(), c.permutation);
  Value result = ttir::utils::createReshapeOp(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_untable"),
      scattered, permutedShape);
  if (!isIdentity(c.permutation)) {
    result = rewriter.create<ttir::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_unpermute"), operandType,
        result, ttmlir::utils::inversePermutation(c.permutation));
  }
  return result;
}

// Scatters `source` ([B, M * W, R]) into the table and restores the operand's
// shape.
Value scatterRows(ConversionPatternRewriter &rewriter, Location loc,
                  Value table, Value indices, Value source, const Canonical &c,
                  ttcore::ReduceType reduce, RankedTensorType operandType) {
  Value scattered = rewriter.create<ttir::RowsScatterOp>(
      loc, table.getType(), table, indices, source, i64Array(rewriter, c.sizes),
      i64Array(rewriter, c.windows), rewriter.getBoolAttr(c.column),
      ttcore::ReduceTypeAttr::get(rewriter.getContext(), reduce));
  return untable(rewriter, loc, scattered, c, operandType);
}

// StableHLO applies a scatter's update computation once per update, so a
// position that appears k times has its unary function applied k times. Up to
// this many positions are applied one at a time to honor that; beyond it the
// function is applied once per row.
constexpr int64_t kMaxUnrolledApply = 1024;

class ScatterConversionPattern
    : public OpConversionPattern<mlir::stablehlo::ScatterOp> {
  using OpConversionPattern<mlir::stablehlo::ScatterOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::ScatterOp srcOp,
                  mlir::stablehlo::ScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = srcOp.getLoc();
    if (srcOp.getInputs().size() != 1) {
      return rewriter.notifyMatchFailure(
          srcOp, "only one scattered tensor is supported");
    }
    auto dimensionNumbers = srcOp.getScatterDimensionNumbers();
    ArrayRef<int64_t> updateWindowDims = dimensionNumbers.getUpdateWindowDims();
    ArrayRef<int64_t> insertedWindowDims =
        dimensionNumbers.getInsertedWindowDims();
    ArrayRef<int64_t> inputBatchingDims = dimensionNumbers.getInputBatchingDims();
    ArrayRef<int64_t> scatterIndicesBatchingDims =
        dimensionNumbers.getScatterIndicesBatchingDims();
    ArrayRef<int64_t> scatterDimsToOperandDims =
        dimensionNumbers.getScatterDimsToOperandDims();
    int64_t indexVectorDim = dimensionNumbers.getIndexVectorDim();

    std::optional<UpdateComputation> computation =
        classifyUpdateComputation(srcOp.getUpdateComputation());
    if (!computation) {
      return rewriter.notifyMatchFailure(
          srcOp, "the update computation is not an overwrite, a sum, product, "
                 "maximum, minimum, subtraction or a supported unary function");
    }

    Value operand = adaptor.getInputs().front();
    Value updates = adaptor.getUpdates().front();
    auto operandType = mlir::cast<RankedTensorType>(operand.getType());
    auto updatesType = mlir::cast<RankedTensorType>(updates.getType());
    auto indicesType =
        mlir::cast<RankedTensorType>(adaptor.getScatterIndices().getType());
    ArrayRef<int64_t> operandShape = operandType.getShape();
    ArrayRef<int64_t> updatesShape = updatesType.getShape();
    if (!mlir::isa<IntegerType>(indicesType.getElementType())) {
      return rewriter.notifyMatchFailure(srcOp, "indices must be integers");
    }
    // Nothing to write.
    if (updatesType.getNumElements() == 0 || operandType.getNumElements() == 0) {
      rewriter.replaceOp(srcOp, operand);
      return success();
    }

    // A complex operand and its updates arrive as floats with a trailing
    // dimension of 2, an extra free row dimension covered whole.
    const int64_t originalRank =
        mlir::cast<RankedTensorType>(srcOp.getInputs().front().getType())
            .getRank();
    const int64_t originalUpdatesRank =
        mlir::cast<RankedTensorType>(srcOp.getUpdates().front().getType())
            .getRank();
    const int64_t extraDims = operandType.getRank() - originalRank;
    if (extraDims < 0 ||
        updatesType.getRank() != originalUpdatesRank + extraDims) {
      return rewriter.notifyMatchFailure(srcOp, "unexpected converted ranks");
    }

    // Update window dims pair, in order, with the operand dims that are
    // neither inserted nor batching, in increasing order.
    SmallVector<int64_t> windowOperandDims;
    for (int64_t d = 0; d < originalRank; ++d) {
      if (!llvm::is_contained(insertedWindowDims, d) &&
          !llvm::is_contained(inputBatchingDims, d)) {
        windowOperandDims.push_back(d);
      }
    }
    if (windowOperandDims.size() != updateWindowDims.size()) {
      return rewriter.notifyMatchFailure(srcOp, "update window dims mismatch");
    }
    SmallVector<int64_t> updateDimOfOperandDim(operandShape.size(), -1);
    for (auto [u, d] : llvm::zip(updateWindowDims, windowOperandDims)) {
      updateDimOfOperandDim[d] = u;
    }
    for (int64_t i = 0; i < extraDims; ++i) {
      updateDimOfOperandDim[originalRank + i] = originalUpdatesRank + i;
    }

    SmallVector<OperandDim> dims(operandShape.size());
    for (auto [d, size] : llvm::enumerate(operandShape)) {
      OperandDim &dim = dims[d];
      dim.size = size;
      dim.window =
          updateDimOfOperandDim[d] >= 0 ? updatesShape[updateDimOfOperandDim[d]] : 1;
      if (static_cast<int64_t>(d) >= originalRank) {
        dim.kind = DimKind::Free;
        continue;
      }
      if (llvm::is_contained(inputBatchingDims, d)) {
        dim.kind = DimKind::Batch;
      } else if (auto *it = llvm::find(scatterDimsToOperandDims, d);
                 it != scatterDimsToOperandDims.end()) {
        dim.kind = DimKind::Indexed;
        dim.column = it - scatterDimsToOperandDims.begin();
      } else {
        dim.kind = DimKind::Free;
      }
    }
    Canonical c = canonicalize(dims);
    SmallVector<int64_t> indexBatchingDims;
    for (int64_t d = 0; d < originalRank; ++d) {
      if (auto *it = llvm::find(inputBatchingDims, d);
          it != inputBatchingDims.end()) {
        indexBatchingDims.push_back(
            scatterIndicesBatchingDims[it - inputBatchingDims.begin()]);
      }
    }

    Value table = buildTable(rewriter, loc, operand, c);
    IndexLayout layout =
        buildIndices(rewriter, loc, adaptor.getScatterIndices(), indexVectorDim,
                     indexBatchingDims, c);
    if (layout.Bi != 1 && layout.Bi != c.B) {
      return rewriter.notifyMatchFailure(
          srcOp, "index batches do not match the table's");
    }
    auto sourceType = RankedTensorType::get(c.gatheredShape(layout.M),
                                            operandType.getElementType());

    if (computation->unary != UnaryApply::None) {
      auto sizes = i64Array(rewriter, c.sizes);
      auto windows = i64Array(rewriter, c.windows);
      auto overwrite = ttcore::ReduceTypeAttr::get(rewriter.getContext(),
                                                   ttcore::ReduceType::Invalid);
      auto columnAttr = rewriter.getBoolAttr(c.column);
      // Each position applies the function to what the previous ones left,
      // so duplicates apply it repeatedly: positions are unrolled, up to a
      // limit. With unique indices one gather, apply and scatter suffice.
      if (srcOp.getUniqueIndices()) {
        Value rows = rewriter.create<ttir::RowsGatherOp>(
            loc, sourceType, table, layout.indices, sizes, windows, columnAttr,
            rewriter.getBoolAttr(false));
        rows = applyUnary(rewriter, loc, computation->unary, rows);
        rewriter.replaceOp(
            srcOp, scatterRows(rewriter, loc, table, layout.indices, rows, c,
                               ttcore::ReduceType::Invalid, operandType));
        return success();
      }
      if (layout.M > kMaxUnrolledApply) {
        return srcOp.emitError()
               << "a scatter applying a function at more than "
               << kMaxUnrolledApply
               << " positions needs unique_indices; duplicates would apply "
                  "it once per position";
      }
      auto indicesType = mlir::cast<RankedTensorType>(layout.indices.getType());
      const int64_t K = indicesType.getDimSize(1);
      auto positionType = indicesType.clone({layout.Bi, K, 1});
      auto rowsType = sourceType.clone(c.gatheredShape(1));
      Value current = table;
      for (int64_t m = 0; m < layout.M; ++m) {
        Value position = rewriter.create<ttir::SliceStaticOp>(
            loc, positionType, layout.indices,
            rewriter.getI32ArrayAttr({0, 0, static_cast<int32_t>(m)}),
            rewriter.getI32ArrayAttr({static_cast<int32_t>(layout.Bi),
                                      static_cast<int32_t>(K),
                                      static_cast<int32_t>(m + 1)}),
            rewriter.getI32ArrayAttr({1, 1, 1}));
        Value rows = rewriter.create<ttir::RowsGatherOp>(
            loc, rowsType, current, position, sizes, windows, columnAttr,
            rewriter.getBoolAttr(false));
        rows = applyUnary(rewriter, loc, computation->unary, rows);
        current = rewriter.create<ttir::RowsScatterOp>(
            loc, current.getType(), current, position, rows, sizes, windows,
            columnAttr, overwrite);
      }
      rewriter.replaceOp(srcOp, untable(rewriter, loc, current, c, operandType));
      return success();
    }

    // The updates as [B, M * W, R]: table batch dims, index position dims,
    // component windows (inserted dims have none), row dims. Update scatter
    // dims (those not in update_window_dims) pair with the index dims other
    // than the index vector, in increasing order.
    SmallVector<int64_t> updateScatterDims;
    for (int64_t u = 0; u < originalUpdatesRank; ++u) {
      if (!llvm::is_contained(updateWindowDims, u)) {
        updateScatterDims.push_back(u);
      }
    }
    SmallVector<int64_t> indexDims;
    for (int64_t d = 0; d < indicesType.getRank(); ++d) {
      if (d != indexVectorDim) {
        indexDims.push_back(d);
      }
    }
    if (indexDims.size() != updateScatterDims.size()) {
      return rewriter.notifyMatchFailure(srcOp, "update scatter dims mismatch");
    }
    auto updateDimOfIndexDim = [&](int64_t d) {
      return updateScatterDims[llvm::find(indexDims, d) - indexDims.begin()];
    };
    SmallVector<int64_t> updatePermutation;
    for (int64_t p : c.batchDims) {
      const OperandDim &dim = c.dims[p];
      if (dim.kind == DimKind::Batch) {
        int64_t original = c.permutation[p];
        int64_t i = llvm::find(inputBatchingDims, original) -
                    inputBatchingDims.begin();
        updatePermutation.push_back(
            updateDimOfIndexDim(scatterIndicesBatchingDims[i]));
      } else {
        updatePermutation.push_back(updateDimOfOperandDim[c.permutation[p]]);
      }
    }
    for (int64_t d : layout.otherDims) {
      updatePermutation.push_back(updateDimOfIndexDim(d));
    }
    for (int64_t p : c.componentDims) {
      int64_t u = updateDimOfOperandDim[c.permutation[p]];
      if (u >= 0) {
        updatePermutation.push_back(u);
      }
    }
    for (int64_t p : c.rowDims) {
      updatePermutation.push_back(updateDimOfOperandDim[c.permutation[p]]);
    }
    if (static_cast<int64_t>(updatePermutation.size()) != updatesType.getRank() ||
        llvm::is_contained(updatePermutation, -1)) {
      return rewriter.notifyMatchFailure(srcOp, "update dims mismatch");
    }
    Value source = updates;
    if (!isIdentity(updatePermutation)) {
      source = rewriter.create<ttir::PermuteOp>(
          ttmlir::utils::appendLocationSuffix(loc, "_permute_updates"),
          updatesType.clone(
              ttmlir::utils::applyPermutation(updatesShape, updatePermutation)),
          source, updatePermutation);
    }
    source = ttir::utils::createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_source"), source,
        sourceType.getShape());
    if (computation->negateUpdates) {
      source = rewriter.create<ttir::NegOp>(loc, sourceType, source);
    }
    rewriter.replaceOp(srcOp,
                       scatterRows(rewriter, loc, table, layout.indices, source,
                                   c, computation->reduce, operandType));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Dynamic update slice: one position whose window is the update's shape.
//===----------------------------------------------------------------------===//

class DynamicUpdateSliceConversionPattern
    : public OpConversionPattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpConversionPattern<
      mlir::stablehlo::DynamicUpdateSliceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp srcOp,
                  mlir::stablehlo::DynamicUpdateSliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = srcOp.getLoc();
    Value operand = adaptor.getOperand();
    Value update = adaptor.getUpdate();
    auto operandType = mlir::cast<RankedTensorType>(operand.getType());
    auto updateType = mlir::cast<RankedTensorType>(update.getType());
    ArrayRef<int64_t> operandShape = operandType.getShape();
    ArrayRef<int64_t> updateShape = updateType.getShape();
    if (operandShape == updateShape) {
      rewriter.replaceOp(srcOp, update);
      return success();
    }
    if (updateType.getNumElements() == 0) {
      rewriter.replaceOp(srcOp, operand);
      return success();
    }

    // StableHLO gives every start index the same type.
    Type startElementType =
        getElementTypeOrSelf(adaptor.getStartIndices().front());
    auto columnType = RankedTensorType::get(
        {1, 1}, rowsIndexType(rewriter, startElementType));
    // Dimensions the update does not cover are indexed; the rest are free.
    SmallVector<OperandDim> dims(operandShape.size());
    SmallVector<Value> columns;
    for (auto [d, size] : llvm::enumerate(operandShape)) {
      OperandDim &dim = dims[d];
      dim.size = size;
      dim.window = updateShape[d];
      dim.kind = DimKind::Free;
      // A complex operand's trailing dimension of 2 has no start index.
      if (static_cast<int64_t>(d) >= adaptor.getStartIndices().size()) {
        continue;
      }
      if (updateShape[d] != size) {
        dim.kind = DimKind::Indexed;
        dim.column = columns.size();
        Value start = adaptor.getStartIndices()[d];
        auto startType = mlir::cast<RankedTensorType>(start.getType());
        if (!mlir::isa<IntegerType>(startType.getElementType())) {
          return rewriter.notifyMatchFailure(srcOp, "start indices must be integers");
        }
        start = ttir::utils::createReshapeOp(
            rewriter, ttmlir::utils::appendLocationSuffix(loc, "_start"), start,
            {1, 1});
        if (!startType.getElementType().isInteger(32)) {
          start = rewriter.create<ttir::TypecastOp>(loc, columnType, start);
        }
        // Unlike a scatter, dynamic_update_slice clamps its start indices so
        // the update always lands in the operand.
        start = rewriter.create<ttir::ClampScalarOp>(
            loc, columnType, start, rewriter.getI32IntegerAttr(0),
            rewriter.getI32IntegerAttr(size - updateShape[d]));
        columns.push_back(start);
      }
    }
    Canonical c = canonicalize(dims);
    // Without batching dims the operand keeps its order, so the components
    // are the indexed dims in operand order, interleaved with zero columns.
    SmallVector<Value> indexColumns;
    Value zeros;
    for (int64_t column : c.columns) {
      if (column >= 0) {
        indexColumns.push_back(columns[column]);
      } else {
        if (!zeros) {
          zeros = rewriter.create<ttir::FullOp>(loc, columnType,
                                                rewriter.getI32IntegerAttr(0));
        }
        indexColumns.push_back(zeros);
      }
    }
    Value indices = indexColumns.front();
    if (indexColumns.size() > 1) {
      indices = rewriter.create<ttir::ConcatOp>(
          ttmlir::utils::appendLocationSuffix(loc, "_starts"),
          columnType.clone({1, static_cast<int64_t>(indexColumns.size())}),
          indexColumns, rewriter.getSI32IntegerAttr(1));
    }
    // One position: [Bi, K, M] = [1, K, 1].
    indices = ttir::utils::createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_starts_pages"),
        indices, {1, static_cast<int64_t>(indexColumns.size()), 1});

    Value table = buildTable(rewriter, loc, operand, c);
    // The update's dims are the operand's: [batch][component windows][rows].
    auto sourceType = RankedTensorType::get(c.gatheredShape(1),
                                            operandType.getElementType());
    Value source = ttir::utils::createReshapeOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_source"), update,
        sourceType.getShape());
    rewriter.replaceOp(srcOp,
                       scatterRows(rewriter, loc, table, indices, source, c,
                                   ttcore::ReduceType::Invalid, operandType));
    return success();
  }
};

} // namespace

void populateIndexedRowsToTTIRPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<GatherConversionPattern, ScatterConversionPattern,
               DynamicUpdateSliceConversionPattern>(typeConverter, ctx);
}

} // namespace mlir::tt
