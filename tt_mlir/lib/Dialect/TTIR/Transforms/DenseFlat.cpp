// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// Flat buffers and unit rows in dense tiles.
//
// TTNN tiles a tensor's last two dims, so a rank-1 tensor of N elements is one
// row of N / 32 tiles, each padded to 32 rows: 32 times its size in DRAM, and a
// matrix of W < 32 rows takes 32 / W times its size. PyTorch's FSDP2 gathers
// parameters into such flat buffers and reduces gradients from [world size, N]
// ones, so on Tenstorrent a model's parameters and gradients took many times
// their size in transit (a TPU picks a layout without the padding itself).
// Viewed as [N / 32, 32], the same elements in row-major order fill whole
// tiles, for both shapes. Likewise a [T, 1, V] tensor, Megatron's [sequence,
// batch, vocabulary] logits with a batch of one, pads each of its T rows to 32:
// viewed as [T, V], its rows share tiles.
//
// The runtime stores such tensors that way (`toLayout` to tiles on device;
// compactShape in tt_mlir_dense_flat.patch has the same rules as storedShape
// here), and this pass gives the public functions' arguments and results of
// those shapes the compact type, with a reshape at the boundary. It then moves
// those reshapes through the ops that keep the elements in order, in either
// direction. For flat buffers: elementwise ops, slices and concatenations on
// 32-element boundaries, all-gathers and reduce-scatters of rank-1 tensors,
// all-reduces and constants; for unit rows: elementwise ops, broadcasts and
// reductions that keep the unit dim. Where two reshapes meet they fold:
// FSDP2's casts, copies and collectives then run on [N / 32, 32] throughout, a
// parameter's view in its own shape is one reshape away, and a cross entropy
// over [T, 1, V] logits runs on [T, V]. An op none of this applies to sees the
// original shape, as before.
//
// LIBTT_DENSE_FLAT=0 turns it off, here and in the runtime.

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace mlir::tt::ttir {

namespace {

constexpr int64_t kWidth = 32;
// The runtime's rule (denseFlatShape): rank-1 arrays of at least 2^20
// elements, and matrices of at most 8 rows and 2^22 elements, a multiple of
// 32. A vocabulary's logits, even for 8 sequences, keep their shape.
constexpr int64_t kMinRank1Elements = int64_t{1} << 20;
constexpr int64_t kMaxRows = 8;
constexpr int64_t kMinMatrixElements = int64_t{1} << 22;

bool enabled() {
  const char *value = std::getenv("LIBTT_DENSE_FLAT");
  return value == nullptr || std::strcmp(value, "0") != 0;
}

RankedTensorType typeOf(Value value) {
  return mlir::cast<RankedTensorType>(value.getType());
}

RankedTensorType withShape(RankedTensorType type, ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, type.getElementType(),
                               type.getEncoding());
}

// Whether a tensor has a dense view [n / 32, 32] of its n elements in order: a
// rank-1 tensor, or a matrix of fewer than 32 rows (other than one of 32
// columns, which is its own view), of a multiple of 32 elements.
bool viewable(RankedTensorType type) {
  if (!type.hasStaticShape() || type.getNumElements() == 0 ||
      type.getNumElements() % kWidth != 0) {
    return false;
  }
  return type.getRank() == 1 ||
         (type.getRank() == 2 && type.getDimSize(0) < kWidth &&
          type.getDimSize(1) != kWidth);
}

// Whether a function boundary of this type takes the view: the runtime stores
// these tensors that way.
bool storedDense(RankedTensorType type) {
  if (!viewable(type)) {
    return false;
  }
  if (type.getRank() == 1) {
    return type.getNumElements() >= kMinRank1Elements;
  }
  return type.getDimSize(0) <= kMaxRows &&
         type.getNumElements() >= kMinMatrixElements;
}

// A rank >= 3 tensor whose second-to-last dim is 1, and its shape without it.
bool unitRow(RankedTensorType type) {
  return type.getRank() >= 3 && type.hasStaticShape() &&
         type.getDimSize(type.getRank() - 2) == 1;
}

SmallVector<int64_t> withoutUnitRow(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> result(shape);
  result.erase(result.end() - 2);
  return result;
}

// Whether a boundary of this type is stored as [d0, d2]: the runtime's rule for
// unit rows, rank 3 only (a key-value cache of one head keeps its layout).
bool storedWithoutUnitRow(RankedTensorType type) {
  return unitRow(type) && type.getRank() == 3 &&
         type.getNumElements() >= kMinRank1Elements;
}

SmallVector<int64_t> denseShape(RankedTensorType type) {
  return {type.getNumElements() / kWidth, kWidth};
}

RankedTensorType denseType(RankedTensorType type) {
  return withShape(type, denseShape(type));
}

// `x` when `value` is a reshape of a dense view x to a viewable shape.
Value denseSource(Value value) {
  auto reshape = value.getDefiningOp<ReshapeOp>();
  if (!reshape || !viewable(typeOf(reshape.getResult()))) {
    return nullptr;
  }
  RankedTensorType inputType = typeOf(reshape.getInput());
  if (inputType.getRank() != 2 || inputType.getDimSize(1) != kWidth) {
    return nullptr;
  }
  return reshape.getInput();
}

Value reshapeTo(PatternRewriter &rewriter, Location loc, Value value,
                ArrayRef<int64_t> shape) {
  return utils::createReshapeOp(rewriter, loc, value, shape);
}

// Whether an operand's dense view costs nothing to build: a constant, a fill
// or a broadcast of one element.
bool freeToView(Value operand) {
  if (!viewable(typeOf(operand))) {
    return false;
  }
  Operation *producer = operand.getDefiningOp();
  if (auto constant = mlir::dyn_cast_or_null<ConstantOp>(producer)) {
    return mlir::isa<DenseElementsAttr>(constant.getValue());
  }
  if (auto broadcast = mlir::dyn_cast_or_null<BroadcastOp>(producer)) {
    return typeOf(broadcast.getInput()).getNumElements() == 1;
  }
  return mlir::isa_and_nonnull<FullOp>(producer);
}

// The dense view of an operand that is free to view (freeToView).
Value freeDenseView(PatternRewriter &rewriter, Location loc, Value operand) {
  RankedTensorType dense = denseType(typeOf(operand));
  Operation *producer = operand.getDefiningOp();
  if (auto constant = mlir::dyn_cast<ConstantOp>(producer)) {
    return rewriter.create<ConstantOp>(
        loc, dense,
        mlir::cast<DenseElementsAttr>(constant.getValue()).reshape(dense));
  }
  if (auto full = mlir::dyn_cast<FullOp>(producer)) {
    return rewriter.create<FullOp>(
        loc, dense,
        rewriter.getDenseI32ArrayAttr(
            {static_cast<int32_t>(dense.getDimSize(0)), int32_t{kWidth}}),
        full.getFillValue());
  }
  auto broadcast = mlir::cast<BroadcastOp>(producer);
  Value scalar = reshapeTo(rewriter, loc, broadcast.getInput(), {1, 1});
  return rewriter.create<BroadcastOp>(
      loc, dense, scalar, rewriter.getDenseI64ArrayAttr(dense.getShape()));
}

int64_t intAt(ArrayAttr array, unsigned index) {
  return mlir::cast<IntegerAttr>(array[index]).getInt();
}

// For a static slice, its begins and sizes per dim, if every step is 1.
std::optional<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
sliceBounds(SliceStaticOp slice) {
  SmallVector<int64_t> begins, sizes;
  for (unsigned i = 0; i < slice.getBegins().size(); ++i) {
    if (intAt(slice.getStep(), i) != 1) {
      return std::nullopt;
    }
    begins.push_back(intAt(slice.getBegins(), i));
    sizes.push_back(intAt(slice.getEnds(), i) - begins.back());
  }
  return std::make_pair(begins, sizes);
}

// Whether `op` keeps the elements of its viewable operands and result in
// order, so that it has an op on their dense views: `rebuild` makes that op.
bool keepsOrder(Operation *op) {
  if (op->getNumResults() != 1) {
    return false;
  }
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || !viewable(resultType)) {
    return false;
  }
  if (mlir::isa<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>(op)) {
    // Every operand is the result's shape, or one element broadcast to it.
    return llvm::all_of(op->getOperands(), [&](Value operand) {
      RankedTensorType type = typeOf(operand);
      return type.getNumElements() == 1 ||
             type.getShape() == resultType.getShape();
    });
  }
  if (auto slice = mlir::dyn_cast<SliceStaticOp>(op)) {
    RankedTensorType inputType = typeOf(slice.getInput());
    auto bounds = sliceBounds(slice);
    if (!bounds || !viewable(inputType)) {
      return false;
    }
    auto &[begins, sizes] = *bounds;
    if (inputType.getRank() == 1) {
      return begins[0] % kWidth == 0;
    }
    // Whole rows, or columns on 32-element boundaries of every row.
    int64_t columns = inputType.getDimSize(1);
    if (columns % kWidth != 0) {
      return false;
    }
    return sizes[1] == columns ||
           (sizes[0] == inputType.getDimSize(0) && begins[1] % kWidth == 0 &&
            sizes[1] % kWidth == 0);
  }
  if (auto concat = mlir::dyn_cast<ConcatOp>(op)) {
    return llvm::all_of(concat.getInputs(), [&](Value input) {
      RankedTensorType type = typeOf(input);
      if (!viewable(type) || type.getRank() != resultType.getRank()) {
        return false;
      }
      // Rank 1, or matrices whose rows are whole rows of the view: stacked
      // along dim 0, or put side by side along dim 1 (as FSDP2 lays out each
      // rank's shards of the gradients it reduces).
      return type.getRank() == 1 || type.getDimSize(1) % kWidth == 0;
    });
  }
  if (auto gather = mlir::dyn_cast<AllGatherOp>(op)) {
    return resultType.getRank() == 1 && viewable(typeOf(gather.getInput()));
  }
  if (auto scatter = mlir::dyn_cast<ReduceScatterOp>(op)) {
    return resultType.getRank() == 1 && viewable(typeOf(scatter.getInput()));
  }
  if (auto reduce = mlir::dyn_cast<AllReduceOp>(op)) {
    return viewable(typeOf(reduce.getInput()));
  }
  return false;
}

// Rows [begin, end) of a dense view.
Value denseRows(PatternRewriter &rewriter, Location loc, Value dense,
                int64_t begin, int64_t end) {
  RankedTensorType type = typeOf(dense);
  if (begin == 0 && end == type.getDimSize(0)) {
    return dense;
  }
  return rewriter.create<SliceStaticOp>(
      loc, withShape(type, {end - begin, kWidth}), dense,
      rewriter.getI32ArrayAttr(
          {static_cast<int32_t>(begin), 0}),
      rewriter.getI32ArrayAttr(
          {static_cast<int32_t>(end), int32_t{kWidth}}),
      rewriter.getI32ArrayAttr({1, 1}));
}

Value concatRows(PatternRewriter &rewriter, Location loc,
                 RankedTensorType type, ArrayRef<Value> pieces) {
  if (pieces.size() == 1) {
    return pieces.front();
  }
  return rewriter.create<ConcatOp>(loc, type, pieces,
                                   rewriter.getSI32IntegerAttr(0));
}

// `op` on `operands`, its viewable operands' dense views (one-element operands
// as [1, 1]).
Value rebuild(PatternRewriter &rewriter, Operation *op, ValueRange operands) {
  Location loc = op->getLoc();
  RankedTensorType resultType = typeOf(op->getResult(0));
  RankedTensorType dense = denseType(resultType);
  if (auto slice = mlir::dyn_cast<SliceStaticOp>(op)) {
    auto [begins, sizes] = *sliceBounds(slice);
    RankedTensorType inputType = typeOf(slice.getInput());
    if (inputType.getRank() == 1) {
      return denseRows(rewriter, loc, operands[0], begins[0] / kWidth,
                       (begins[0] + sizes[0]) / kWidth);
    }
    // A row is columns / 32 rows of the view.
    int64_t rowBlocks = inputType.getDimSize(1) / kWidth;
    if (sizes[1] == inputType.getDimSize(1)) {
      return denseRows(rewriter, loc, operands[0], begins[0] * rowBlocks,
                       (begins[0] + sizes[0]) * rowBlocks);
    }
    SmallVector<Value> pieces;
    for (int64_t row = 0; row < sizes[0]; ++row) {
      int64_t first = (begins[0] + row) * rowBlocks + begins[1] / kWidth;
      pieces.push_back(denseRows(rewriter, loc, operands[0], first,
                                 first + sizes[1] / kWidth));
    }
    return concatRows(rewriter, loc, dense, pieces);
  }
  if (auto concat = mlir::dyn_cast<ConcatOp>(op);
      concat && resultType.getRank() == 2 && concat.getDim() == 1) {
    // Row by row, each input's blocks of that row.
    SmallVector<Value> pieces;
    for (int64_t row = 0; row < resultType.getDimSize(0); ++row) {
      for (auto [input, view] : llvm::zip_equal(concat.getInputs(), operands)) {
        int64_t rowBlocks = typeOf(input).getDimSize(1) / kWidth;
        pieces.push_back(denseRows(rewriter, loc, view, row * rowBlocks,
                                   (row + 1) * rowBlocks));
      }
    }
    return concatRows(rewriter, loc, dense, pieces);
  }
  // The other ops keep their attributes: their dim, if any, is 0 in both
  // (rank-1 collectives and concatenations, and matrices stacked by rows).
  OperationState state(loc, op->getName().getStringRef(), operands,
                       TypeRange{dense}, op->getAttrs());
  return rewriter.create(state)->getResult(0);
}

// op(reshape(x), ...) -> reshape(op(x, ...)): an op whose viewable operands
// are all dense views reshaped back, or free to view, runs on the views.
class SinkDenseView : public RewritePattern {
public:
  SinkDenseView(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!keepsOrder(op)) {
      return failure();
    }
    bool anyView = false;
    for (Value operand : op->getOperands()) {
      if (typeOf(operand).getNumElements() == 1) {
        continue;
      }
      if (denseSource(operand)) {
        anyView = true;
        continue;
      }
      if (!freeToView(operand)) {
        return failure();
      }
    }
    if (!anyView) {
      return failure();
    }
    rewriter.setInsertionPoint(op);
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (typeOf(operand).getNumElements() == 1) {
        operands.push_back(reshapeTo(rewriter, op->getLoc(), operand, {1, 1}));
      } else if (Value source = denseSource(operand)) {
        operands.push_back(source);
      } else {
        operands.push_back(freeDenseView(rewriter, op->getLoc(), operand));
      }
    }
    Value dense = rebuild(rewriter, op, operands);
    rewriter.replaceOp(op, reshapeTo(rewriter, op->getLoc(), dense,
                                     typeOf(op->getResult(0)).getShape()));
    return success();
  }
};

// reshape(op(a, ...)) to the dense view -> op(reshape(a), ...): the view of an
// op's result is the op on its operands' views. The op's other users take it
// reshaped back, so it runs once, and those that keep the order take the view
// in turn (SinkDenseView).
class HoistDenseView : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    Value input = reshape.getInput();
    RankedTensorType inputType = typeOf(input);
    if (!viewable(inputType) || typeOf(reshape.getResult()).getShape() !=
                                    ArrayRef<int64_t>(denseShape(inputType))) {
      return failure();
    }
    Operation *producer = input.getDefiningOp();
    if (!producer || (!freeToView(input) && !keepsOrder(producer))) {
      return failure();
    }
    rewriter.setInsertionPoint(producer);
    Value dense;
    if (freeToView(input)) {
      dense = freeDenseView(rewriter, producer->getLoc(), input);
    } else {
      SmallVector<Value> operands;
      for (Value operand : producer->getOperands()) {
        RankedTensorType type = typeOf(operand);
        operands.push_back(reshapeTo(rewriter, producer->getLoc(), operand,
                                     type.getNumElements() == 1
                                         ? SmallVector<int64_t>{1, 1}
                                         : denseShape(type)));
      }
      dense = rebuild(rewriter, producer, operands);
    }
    if (!input.hasOneUse()) {
      rewriter.replaceAllUsesExcept(
          input,
          reshapeTo(rewriter, producer->getLoc(), dense,
                    inputType.getShape()),
          reshape);
    }
    rewriter.replaceOp(reshape, dense);
    if (producer->use_empty()) {
      rewriter.eraseOp(producer);
    }
    return success();
  }
};

// reshape(reshape(x)) -> reshape(x), or x when the shapes match.
class FoldReshapes : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    auto inner = reshape.getInput().getDefiningOp<ReshapeOp>();
    if (!inner) {
      return failure();
    }
    Value source = inner.getInput();
    if (typeOf(source).getShape() == typeOf(reshape.getResult()).getShape()) {
      rewriter.replaceOp(reshape, source);
      return success();
    }
    rewriter.replaceOp(reshape,
                       reshapeTo(rewriter, reshape.getLoc(), source,
                                 typeOf(reshape.getResult()).getShape()));
    return success();
  }
};

// `x` when `value` is a reshape of x, its shape without the unit row.
Value unitRowSource(Value value) {
  auto reshape = value.getDefiningOp<ReshapeOp>();
  if (!reshape || !unitRow(typeOf(reshape.getResult()))) {
    return nullptr;
  }
  if (typeOf(reshape.getInput()).getShape() !=
      ArrayRef<int64_t>(withoutUnitRow(typeOf(reshape.getResult()).getShape()))) {
    return nullptr;
  }
  return reshape.getInput();
}

bool isReduction(Operation *op) {
  return mlir::isa<SumOp, MeanOp, MaxOp, MinOp, ProdOp, ArgMaxOp, ReduceAndOp,
                   ReduceOrOp>(op);
}

// The dims a reduction over a unit-row input reduces, if they leave the unit
// row alone.
std::optional<SmallVector<int64_t>> reducedDims(Operation *op) {
  int64_t rank = typeOf(op->getOperand(0)).getRank();
  SmallVector<int64_t> dims;
  auto dimArg = op->getAttrOfType<ArrayAttr>("dim_arg");
  if (!dimArg) {
    return std::nullopt;
  }
  for (Attribute dim : dimArg) {
    int64_t d = mlir::cast<IntegerAttr>(dim).getInt();
    d = d < 0 ? d + rank : d;
    if (d == rank - 2) {
      return std::nullopt;
    }
    dims.push_back(d);
  }
  return dims;
}

// Whether `op` has a unit-row form: an elementwise op or broadcast of unit-row
// operands and result, or a reduction of a unit-row input along other dims.
bool keepsUnitRow(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumOperands() == 0) {
    return false;
  }
  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }
  if (isReduction(op)) {
    return unitRow(typeOf(op->getOperand(0))) && reducedDims(op).has_value();
  }
  if (!unitRow(resultType)) {
    return false;
  }
  if (mlir::isa<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary,
                BroadcastOp>(op)) {
    return llvm::all_of(op->getOperands(), [&](Value operand) {
      RankedTensorType type = typeOf(operand);
      return type.getRank() == resultType.getRank() && unitRow(type);
    });
  }
  return false;
}

// `op` on its operands without the unit row, reshaped back to its result.
Value rebuildWithoutUnitRow(PatternRewriter &rewriter, Operation *op,
                            ValueRange operands) {
  Location loc = op->getLoc();
  RankedTensorType resultType = typeOf(op->getResult(0));
  SmallVector<NamedAttribute> attrs(op->getAttrs());
  RankedTensorType viewType;
  if (isReduction(op)) {
    int64_t unit = typeOf(op->getOperand(0)).getRank() - 2;
    SmallVector<int32_t> dims;
    for (int64_t d : *reducedDims(op)) {
      dims.push_back(static_cast<int32_t>(d < unit ? d : d - 1));
    }
    for (NamedAttribute &attr : attrs) {
      if (attr.getName() == "dim_arg") {
        attr.setValue(rewriter.getI32ArrayAttr(dims));
      }
    }
    // The input's shape without the unit row and the reduced dims, or with
    // them as 1.
    bool keepDim = op->getAttrOfType<BoolAttr>("keep_dim").getValue();
    SmallVector<int64_t> shape;
    ArrayRef<int64_t> input = typeOf(operands[0]).getShape();
    for (int64_t d = 0; d < static_cast<int64_t>(input.size()); ++d) {
      bool reduced = llvm::is_contained(dims, static_cast<int32_t>(d));
      if (!reduced) {
        shape.push_back(input[d]);
      } else if (keepDim) {
        shape.push_back(1);
      }
    }
    viewType = withShape(resultType, shape);
  } else {
    viewType = withShape(resultType, withoutUnitRow(resultType.getShape()));
    if (auto broadcast = mlir::dyn_cast<BroadcastOp>(op)) {
      for (NamedAttribute &attr : attrs) {
        if (attr.getName() == "broadcast_dimensions") {
          attr.setValue(rewriter.getDenseI64ArrayAttr(
              withoutUnitRow(broadcast.getBroadcastDimensions())));
        }
      }
    }
  }
  OperationState state(loc, op->getName().getStringRef(), operands,
                       TypeRange{viewType}, attrs);
  return rewriter.create(state)->getResult(0);
}

// The operands of a unit-row op without their unit row: reshapes of x back,
// or new reshapes.
SmallVector<Value> unitRowOperands(PatternRewriter &rewriter, Operation *op) {
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    if (Value source = unitRowSource(operand)) {
      operands.push_back(source);
    } else {
      operands.push_back(reshapeTo(rewriter, op->getLoc(), operand,
                                   withoutUnitRow(typeOf(operand).getShape())));
    }
  }
  return operands;
}

// op(reshape(x), ...) -> reshape(op(x, ...)) for unit rows: an op of
// unit-row operands, at least one of them a reshape of x without it, runs
// without it.
class SinkUnitRowView : public RewritePattern {
public:
  SinkUnitRowView(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!keepsUnitRow(op) || llvm::none_of(op->getOperands(), [](Value v) {
          return unitRowSource(v) != nullptr;
        })) {
      return failure();
    }
    rewriter.setInsertionPoint(op);
    Value view = rebuildWithoutUnitRow(rewriter, op, unitRowOperands(rewriter, op));
    rewriter.replaceOp(op, reshapeTo(rewriter, op->getLoc(), view,
                                     typeOf(op->getResult(0)).getShape()));
    return success();
  }
};

// reshape(op(a, ...)) without the unit row -> op(reshape(a), ...): as
// HoistDenseView, for unit rows.
class HoistUnitRowView : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    Value input = reshape.getInput();
    RankedTensorType inputType = typeOf(input);
    if (!unitRow(inputType) ||
        typeOf(reshape.getResult()).getShape() !=
            ArrayRef<int64_t>(withoutUnitRow(inputType.getShape()))) {
      return failure();
    }
    Operation *producer = input.getDefiningOp();
    // A reduction's result has the unit row only when it keeps its dims.
    if (!producer || !keepsUnitRow(producer) ||
        (isReduction(producer) &&
         !producer->getAttrOfType<BoolAttr>("keep_dim").getValue())) {
      return failure();
    }
    rewriter.setInsertionPoint(producer);
    Value view = rebuildWithoutUnitRow(rewriter, producer,
                                       unitRowOperands(rewriter, producer));
    if (!input.hasOneUse()) {
      rewriter.replaceAllUsesExcept(
          input,
          reshapeTo(rewriter, producer->getLoc(), view, inputType.getShape()),
          reshape);
    }
    rewriter.replaceOp(reshape, view);
    if (producer->use_empty()) {
      rewriter.eraseOp(producer);
    }
    return success();
  }
};

// Sets a boundary's local shape, if its attributes give one.
void setLocalShape(DictionaryAttr attrs, RankedTensorType dense,
                   llvm::function_ref<void(StringRef, Attribute)> set) {
  if (!attrs || !attrs.get(ttcore::LocalShapeAttr::name)) {
    return;
  }
  set(ttcore::LocalShapeAttr::name,
      ttcore::LocalShapeAttr::get(dense.getContext(), dense));
}

// Whether the boundary's tensor is the device's own (unsharded, or presharded
// with its local shape): a mesh_shard would cut it otherwise.
bool wholeOnDevice(DictionaryAttr attrs, RankedTensorType type) {
  if (!attrs) {
    return true;
  }
  if (auto local = mlir::dyn_cast_or_null<ttcore::LocalShapeAttr>(
          attrs.get(ttcore::LocalShapeAttr::name))) {
    return local.getLocalShape().getShape() == type.getShape();
  }
  return true;
}

// The shape the runtime stores a boundary of this type in, if not its own.
std::optional<SmallVector<int64_t>> storedShape(RankedTensorType type) {
  if (storedDense(type)) {
    return denseShape(type);
  }
  if (storedWithoutUnitRow(type)) {
    return withoutUnitRow(type.getShape());
  }
  return std::nullopt;
}

// Which compact forms a function's boundaries took.
struct Viewed {
  bool dense = false;
  bool unitRow = false;
};

Viewed viewBoundaries(func::FuncOp func) {
  Viewed viewed;
  if (func.isPrivate() || func.isExternal()) {
    return viewed;
  }
  OpBuilder builder(func.getContext());
  Block &entry = func.getBody().front();
  SmallVector<Type> inputs(func.getArgumentTypes());
  SmallVector<Type> results(func.getResultTypes());

  builder.setInsertionPointToStart(&entry);
  for (BlockArgument arg : entry.getArguments()) {
    RankedTensorType type = mlir::dyn_cast<RankedTensorType>(arg.getType());
    if (!type || !storedShape(type) ||
        !wholeOnDevice(func.getArgAttrDict(arg.getArgNumber()), type) ||
        llvm::any_of(arg.getUsers(), [](Operation *user) {
          return mlir::isa<MeshShardOp>(user);
        })) {
      continue;
    }
    RankedTensorType dense = withShape(type, *storedShape(type));
    (storedDense(type) ? viewed.dense : viewed.unitRow) = true;
    arg.setType(dense);
    inputs[arg.getArgNumber()] = dense;
    auto view = builder.create<ReshapeOp>(
        func.getLoc(), type, arg,
        builder.getI32ArrayAttr(SmallVector<int32_t>(type.getShape())));
    arg.replaceAllUsesExcept(view.getResult(), view);
    setLocalShape(func.getArgAttrDict(arg.getArgNumber()), dense,
                  [&](StringRef name, Attribute value) {
                    func.setArgAttr(arg.getArgNumber(), name, value);
                  });
  }

  func.walk([&](func::ReturnOp ret) {
    builder.setInsertionPoint(ret);
    for (OpOperand &operand : ret->getOpOperands()) {
      unsigned index = operand.getOperandNumber();
      RankedTensorType type =
          mlir::dyn_cast<RankedTensorType>(operand.get().getType());
      if (!type || !storedShape(type) ||
          !wholeOnDevice(func.getResultAttrDict(index), type) ||
          operand.get().getDefiningOp<MeshShardOp>()) {
        continue;
      }
      RankedTensorType dense = withShape(type, *storedShape(type));
      (storedDense(type) ? viewed.dense : viewed.unitRow) = true;
      auto view = builder.create<ReshapeOp>(
          ret.getLoc(), dense, operand.get(),
          builder.getI32ArrayAttr(SmallVector<int32_t>(dense.getShape())));
      operand.set(view.getResult());
      results[index] = dense;
      setLocalShape(func.getResultAttrDict(index), dense,
                    [&](StringRef name, Attribute value) {
                      func.setResultAttr(index, name, value);
                    });
    }
  });
  func.setType(builder.getFunctionType(inputs, results));
  return viewed;
}

class TTIRDenseFlatPass
    : public PassWrapper<TTIRDenseFlatPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRDenseFlatPass)

  StringRef getArgument() const final { return "ttir-dense-flat"; }
  StringRef getDescription() const final {
    return "Store flat buffers and unit rows in dense tiles";
  }

  void runOnOperation() final {
    if (!enabled()) {
      return;
    }
    ModuleOp module = getOperation();
    Viewed viewed;
    module.walk([&](func::FuncOp func) {
      Viewed boundaries = viewBoundaries(func);
      viewed.dense |= boundaries.dense;
      viewed.unitRow |= boundaries.unitRow;
    });
    // Each family only where boundaries took its form: other programs keep
    // their ops as they are.
    RewritePatternSet patterns(&getContext());
    if (viewed.dense) {
      patterns.add<SinkDenseView, HoistDenseView>(&getContext());
    }
    if (viewed.unitRow) {
      patterns.add<SinkUnitRowView, HoistUnitRowView>(&getContext());
    }
    if (!viewed.dense && !viewed.unitRow) {
      return;
    }
    patterns.add<FoldReshapes>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTTIRDenseFlatPass() {
  return std::make_unique<TTIRDenseFlatPass>();
}

} // namespace mlir::tt::ttir
