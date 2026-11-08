// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/WideRowsInFullTiles.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttir::fusing {

namespace {

constexpr int64_t kTileWidth = 32;
// At most half a tile of rows, at least a vocabulary of columns: the shapes
// whose tiles are mostly padding, as in the 16-row tile kernels of
// top_half_tiles.patch.
constexpr int64_t kMaxRows = 16;
constexpr int64_t kMinColumns = 32768;

RankedTensorType typeOf(Value value) {
  return mlir::cast<RankedTensorType>(value.getType());
}

// [..., B, V] -> [..., B, V / C, C] for the wide few-row shapes, or nullopt.
// C is the widest multiple of 32 up to 1024 that divides V into at least 32
// rows: the reduction over C then runs on many tile rows at once, and the
// one over V / C on few tiles (a [1, 151936] row is [1, 1187, 128]; a
// [1, 248320] row [1, 485, 512]).
std::optional<SmallVector<int64_t>> fullTilesView(ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return std::nullopt;
  }
  int64_t columns = shape.back();
  int64_t rows = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
  if (rows > kMaxRows || columns < kMinColumns) {
    return std::nullopt;
  }
  for (int64_t width = 1024; width >= kTileWidth; width /= 2) {
    if (columns % width == 0 && columns / width >= kTileWidth) {
      SmallVector<int64_t> view(shape.drop_back());
      view.push_back(columns / width);
      view.push_back(width);
      return view;
    }
  }
  return std::nullopt;
}

// The view of an operand of an elementwise op whose result `shape` takes the
// view `view`: an operand of the result's shape takes `view`, one broadcast
// along the last dim [..., 1, 1], one broadcast along others its own dims
// with the last split as in `view`, and a scalar stays ones. nullopt when the
// operand broadcasts otherwise.
std::optional<SmallVector<int64_t>> operandView(RankedTensorType operandType,
                                                ArrayRef<int64_t> shape,
                                                ArrayRef<int64_t> view) {
  if (operandType.getNumElements() == 1) {
    return SmallVector<int64_t>(view.size(), 1);
  }
  ArrayRef<int64_t> operandShape = operandType.getShape();
  if (operandShape.size() != shape.size()) {
    return std::nullopt;
  }
  for (auto [operandDim, dim] :
       llvm::zip_equal(operandShape.drop_back(), shape.drop_back())) {
    if (operandDim != dim && operandDim != 1) {
      return std::nullopt;
    }
  }
  SmallVector<int64_t> result(operandShape.drop_back());
  if (operandShape.back() == shape.back()) {
    result.append(view.take_back(2).begin(), view.take_back(2).end());
  } else if (operandShape.back() == 1) {
    result.append({1, 1});
  } else {
    return std::nullopt;
  }
  return result;
}

} // namespace

template <typename ReductionOp>
mlir::LogicalResult ReduceWideRowsInFullTiles<ReductionOp>::matchAndRewrite(
    ReductionOp op, mlir::PatternRewriter &rewriter) const {
  RankedTensorType inputType = typeOf(op.getInput());
  int64_t rank = inputType.getRank();
  std::optional<ArrayAttr> dims = op.getDimArg();
  if (!dims || dims->size() != 1) {
    return mlir::failure();
  }
  int64_t dim = mlir::cast<IntegerAttr>((*dims)[0]).getInt();
  if (dim < 0) {
    dim += rank;
  }
  std::optional<SmallVector<int64_t>> view =
      fullTilesView(inputType.getShape());
  if (dim != rank - 1 || !view) {
    return mlir::failure();
  }

  // One reduction over both dims of the view: TTNN reduces C on the view's
  // tile rows in parallel and then V / C (measured faster than two
  // reductions, as tt_mlir_wide_row_reduction.patch splits them in groups of
  // 32, which then applies to neither).
  Value input =
      utils::createReshapeOp(rewriter, op.getLoc(), op.getInput(), *view);
  bool keepDim = op.getKeepDim();
  SmallVector<int64_t> reducedShape(view->begin(), view->end() - 2);
  if (keepDim) {
    reducedShape.append({1, 1});
  }
  RankedTensorType resultType = typeOf(op.getResult());
  auto reduced = rewriter.create<ReductionOp>(
      op.getLoc(),
      RankedTensorType::get(reducedShape, resultType.getElementType(),
                            resultType.getEncoding()),
      input, rewriter.getBoolAttr(keepDim),
      rewriter.getI32ArrayAttr({static_cast<int32_t>(rank - 1),
                                static_cast<int32_t>(rank)}));
  Value result = reduced.getResult();
  if (keepDim) {
    result = utils::createReshapeOp(rewriter, op.getLoc(), result,
                                    resultType.getShape());
  }
  rewriter.replaceOp(op, result);
  return mlir::success();
}

template class ReduceWideRowsInFullTiles<SumOp>;
template class ReduceWideRowsInFullTiles<MaxOp>;
template class ReduceWideRowsInFullTiles<MinOp>;

mlir::LogicalResult HoistWideRowsViewThroughElementwise::matchAndRewrite(
    ReshapeOp op, mlir::PatternRewriter &rewriter) const {
  Operation *producer = op.getInput().getDefiningOp();
  if (!producer || producer->getNumResults() != 1 ||
      !mlir::isa<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>(
          producer)) {
    return mlir::failure();
  }
  RankedTensorType type = typeOf(op.getInput());
  std::optional<SmallVector<int64_t>> view = fullTilesView(type.getShape());
  if (!view || typeOf(op.getResult()).getShape() != ArrayRef<int64_t>(*view)) {
    return mlir::failure();
  }
  // Every user takes the view, so the op in the original shape goes away.
  for (Operation *user : producer->getUsers()) {
    auto reshape = mlir::dyn_cast<ReshapeOp>(user);
    if (!reshape || reshape.getType() != op.getType()) {
      return mlir::failure();
    }
  }

  // At the op, where its operands are defined.
  rewriter.setInsertionPoint(producer);
  SmallVector<Value> operands;
  for (Value operand : producer->getOperands()) {
    // A broadcast operand (a cutoff per row, broadcast over the vocabulary)
    // broadcasts its input in the view instead: materialized, it would be as
    // wide as the vocabulary.
    if (auto broadcast = operand.getDefiningOp<BroadcastOp>()) {
      if (std::optional<SmallVector<int64_t>> inputShape = operandView(
              typeOf(broadcast.getInput()), type.getShape(), *view)) {
        operands.push_back(utils::createReshapeOp(
            rewriter, producer->getLoc(), broadcast.getInput(), *inputShape));
        continue;
      }
    }
    std::optional<SmallVector<int64_t>> operandShape =
        operandView(typeOf(operand), type.getShape(), *view);
    if (!operandShape) {
      return mlir::failure();
    }
    operands.push_back(utils::createReshapeOp(rewriter, producer->getLoc(),
                                              operand, *operandShape));
  }
  Operation *viewed = rewriter.create(
      producer->getLoc(),
      rewriter.getStringAttr(producer->getName().getStringRef()), operands,
      TypeRange{RankedTensorType::get(*view, type.getElementType(),
                                      type.getEncoding())},
      producer->getAttrs());
  for (Operation *user : llvm::make_early_inc_range(producer->getUsers())) {
    rewriter.replaceOp(user, viewed->getResult(0));
  }
  return mlir::success();
}

} // namespace mlir::tt::ttir::fusing
