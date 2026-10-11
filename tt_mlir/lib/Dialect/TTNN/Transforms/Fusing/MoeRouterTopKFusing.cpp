// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/MoeRouterTopKFusing.h"

namespace mlir::tt::ttnn::fusing {
namespace {
// Whether `value` is available at `op`: a block argument of its block or
// defined before it in the same block.
bool definedBefore(Value value, Operation *op) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    return arg.getOwner() == op->getBlock();
  }
  Operation *def = value.getDefiningOp();
  return def->getBlock() == op->getBlock() && def->isBeforeInBlock(op);
}

bool allUsersAfter(Value value, Operation *op) {
  return llvm::all_of(value.getUsers(), [&](Operation *user) {
    return user->getBlock() == op->getBlock() && op->isBeforeInBlock(user);
  });
}
} // namespace

// A mixture-of-experts router: the top k of each row of the router logits,
// the softmax of the k values (the renormalized top k of their softmax, see
// TopKSoftmaxRenormalizePattern) and the indices, with the rows of padding
// tokens masked. One program does all of it for up to 32 rows (decode).
LogicalResult
MoeRouterTopKFusing::matchAndRewrite(SoftmaxOp op,
                                     PatternRewriter &rewriter) const {
  auto topk = op.getInput().getDefiningOp<TopKOp>();
  if (!topk || op.getInput() != topk.getValues() ||
      !topk.getValues().hasOneUse() || !topk.getLargest() ||
      topk->getBlock() != op->getBlock()) {
    return failure();
  }
  auto logitsType = cast<RankedTensorType>(topk.getInputTensor().getType());
  auto weightsType = cast<RankedTensorType>(op.getType());
  auto indicesType = cast<RankedTensorType>(topk.getIndices().getType());
  const int64_t rank = logitsType.getRank();
  auto last = [&](int64_t dim) { return dim == -1 || dim == rank - 1; };
  const int64_t k = topk.getK();
  if (rank != 2 || !last(topk.getDim()) || !last(op.getDimension()) ||
      logitsType.getDimSize(0) < 1 || logitsType.getDimSize(0) > 32 ||
      logitsType.getDimSize(1) > 512 || k < 1 || k > 32 ||
      logitsType.getDimSize(1) < k ||
      !(logitsType.getElementType().isF32() ||
        logitsType.getElementType().isBF16()) ||
      !(weightsType.getElementType().isF32() ||
        weightsType.getElementType().isBF16()) ||
      !indicesType.getElementType().isInteger(32)) {
    return failure();
  }

  // The masking of the indices, when it is their only use:
  // where(mask, indices, fill) with a [1, 1] or [rows, 1] mask and one fill.
  WhereOp where;
  Value indices = topk.getIndices();
  if (indices.hasOneUse()) {
    where = dyn_cast<WhereOp>(*indices.getUsers().begin());
  }
  Operation *insertion = topk;
  if (where) {
    auto maskType = cast<RankedTensorType>(where.getFirst().getType());
    auto fillType = cast<RankedTensorType>(where.getThird().getType());
    const bool shapes =
        where.getSecond() == indices && where.getFirst() != indices &&
        where.getThird() != indices && where.getType() == indicesType &&
        maskType.getRank() == 2 && maskType.getDimSize(1) == 1 &&
        (maskType.getDimSize(0) == 1 ||
         maskType.getDimSize(0) == logitsType.getDimSize(0)) &&
        fillType.getNumElements() == 1 && fillType.getRank() == 2 &&
        fillType.getElementType() == indicesType.getElementType() &&
        where->getBlock() == op->getBlock();
    if (!shapes) {
      where = nullptr;
    } else if (!definedBefore(where.getFirst(), topk) ||
               !definedBefore(where.getThird(), topk)) {
      // The fused op must follow the mask and the fill; then it goes where
      // the masking is, ahead of every use of the softmax.
      if (op->isBeforeInBlock(where) && allUsersAfter(op.getResult(), where)) {
        insertion = where;
      } else {
        where = nullptr;
      }
    }
  }

  rewriter.setInsertionPoint(insertion);
  auto fused = rewriter.create<MoeRouterTopKOp>(
      op.getLoc(), TypeRange{weightsType, indicesType}, topk.getInputTensor(),
      where ? where.getFirst() : Value(), where ? where.getThird() : Value(),
      topk.getKAttr());
  rewriter.replaceOp(op, fused.getWeights());
  if (where) {
    rewriter.replaceOp(where, fused.getIndices());
  } else {
    rewriter.replaceAllUsesWith(indices, fused.getIndices());
  }
  rewriter.eraseOp(topk);
  return success();
}

} // namespace mlir::tt::ttnn::fusing
