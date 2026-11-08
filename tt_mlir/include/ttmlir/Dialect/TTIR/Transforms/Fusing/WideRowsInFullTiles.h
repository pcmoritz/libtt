// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_WIDEROWSINFULLTILES_H
#define LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_WIDEROWSINFULLTILES_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttir::fusing {

// A few rows of a vocabulary, [..., B, V] with B <= 16 and V >= 32768 a
// multiple of 32, fill B of the 32 rows of each of their tiles: a [1, 151936]
// row takes 4748 tiles. Viewed as [..., B, V / C, C], with C a multiple of 32
// that divides V, the same values fill whole tiles ([1, 1187, 128]: 152 of
// them), so a reduction over the last dim, and the elementwise ops that
// compute its input, move and compute about 1/32 of the data.
// Samplers run such reductions over and over: SGLang-TorchTPU's top-p and
// top-k filters search for their cutoff bit by bit, 30 rounds of
// sum(where(probs >= cutoff, probs, 0)) over the vocabulary.
//
// Reductions over the last dim run on the view, over its last two dims.
template <typename ReductionOp>
class ReduceWideRowsInFullTiles : public mlir::OpRewritePattern<ReductionOp> {
public:
  using mlir::OpRewritePattern<ReductionOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReductionOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

// The view of an elementwise op's result is the op on the views of its
// operands, when every user of the result takes the view: operands of the
// result's shape take the same view, operands broadcast along the last dim
// [..., 1, 1], and scalars stay scalars.
class HoistWideRowsViewThroughElementwise
    : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using mlir::OpRewritePattern<ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttir::fusing

#endif
