// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

namespace {

// Writes the k and v of every token where the op that reads them wants them:
// the workarounds, after fusing, move them into the paged cache update's L1
// shard.
class WriteKeyValuesInPlace
    : public mlir::OpRewritePattern<SplitQKVNormRopeDecodeOp> {
public:
  using OpRewritePattern<SplitQKVNormRopeDecodeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SplitQKVNormRopeDecodeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Value keyValues = op.getKeyValues();
    auto move = keyValues.hasOneUse() ? dyn_cast<ToMemoryConfigOp>(
                                            *keyValues.getUsers().begin())
                                      : nullptr;
    if (!move || move.getType() == keyValues.getType()) {
      return mlir::failure();
    }
    auto layout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(move.getType().getEncoding());
    if (!layout || !layout.isTiled() ||
        layout.getDataType() != ttcore::DataType::BFloat16) {
      return mlir::failure();
    }
    rewriter.modifyOpInPlace(op, [&] { keyValues.setType(move.getType()); });
    rewriter.replaceOp(move, keyValues);
    return mlir::success();
  }
};

} // namespace

void SplitQKVNormRopeDecodeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<WriteKeyValuesInPlace>(context);
}

} // namespace mlir::tt::ttnn
