// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

namespace {

// Writes the k and v of every token (k and v apart, in JAX models) where the
// ops that read them want them: the workarounds, after fusing, move them into
// the paged cache updates' L1 shards.
class WriteKeyValuesInPlace
    : public mlir::OpRewritePattern<SplitQKVNormRopeDecodeOp> {
public:
  using OpRewritePattern<SplitQKVNormRopeDecodeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SplitQKVNormRopeDecodeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool changed = false;
    for (Value result : {op.getKeyValues(), op.getValue()}) {
      if (!result) {
        continue;
      }
      auto move = result.hasOneUse() ? dyn_cast<ToMemoryConfigOp>(
                                           *result.getUsers().begin())
                                     : nullptr;
      if (!move || move.getType() == result.getType()) {
        continue;
      }
      auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          move.getType().getEncoding());
      if (!layout || !layout.isTiled() ||
          layout.getDataType() != ttcore::DataType::BFloat16) {
        continue;
      }
      rewriter.modifyOpInPlace(op, [&] { result.setType(move.getType()); });
      rewriter.replaceOp(move, result);
      changed = true;
    }
    return mlir::success(changed);
  }
};

} // namespace

void SplitQKVNormRopeDecodeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<WriteKeyValuesInPlace>(context);
}

} // namespace mlir::tt::ttnn
