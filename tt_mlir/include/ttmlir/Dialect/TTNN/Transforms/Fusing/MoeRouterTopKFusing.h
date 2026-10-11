// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
namespace mlir::tt::ttnn::fusing {
// softmax(topk(x).values) [, where(mask, topk(x).indices, fill)] -> moe_router_topk
class MoeRouterTopKFusing : public OpRewritePattern<SoftmaxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(SoftmaxOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn::fusing
