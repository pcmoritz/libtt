// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVNORMROPEDECODEFUSING_H
#define LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVNORMROPEDECODEFUSING_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::fusing {

// Fuses a decode step's attention prologue as PyTorch models trace it
// (SGLang-TorchTPU's Qwen3, whose KV cache holds k and v side by side):
//
//   qkv [B, (Hq + 2 Hkv) D] -> q heads: rms_norm -> rotary_embedding -> pad
//                              to the cache's width
//                           -> k heads: rms_norm -> rotary_embedding -+
//                           -> v heads -------------------------------+->
//                              concat [B, Hkv, 2 D] -> [1, 1, Hkv, 2 D]
//                              per token
//
// where the heads are reshapes and slices of the projection, into one
// split_qkv_norm_rope_decode (one program instead of about fifteen). It
// anchors on the concat of k and v.
class SplitQKVNormRopeDecodeFusing : public mlir::OpRewritePattern<ConcatOp> {
public:
  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConcatOp concat,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::fusing

#endif
