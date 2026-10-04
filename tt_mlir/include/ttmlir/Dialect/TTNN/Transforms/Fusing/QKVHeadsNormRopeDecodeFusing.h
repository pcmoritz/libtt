// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_QKVHEADSNORMROPEDECODEFUSING_H
#define LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_QKVHEADSNORMROPEDECODEFUSING_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::fusing {

// Fuses the decode attention prologue that NLPDecodeQKVProjectionFusing and
// the decode RoPE fusion leave behind:
//
//   nlp_create_qkv_heads_decode -> q: rms_norm -> rotary_embedding
//                               -> k: rms_norm -> rotary_embedding
//                               -> v
//
// with token_index 0 and the same cos/sin for q and k, into one
// nlp_create_qkv_heads_decode_norm_rope (one program instead of five). Its
// weight row per head is a repeat and concat of the two norm weights, which
// const-eval folds for model weights.
class NLPCreateQKVHeadsDecodeNormRopeFusing
    : public mlir::OpRewritePattern<NLPCreateQKVHeadsDecodeOp> {
public:
  using OpRewritePattern<NLPCreateQKVHeadsDecodeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(NLPCreateQKVHeadsDecodeOp decodeOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::fusing

#endif
