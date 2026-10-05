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

// Fuses the decode attention prologue of a projection whose q heads are each
// followed by an output gate head, as in Qwen3.5, which
// nlp_create_qkv_heads_decode does not split:
//
//   matmul -> [q|gate] heads: reshape [.., 2 * D] -> slice q -> rms_norm ->
//   rope
//                                                -> slice gate
//          -> k: rms_norm -> rope
//          -> v: reshape [1, B, Hkv, D]
//
// where rope is a rotary_embedding with token_index 0 of the whole head or of
// its first elements, sliced off and concatenated back. q, k and v become the
// results of one nlp_create_qkv_heads_decode_norm_rope with gated_query; the
// gate keeps its slices of the projection.
class GatedQKVHeadsDecodeNormRopeFusing
    : public mlir::OpRewritePattern<MatmulOp> {
public:
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MatmulOp matmulOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::fusing

#endif
