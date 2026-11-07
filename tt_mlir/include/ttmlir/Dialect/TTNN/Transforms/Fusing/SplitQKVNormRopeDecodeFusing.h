// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVNORMROPEDECODEFUSING_H
#define LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_SPLITQKVNORMROPEDECODEFUSING_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::fusing {

// Fuses a decode step's attention prologue as PyTorch models trace it
// (SGLang-TorchTPU's Qwen3 and Qwen3.8, whose KV cache holds k and v side by
// side):
//
//   qkv [B, (s Hq + 2 Hkv) D] -> q heads: rms_norm -> rotary_embedding -> pad
//                                to the cache's width
//                             -> k heads: rms_norm -> rotary_embedding -+
//                             -> v heads -------------------------------+->
//                                concat [B, Hkv, 2 D] -> [1, B, Hkv, 2 D]
//
// where the heads are reshapes and slices of the projection, each q head
// followed by its gate head in gated attention (s = 2), and a partial rotary
// rotates the first R dims of the normed heads and concatenates the rest
// after them, into one split_qkv_norm_rope_decode (one program instead of
// fifteen or more). It anchors on the concat of k and v.
class SplitQKVNormRopeDecodeFusing : public mlir::OpRewritePattern<ConcatOp> {
public:
  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConcatOp concat,
                  mlir::PatternRewriter &rewriter) const override;
};

// Fuses the output gate of gated attention as PyTorch models trace it
// (SGLang-TorchTPU's Qwen3.8):
//
//   attention [1, T, H, W] -> slice the values [..., v0 : v0 + D] -> reshape
//   [T, H D] -------------------------------------------------------------+
//   qkv [T, N] -> reshape, slice the gate heads (each S after the last, the  +-> multiply
//   first at g0) -> typecast to FP32 -> sigmoid -> typecast to BF16 ->      |
//   reshape [T, H D] -------------------------------------------------------+
//
// into one attention_output_gate (one program instead of nine), with the same
// roundings. It anchors on the multiply.
class AttentionOutputGateFusing : public mlir::OpRewritePattern<MultiplyOp> {
public:
  using OpRewritePattern<MultiplyOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MultiplyOp multiply,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::fusing

#endif
