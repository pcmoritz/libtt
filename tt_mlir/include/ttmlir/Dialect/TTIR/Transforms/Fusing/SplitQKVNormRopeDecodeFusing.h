// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_SPLITQKVNORMROPEDECODEFUSING_H
#define LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_SPLITQKVNORMROPEDECODEFUSING_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::ttir::fusing {

// The composite this fusion makes, which TTNNResolveComposites turns into
// ttnn.split_qkv_norm_rope_decode.
inline constexpr llvm::StringLiteral kSplitQKVNormRopeDecode =
    "split_qkv_norm_rope_decode";

// Fuses a decode step's attention prologue as PyTorch models trace it
// (SGLang-TorchTPU's Qwen3 and Qwen3.8, whose KV cache holds k and v side by
// side):
//
//   qkv [B, (s Hq + 2 Hkv) D] -> q heads: rms_norm -> rotary_embedding (->
//                                pad to the cache's width, for prefill)
//                             -> k heads: rms_norm -> rotary_embedding -+
//                             -> v heads -------------------------------+->
//                                concat [B, Hkv, 2 D] -> [1, B, Hkv, 2 D]
//
// where the heads are reshapes and slices of the projection, each q head
// followed by its gate head in gated attention (s = 2), the rotary
// embeddings are the composites of the RoPE fusions, and a partial rotary
// rotates the first R dims of the normed heads and concatenates the rest
// after them. It anchors on the concat of k and v.
//
// The prologue becomes one ttcore.composite, kSplitQKVNormRopeDecode, of
// (qkv as [B, W], q_weight, k_weight, cos, sin), whose decomposition holds
// the ops it replaces. TTNNResolveComposites promotes it to the fused op when
// the layouts TTNNLayout chose suit the kernel and inlines the decomposition
// otherwise. Matched on TTIR, the prologue keeps its shapes from the passes
// that change shapes before lowering (ttir-physical-layouts), as XLA forms
// its library calls before it assigns layouts.
class SplitQKVNormRopeDecodeFusing : public mlir::OpRewritePattern<ConcatOp> {
public:
  using OpRewritePattern<ConcatOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConcatOp concat,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttir::fusing

#endif
