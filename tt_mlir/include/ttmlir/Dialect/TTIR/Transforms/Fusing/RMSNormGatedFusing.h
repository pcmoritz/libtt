// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_RMSNORMGATEDFUSING_H
#define LIBTT_TT_MLIR_INCLUDE_TTMLIR_DIALECT_TTIR_TRANSFORMS_FUSING_RMSNORMGATEDFUSING_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttir::fusing {

// Fuses the gated RMS norm of linear-attention layers, as PyTorch models
// trace it (SGLang-TorchTPU's Qwen3.5 and Qwen3.8 gated DeltaNet output,
// after the RMS norm and SiLU fusions):
//
//   multiply(rms_norm(x, w), silu(z)) [-> typecast] [-> reshape]
//
// over the last dimension D (up to 128, a head) into one rms_norm_gated.
// The op computes in FP32 and reads x where reshapes and BF16 / FP32
// typecasts lead back to (the FP32 output of the gated delta rule, one BF16
// rounding fewer), makes a typecast result in that type, and a reshape of it
// to rows of several heads, [tokens, heads * D], in that layout. z, a slice
// of heads of a projection (SGLang's qkvz), is read in place from the
// projection's columns.
class RMSNormGatedFusing : public mlir::OpRewritePattern<MultiplyOp> {
public:
  using OpRewritePattern<MultiplyOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(MultiplyOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttir::fusing

#endif
