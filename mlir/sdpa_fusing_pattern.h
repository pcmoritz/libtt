#pragma once

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace libtt::mlir_frontend {

inline constexpr llvm::StringLiteral kSdpaDecodeTarget = "tt.sdpa_decode";

// Fuses the StableHLO decode-attention idiom into a backend-private
// tt.sdpa_decode custom_call.
//
// Matches:
//   transpose(dot_general(softmax((Q @ repeat(gather(K, loc))^T) * scale + mask),
//                         repeat(gather(V, loc))))
//
// Produces:
//   stablehlo.custom_call @tt.sdpa_decode(Q, K, V, seq_lens, loc)
class SDPADecodeFusing
    : public mlir::OpRewritePattern<mlir::stablehlo::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::TransposeOp transposeOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace libtt::mlir_frontend
