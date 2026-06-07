#pragma once

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace libtt::mlir_frontend {

inline constexpr llvm::StringLiteral kRmsNormTarget = "tt.rms_norm";

// Fuses the StableHLO RMSNorm idiom into a backend-private tt.rms_norm
// custom_call.
//
// Matches the BF16 LLM form:
//   convert((x - 0) * (rsqrt(reduce(x * x) * scale + epsilon) * gamma))
//
// Produces:
//   stablehlo.custom_call @tt.rms_norm(x, gamma)
class RMSNormFusing
    : public mlir::OpRewritePattern<mlir::stablehlo::ConvertOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::ConvertOp convertOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace libtt::mlir_frontend
