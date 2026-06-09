#pragma once

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace libtt::mlir_frontend {

inline constexpr llvm::StringLiteral kRopeTarget = "tt.rope";

// Fuses the Neox-style split-half RoPE idiom:
//   concat(x1 * cos - x2 * sin, x2 * cos + x1 * sin)
//
// Produces:
//   stablehlo.custom_call @tt.rope(x, cos, sin)
class RopeFusing
    : public mlir::OpRewritePattern<mlir::stablehlo::ConcatenateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::ConcatenateOp concatOp,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace libtt::mlir_frontend
