#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace libtt::mlir_frontend {

inline constexpr const char* kSelectSplatTarget = "tt.select_splat";

class SelectSplatFusing
    : public mlir::OpRewritePattern<mlir::stablehlo::SelectOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::SelectOp selectOp,
                  mlir::PatternRewriter& rewriter) const override;
};

mlir::LogicalResult runSelectSplatFusing(mlir::MLIRContext& context,
                                         mlir::ModuleOp module);

} // namespace libtt::mlir_frontend
