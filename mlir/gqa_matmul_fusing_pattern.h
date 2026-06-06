#pragma once

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace libtt::mlir_frontend {

inline constexpr llvm::StringLiteral kGroupedHeadExpansionTarget =
    "libtt.grouped_head_expand";

struct GroupedHeadExpansion {
  mlir::Value source;
  llvm::SmallVector<int64_t> logicalShape;
  uint32_t groupedDimension = 0;
  uint32_t groupSize = 0;
};

// Fuses the StableHLO grouped-query head expansion idiom into a backend-private
// marker custom_call consumed by matmul lowering.
//
// Matches:
//   dot_general(reshape(broadcast_in_dim(lhs)), rhs)
//
// Produces:
//   dot_general(stablehlo.custom_call @libtt.grouped_head_expand(lhs), rhs)
class GQAMatmulFusing
    : public mlir::OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::stablehlo::DotGeneralOp dotOp,
                  mlir::PatternRewriter &rewriter) const override;
};

std::optional<GroupedHeadExpansion>
matchPackedGqaMatmulLhsMarker(mlir::stablehlo::DotGeneralOp dotOp,
                              mlir::Value value);

bool isGroupedHeadExpansionMarker(mlir::stablehlo::CustomCallOp customCallOp);

} // namespace libtt::mlir_frontend
