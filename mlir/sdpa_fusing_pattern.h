#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
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

private:
  struct RepeatedCacheMatch;
  struct ScorePathMatch;
  struct Components;

  // Custom call / identity utilities.
  static bool isIdentityCustomCall(mlir::stablehlo::CustomCallOp customCallOp);
  static mlir::Value peelIdentityCustomCalls(mlir::Value value);

  template <typename OpTy>
  static OpTy definingOpSkippingIdentityCustomCalls(mlir::Value value) {
    return peelIdentityCustomCalls(value).template getDefiningOp<OpTy>();
  }

  // Type / shape utilities.
  static std::optional<mlir::RankedTensorType> getStaticRankedTensor(mlir::Value value);
  static bool int64ArrayEquals(llvm::ArrayRef<int64_t> values,
                               std::initializer_list<int64_t> expected);
  static bool isStaticBf16Tensor(mlir::Value value);
  static bool isS32TensorWithLength(mlir::Value value, int64_t length);

  // Constant extraction.
  static std::optional<uint32_t> bf16PackedConstant(mlir::Value value);

  // Pattern matching.
  static std::optional<mlir::Value> findS32TensorWithLength(mlir::Value value, int64_t length);
  static std::optional<mlir::Value> findS32TensorWithLength(
      mlir::Value value, int64_t length, llvm::DenseSet<mlir::Value> &visited);
  static std::optional<mlir::stablehlo::GatherOp> gatherFromCacheValue(mlir::Value value);
  static std::optional<RepeatedCacheMatch> matchRepeatedCache(mlir::Value value, int64_t qHeads);
  static std::optional<mlir::Value> peelSoftmaxInput(mlir::Value probabilities);
  static std::optional<ScorePathMatch> matchScorePath(mlir::Value maskedScores,
                                                      int64_t qHeads,
                                                      int64_t keyTokens);
  static std::optional<Components> matchSdpaDecode(mlir::stablehlo::TransposeOp transposeOp);

  // Op creation.
  static mlir::LogicalResult createSdpaDecodeOp(mlir::PatternRewriter &rewriter,
                                                mlir::stablehlo::TransposeOp root,
                                                const Components &components);
};

} // namespace libtt::mlir_frontend
