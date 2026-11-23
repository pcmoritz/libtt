// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// tt.attn_res: Inkling 2's block attention residuals in one program
// (ttir.attn_res, ttnn::experimental::attn_res).
//
//   out = tt.attn_res(s_0, .., s_{n-1}[, addend], queries[, weight])
//
// with backend_config num_sources = n, has_addend, has_weight, eps_res,
// eps_norm and normalize; sources and the addend are [T, W], queries and
// the weight [Q, W], out [Q, T, W]. libtt's sglang_tpu_kernels.py emits it
// for decode steps, where the kernel applies (T <= 4, n <= 8, Q <= 3).

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt {
namespace {

class AttnResConversionPattern
    : public OpConversionPattern<stablehlo::CustomCallOp> {
  using OpConversionPattern<stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(stablehlo::CustomCallOp op,
                  stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetName() != "tt.attn_res") {
      return failure();
    }
    auto config = dyn_cast_or_null<DictionaryAttr>(op.getBackendConfigAttr());
    if (!config) {
      config = dyn_cast_or_null<DictionaryAttr>(
          op->getDiscardableAttr("mhlo.backend_config"));
    }
    auto sources = config ? config.getAs<IntegerAttr>("num_sources") : nullptr;
    auto hasAddend = config ? config.getAs<BoolAttr>("has_addend") : nullptr;
    auto hasWeight = config ? config.getAs<BoolAttr>("has_weight") : nullptr;
    auto normalize = config ? config.getAs<BoolAttr>("normalize") : nullptr;
    auto epsRes = config ? config.getAs<FloatAttr>("eps_res") : nullptr;
    auto epsNorm = config ? config.getAs<FloatAttr>("eps_norm") : nullptr;
    if (!sources || !hasAddend || !hasWeight || !normalize || !epsRes ||
        !epsNorm) {
      return rewriter.notifyMatchFailure(
          op, "tt.attn_res takes num_sources, has_addend, has_weight, "
              "normalize, eps_res and eps_norm");
    }
    const int64_t n = sources.getInt();
    ValueRange operands = adaptor.getOperands();
    const int64_t expected =
        n + int64_t{hasAddend.getValue()} + 1 + int64_t{hasWeight.getValue()};
    if (n < 1 || static_cast<int64_t>(operands.size()) != expected ||
        op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "tt.attn_res's operands are its sources, the addend, the "
              "queries and the weight, with one result");
    }
    int64_t next = n;
    Value addend = hasAddend.getValue() ? operands[next++] : Value();
    Value queries = operands[next++];
    Value weight = hasWeight.getValue() ? operands[next++] : Value();
    auto type = getTypeConverter()->convertType(op.getResult(0).getType());
    rewriter.replaceOpWithNewOp<ttir::AttnResOp>(
        op, type, operands.take_front(n), addend, queries, weight,
        rewriter.getF32FloatAttr(epsRes.getValueAsDouble()),
        rewriter.getF32FloatAttr(epsNorm.getValueAsDouble()),
        rewriter.getBoolAttr(normalize.getValue()));
    return success();
  }
};

} // namespace

void populateAttnResToTTIRPatterns(MLIRContext *context,
                                   RewritePatternSet &patterns,
                                   TypeConverter &typeConverter) {
  patterns.add<AttnResConversionPattern>(typeConverter, context);
}

} // namespace mlir::tt
