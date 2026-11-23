// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// tt.attn_res: Inkling 2's block attention residuals in one program
// (ttir.attn_res, ttnn::experimental::attn_res), and tt.l2_clip, its
// per-head L2 clips (ttir.l2_clip, ttnn::experimental::l2_clip).
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

// tt.l2_clip(input[, scale][, row_scale]) with backend_config group,
// threshold, epsilon, offset, width, has_scale and has_row_scale:
// ttir.l2_clip.
class L2ClipConversionPattern
    : public OpConversionPattern<stablehlo::CustomCallOp> {
  using OpConversionPattern<stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(stablehlo::CustomCallOp op,
                  stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetName() != "tt.l2_clip") {
      return failure();
    }
    auto config = dyn_cast_or_null<DictionaryAttr>(op.getBackendConfigAttr());
    if (!config) {
      config = dyn_cast_or_null<DictionaryAttr>(
          op->getDiscardableAttr("mhlo.backend_config"));
    }
    auto group = config ? config.getAs<IntegerAttr>("group") : nullptr;
    auto offset = config ? config.getAs<IntegerAttr>("offset") : nullptr;
    auto width = config ? config.getAs<IntegerAttr>("width") : nullptr;
    auto threshold = config ? config.getAs<FloatAttr>("threshold") : nullptr;
    auto epsilon = config ? config.getAs<FloatAttr>("epsilon") : nullptr;
    auto hasScale = config ? config.getAs<BoolAttr>("has_scale") : nullptr;
    auto hasRowScale = config ? config.getAs<BoolAttr>("has_row_scale") : nullptr;
    if (!group || !offset || !width || !threshold || !epsilon || !hasScale ||
        !hasRowScale) {
      return rewriter.notifyMatchFailure(
          op, "tt.l2_clip takes group, offset, width, threshold, epsilon, "
              "has_scale and has_row_scale");
    }
    ValueRange operands = adaptor.getOperands();
    const size_t expected =
        1 + size_t{hasScale.getValue()} + size_t{hasRowScale.getValue()};
    if (operands.size() != expected || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "tt.l2_clip's operands are the input, the scale and the row "
              "scale, with one result");
    }
    size_t next = 1;
    Value scale = hasScale.getValue() ? operands[next++] : Value();
    Value rowScale = hasRowScale.getValue() ? operands[next++] : Value();
    auto type = getTypeConverter()->convertType(op.getResult(0).getType());
    auto i64 = [&](IntegerAttr attr) {
      return rewriter.getI64IntegerAttr(attr.getInt());
    };
    rewriter.replaceOpWithNewOp<ttir::L2ClipOp>(
        op, type, operands[0], scale, rowScale, i64(group),
        rewriter.getF32FloatAttr(threshold.getValueAsDouble()),
        rewriter.getF32FloatAttr(epsilon.getValueAsDouble()), i64(offset),
        i64(width));
    return success();
  }
};

} // namespace

void populateAttnResToTTIRPatterns(MLIRContext *context,
                                   RewritePatternSet &patterns,
                                   TypeConverter &typeConverter) {
  patterns.add<AttnResConversionPattern, L2ClipConversionPattern>(
      typeConverter, context);
}

} // namespace mlir::tt
