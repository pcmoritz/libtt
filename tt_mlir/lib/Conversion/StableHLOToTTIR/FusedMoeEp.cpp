// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// tt.fused_moe_ep: a mixture-of-experts layer's experts on one device of an
// expert-parallel mesh, over TTNN's fused MoE kernel (ttir.moe_compute).
//
//   out = tt.fused_moe_ep(x, topk_weights, topk_ids, w1, w3, w2)
//
// Called per device (inside a shard_map) with
//   x            [T, H]          the tokens,
//   topk_weights [T, k]          each token's routing weights,
//   topk_ids     [T, k]          each token's experts; negative ids mark
//                                padding, which is skipped,
//   w1, w3       [E_local, H, I] this device's gate and up projections,
//   w2           [E_local, I, H] and down projections,
// where the experts are split evenly over every device of the mesh in device
// order, so device d holds experts d * E_local to (d + 1) * E_local - 1.
// Returns each token's routing-weighted sum of the SiLU-gated experts this
// device holds, [T, H] in x's type; the layer is the sum over devices.
//
// The kernel stores the experts in BFP4, takes up to 512 tokens per call,
// runs only the (token, expert) pairs of the experts its device owns and
// writes each pair's output unweighted. This lowering chunks the tokens, builds
// the expert-to-device mapping, gives padding an expert id past the experts, and
// sums each token's slots with its weights in one FP32 matmul. The frontend
// attribute `rings` (default 3) sets how many matmul rings run a device's
// experts side by side.

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>

namespace mlir::tt {
namespace {

constexpr int64_t kMaxTokensPerCall = 512;
constexpr uint32_t kDefaultRings = 3;
// Up to this many tokens per call, the combine's placement matrix comes from
// [t, k * t] constants (64 KiB per 8 slots at 64 tokens).
constexpr int64_t kMaxTokensForConstantPlacement = 64;

class FusedMoeEpConversionPattern
    : public OpConversionPattern<stablehlo::CustomCallOp> {
  using OpConversionPattern<stablehlo::CustomCallOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(stablehlo::CustomCallOp op,
                  stablehlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetName() != "tt.fused_moe_ep") {
      return failure();
    }
    if (adaptor.getOperands().size() != 6 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "tt.fused_moe_ep takes x, topk_weights, topk_ids, w1, w3 and "
              "w2 and returns one tensor");
    }
    Value x = adaptor.getOperands()[0];
    Value weights = adaptor.getOperands()[1];
    Value ids = adaptor.getOperands()[2];
    Value w1 = adaptor.getOperands()[3];
    Value w3 = adaptor.getOperands()[4];
    Value w2 = adaptor.getOperands()[5];
    auto xType = cast<RankedTensorType>(x.getType());
    auto idsType = cast<RankedTensorType>(ids.getType());
    auto w1Type = cast<RankedTensorType>(w1.getType());
    auto outType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult(0).getType()));
    if (xType.getRank() != 2 || idsType.getRank() != 2 ||
        w1Type.getRank() != 3) {
      return rewriter.notifyMatchFailure(
          op, "tt.fused_moe_ep expects x [T, H], topk_ids [T, k] and "
              "w1 [E_local, H, I]");
    }

    uint32_t rings = kDefaultRings;
    if (auto attrs = op->getAttrOfType<DictionaryAttr>(
            "mhlo.frontend_attributes")) {
      if (auto value = attrs.getAs<StringAttr>("rings");
          value && value.getValue().getAsInteger(10, rings)) {
        return rewriter.notifyMatchFailure(op, "rings must be an integer");
      }
    }

    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();
    const int64_t tokens = xType.getDimSize(0);
    const int64_t hidden = xType.getDimSize(1);
    const int64_t k = idsType.getDimSize(1);
    const int64_t localExperts = w1Type.getDimSize(0);
    const int64_t intermediate = w1Type.getDimSize(2);
    int64_t devices = 1;
    if (auto meshes = op->getParentOfType<ModuleOp>()
                          ->getAttrOfType<ttcore::MeshesAttr>(
                              ttcore::MeshesAttr::name);
        meshes && !meshes.getMeshes().empty()) {
      for (int64_t size : meshes.getMeshes()[0].getShape()) {
        devices *= size;
      }
    }
    const int64_t experts = localExperts * devices;

    Type bf16 = rewriter.getBF16Type();
    Type f32 = rewriter.getF32Type();
    Type u16 = IntegerType::get(ctx, 16, IntegerType::Unsigned);
    auto typed = [](Value value, ArrayRef<int64_t> shape, Type elementType) {
      return RankedTensorType::get(
          shape, elementType,
          cast<RankedTensorType>(value.getType()).getEncoding());
    };
    auto castTo = [&](Value value, Type elementType) -> Value {
      auto type = cast<RankedTensorType>(value.getType());
      if (type.getElementType() == elementType) {
        return value;
      }
      return rewriter.create<ttir::TypecastOp>(loc, type.clone(elementType),
                                               value);
    };
    auto reshape = [&](Value value, ArrayRef<int64_t> shape) -> Value {
      auto type = cast<RankedTensorType>(value.getType());
      return rewriter.create<ttir::ReshapeOp>(
          loc, type.clone(shape), value,
          rewriter.getI32ArrayAttr(llvm::to_vector_of<int32_t>(shape)));
    };
    auto rows = [&](Value value, int64_t begin, int64_t end) -> Value {
      auto type = cast<RankedTensorType>(value.getType());
      if (begin == 0 && end == type.getDimSize(0)) {
        return value;
      }
      SmallVector<int32_t> begins(type.getRank(), 0);
      SmallVector<int32_t> ends = llvm::to_vector_of<int32_t>(type.getShape());
      SmallVector<int32_t> steps(type.getRank(), 1);
      begins[0] = begin;
      ends[0] = end;
      SmallVector<int64_t> shape(type.getShape());
      shape[0] = end - begin;
      return rewriter.create<ttir::SliceStaticOp>(
          loc, type.clone(shape), value, rewriter.getI32ArrayAttr(begins),
          rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));
    };

    // The kernel's operands: the ids as they are (the kernel reads an int32
    // id's low 16 bits, so that padding's negative ids land past the experts,
    // where it skips them), activations in BF16, and the experts with a
    // leading layer dimension. Its local output reads no scores: the weights
    // only fill the operand.
    Value kernelIds = ids;
    Value kernelX = reshape(castTo(x, bf16), {1, tokens, hidden});
    auto withLayer = [&](Value w) {
      auto shape = cast<RankedTensorType>(w.getType()).getShape();
      return reshape(w, {1, shape[0], shape[1], shape[2]});
    };
    Value kernelW1 = withLayer(w1);
    Value kernelW3 = withLayer(w3);
    Value kernelW2 = withLayer(w2);

    // Every device's row of the mapping gives each expert's device.
    SmallVector<APInt> owners;
    for (int64_t d = 0; d < devices; ++d) {
      for (int64_t e = 0; e < experts; ++e) {
        owners.push_back(APInt(16, e / localExperts));
      }
    }
    auto mappingType = RankedTensorType::get({devices, experts}, u16);
    Value mapping = rewriter.create<ttir::ConstantOp>(
        loc, mappingType, DenseElementsAttr::get(mappingType, owners));

    auto activation = ttcore::MoEActivationFunctionAttr::get(
        ctx, ttcore::MoEActivationFunction::Silu);
    SmallVector<Value> outputs;
    for (int64_t begin = 0; begin < tokens; begin += kMaxTokensPerCall) {
      const int64_t end = std::min(tokens, begin + kMaxTokensPerCall);
      const int64_t t = end - begin;
      Value chunkX = kernelX;
      if (begin != 0 || end != tokens) {
        chunkX = rewriter.create<ttir::SliceStaticOp>(
            loc, typed(kernelX, {1, t, hidden}, bf16), kernelX,
            rewriter.getI32ArrayAttr({0, static_cast<int32_t>(begin), 0}),
            rewriter.getI32ArrayAttr(
                {1, static_cast<int32_t>(end), static_cast<int32_t>(hidden)}),
            rewriter.getI32ArrayAttr({1, 1, 1}));
      }
      // Row j * t + i holds token i's j-th expert output, unweighted.
      Value slots = rewriter.create<ttir::MoeComputeOp>(
          loc, typed(kernelX, {k * t, hidden}, bf16), chunkX,
          rows(kernelIds, begin, end), rows(weights, begin, end), mapping,
          kernelW1, kernelW3, kernelW2, /*bias_0=*/Value(), /*bias_1=*/Value(),
          /*bias_2=*/Value(), rewriter.getUI32IntegerAttr(0),
          rewriter.getUI32IntegerAttr(4),
          rewriter.getUI32IntegerAttr(intermediate), activation,
          /*cluster_axis=*/IntegerAttr(), rewriter.getUI32IntegerAttr(rings));

      // Each token's weighted sum of its slots as one matmul in FP32: row i
      // of the placement matrix holds token i's weights at its slots' rows,
      // weight j at column j * t + i.
      Value chunkWeights = castTo(rows(weights, begin, end), f32);
      Value placement;
      if (t <= kMaxTokensForConstantPlacement) {
        // Spread each weight over its slot's t columns with a 0/1 matmul and
        // keep the token's own column with a 0/1 mask: no reshapes, which
        // cost more than these small matmul and multiply.
        SmallVector<float> spread(k * k * t, 0.0f), own(t * k * t, 0.0f);
        for (int64_t j = 0; j < k; ++j) {
          for (int64_t i = 0; i < t; ++i) {
            spread[j * k * t + j * t + i] = 1.0f;
            own[i * k * t + j * t + i] = 1.0f;
          }
        }
        auto constant = [&](ArrayRef<int64_t> shape, ArrayRef<float> values) {
          auto type = RankedTensorType::get(shape, f32);
          return rewriter.create<ttir::ConstantOp>(
              loc, type, DenseElementsAttr::get(type, values));
        };
        Value spreadWeights = rewriter.create<ttir::MatmulOp>(
            loc, typed(chunkWeights, {t, k * t}, f32), chunkWeights,
            constant({k, k * t}, spread));
        placement = rewriter.create<ttir::MultiplyOp>(
            loc, typed(chunkWeights, {t, k * t}, f32), spreadWeights,
            constant({t, k * t}, own));
      } else {
        SmallVector<float> eye(t * t, 0.0f);
        for (int64_t i = 0; i < t; ++i) {
          eye[i * t + i] = 1.0f;
        }
        auto eyeType = RankedTensorType::get({t, 1, t}, f32);
        placement = reshape(
            rewriter.create<ttir::MultiplyOp>(
                loc, typed(chunkWeights, {t, k, t}, f32),
                rewriter.create<ttir::BroadcastOp>(
                    loc, typed(chunkWeights, {t, k, t}, f32),
                    reshape(chunkWeights, {t, k, 1}),
                    rewriter.getDenseI64ArrayAttr({1, 1, t})),
                rewriter.create<ttir::BroadcastOp>(
                    loc, typed(chunkWeights, {t, k, t}, f32),
                    rewriter.create<ttir::ConstantOp>(
                        loc, eyeType,
                        DenseElementsAttr::get(eyeType, ArrayRef<float>(eye))),
                    rewriter.getDenseI64ArrayAttr({1, k, 1}))),
            {t, k * t});
      }
      Value sum = rewriter.create<ttir::MatmulOp>(
          loc, typed(chunkWeights, {t, hidden}, f32),
          placement, slots);
      outputs.push_back(castTo(sum, outType.getElementType()));
    }
    if (outputs.size() == 1) {
      rewriter.replaceOp(op, outputs.front());
    } else {
      rewriter.replaceOpWithNewOp<ttir::ConcatOp>(op, outType, outputs,
                                                  rewriter.getSI32IntegerAttr(0));
    }
    return success();
  }
};

} // namespace

void populateFusedMoeEpToTTIRPatterns(MLIRContext *context,
                                      RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  patterns.add<FusedMoeEpConversionPattern>(typeConverter, context);
}

} // namespace mlir::tt
