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
// the expert-to-device mapping, and sums each token's slots with its weights in
// one matmul accumulated in FP32; the kernel reads the ids as they are and
// skips padding's negative ids. The frontend attribute `rings` (default 3)
// sets how many matmul rings run a device's experts side by side.

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace mlir::tt {
namespace {

constexpr int64_t kMaxTokensPerCall = 512;
constexpr uint32_t kDefaultRings = 3;

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
    Value kernelX = castTo(x, bf16);
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
    // A 0/1 matrix, 1 where row r's index rows[r] equals column c's cols[c]:
    // only the two index vectors are constants, and the comparison runs once,
    // when the program loads.
    auto mask = [&](ArrayRef<float> rows, ArrayRef<float> cols) -> Value {
      auto vector = [&](ArrayRef<int64_t> shape, ArrayRef<float> values) {
        auto type = RankedTensorType::get(shape, f32);
        return rewriter.create<ttir::ConstantOp>(
            loc, type, DenseElementsAttr::get(type, values));
      };
      const int64_t m = rows.size(), n = cols.size();
      return rewriter.create<ttir::EqualOp>(
          loc, RankedTensorType::get({m, n}, f32), vector({m, 1}, rows),
          vector({1, n}, cols));
    };
    SmallVector<Value> outputs;
    for (int64_t begin = 0; begin < tokens; begin += kMaxTokensPerCall) {
      const int64_t end = std::min(tokens, begin + kMaxTokensPerCall);
      const int64_t t = end - begin;
      // Row j * t + i holds token i's j-th expert output, unweighted.
      Value slots = rewriter.create<ttir::MoeComputeOp>(
          loc, typed(kernelX, {k * t, hidden}, bf16),
          reshape(rows(kernelX, begin, end), {1, t, hidden}),
          rows(ids, begin, end), rows(weights, begin, end), mapping, kernelW1,
          kernelW3, kernelW2, /*bias_0=*/Value(), /*bias_1=*/Value(),
          /*bias_2=*/Value(), rewriter.getUI32IntegerAttr(0),
          rewriter.getUI32IntegerAttr(4),
          rewriter.getUI32IntegerAttr(intermediate), activation,
          /*cluster_axis=*/IntegerAttr(), rewriter.getUI32IntegerAttr(rings));

      // Each token's weighted sum of its slots as one matmul in FP32: row i
      // of the placement matrix holds token i's weight j at column j * t + i,
      // its weights spread over t columns each and masked to its own column.
      // One token's slots are rows 0 to k - 1: its weights as they are.
      Value chunkWeights = castTo(rows(weights, begin, end), f32);
      Value placement = chunkWeights;
      if (t > 1) {
        // Column j * t + i is slot j of token i.
        SmallVector<float> slot(k), token(t), slotOf(k * t), tokenOf(k * t);
        std::iota(slot.begin(), slot.end(), 0.0f);
        std::iota(token.begin(), token.end(), 0.0f);
        for (int64_t c = 0; c < k * t; ++c) {
          slotOf[c] = c / t;
          tokenOf[c] = c % t;
        }
        Value spread = mask(slot, slotOf);
        Value own = mask(token, tokenOf);
        auto placementType = typed(chunkWeights, {t, k * t}, f32);
        placement = rewriter.create<ttir::MultiplyOp>(
            loc, placementType,
            rewriter.create<ttir::MatmulOp>(loc, placementType, chunkWeights,
                                            spread),
            own);
      }
      // The matmul accumulates in FP32 and writes the output's type.
      outputs.push_back(rewriter.create<ttir::MatmulOp>(
          loc, typed(chunkWeights, {t, hidden}, outType.getElementType()),
          placement, slots));
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
