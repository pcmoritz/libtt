// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SplitQKVNormRopeDecodeFusing.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn::fusing {

namespace {

// A tiled tensor of BF16 (or FP32 if `float32`); interleaved in DRAM if
// `dram`.
bool isTiled(Value value, bool dram, bool float32 = false) {
  auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(value.getType()).getEncoding());
  return layout && layout.isTiled() &&
         (layout.getDataType() == ttcore::DataType::BFloat16 ||
          (float32 && layout.getDataType() == ttcore::DataType::Float32)) &&
         (!dram || layout.hasInterleavedDRAMTensorMemoryLayout());
}

} // namespace

Operation *buildSplitQKVNormRopeDecode(ttcore::CompositeOp composite,
                                       OpBuilder &builder) {
  auto attributes = composite.getCompositeAttributes();
  auto numQHeads =
      attributes ? attributes->getAs<IntegerAttr>("num_q_heads") : nullptr;
  auto numKVHeads =
      attributes ? attributes->getAs<IntegerAttr>("num_kv_heads") : nullptr;
  auto epsilon = attributes ? attributes->getAs<FloatAttr>("epsilon") : nullptr;
  if (!numQHeads || !numKVHeads || !epsilon ||
      composite.getInputs().size() != 5 || composite->getNumResults() != 2) {
    return nullptr;
  }
  // (qkv rows, q_weight, k_weight, cos, sin) -> (query, key_values). The
  // kernel reads the rows and the cos and sin tables from DRAM, the norm
  // weights too (BF16, or FP32 as 1 + w of a zero-centered norm), and writes
  // BF16 tiles.
  ValueRange inputs = composite.getInputs();
  if (!isTiled(inputs[0], /*dram=*/true) || !isTiled(inputs[3], true) ||
      !isTiled(inputs[4], true) ||
      !isTiled(inputs[1], true, /*float32=*/true) ||
      !isTiled(inputs[2], true, /*float32=*/true) ||
      !isTiled(composite->getResult(0), /*dram=*/false) ||
      !isTiled(composite->getResult(1), /*dram=*/false)) {
    return nullptr;
  }
  return builder.create<SplitQKVNormRopeDecodeOp>(
      composite.getLoc(), composite->getResult(0).getType(),
      composite->getResult(1).getType(), inputs[0], inputs[1], inputs[2],
      inputs[3], inputs[4], numQHeads, numKVHeads, epsilon);
}

} // namespace mlir::tt::ttnn::fusing

namespace mlir::tt::ttnn {

namespace {

// Writes the k and v of every token where the op that reads them wants them:
// the workarounds, after fusing, move them into the paged cache update's L1
// shard.
class WriteKeyValuesInPlace
    : public mlir::OpRewritePattern<SplitQKVNormRopeDecodeOp> {
public:
  using OpRewritePattern<SplitQKVNormRopeDecodeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SplitQKVNormRopeDecodeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    Value keyValues = op.getKeyValues();
    auto move = keyValues.hasOneUse() ? dyn_cast<ToMemoryConfigOp>(
                                            *keyValues.getUsers().begin())
                                      : nullptr;
    if (!move || move.getType() == keyValues.getType()) {
      return mlir::failure();
    }
    auto layout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(move.getType().getEncoding());
    if (!layout || !layout.isTiled() ||
        layout.getDataType() != ttcore::DataType::BFloat16) {
      return mlir::failure();
    }
    rewriter.modifyOpInPlace(op, [&] { keyValues.setType(move.getType()); });
    rewriter.replaceOp(move, keyValues);
    return mlir::success();
  }
};

} // namespace

void SplitQKVNormRopeDecodeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<WriteKeyValuesInPlace>(context);
}

} // namespace mlir::tt::ttnn
