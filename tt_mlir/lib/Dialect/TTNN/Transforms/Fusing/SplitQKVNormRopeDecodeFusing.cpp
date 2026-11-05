// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SplitQKVNormRopeDecodeFusing.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttnn::fusing {

namespace {

// The single user of a value that is not a deallocation, or nullptr.
Operation *getSingleNonDeallocUser(Value value) {
  Operation *found = nullptr;
  for (Operation *user : value.getUsers()) {
    if (isa<DeallocateOp>(user)) {
      continue;
    }
    if (found) {
      return nullptr;
    }
    found = user;
  }
  return found;
}

Value throughReshapes(Value value) {
  while (auto reshape = value.getDefiningOp<ReshapeOp>()) {
    value = reshape.getInput();
  }
  return value;
}

// The end of the chain of reshapes that are each the single user of the
// previous value, from `value`; the chain's ops are added to `ops`.
Value alongReshapes(Value value, llvm::SetVector<Operation *> &ops) {
  while (auto reshape =
             dyn_cast_or_null<ReshapeOp>(getSingleNonDeallocUser(value))) {
    ops.insert(reshape);
    value = reshape.getResult();
  }
  return value;
}

ArrayRef<int64_t> shapeOf(Value value) {
  return mlir::cast<RankedTensorType>(value.getType()).getShape();
}

// The value that reshapes and static slices of `value` read, and the flat
// (row-major) index there of `value`'s element `flat`.
std::pair<Value, int64_t> sourceOf(Value value, int64_t flat) {
  for (;;) {
    if (auto reshape = value.getDefiningOp<ReshapeOp>()) {
      value = reshape.getInput();
      continue;
    }
    auto slice = value.getDefiningOp<SliceStaticOp>();
    if (!slice) {
      return {value, flat};
    }
    ArrayRef<int64_t> outShape = shapeOf(slice.getResult());
    ArrayRef<int64_t> inShape = shapeOf(slice.getInput());
    int64_t inFlat = 0;
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(outShape.size()) - 1; i >= 0; --i) {
      const int64_t index = flat % outShape[i];
      flat /= outShape[i];
      const int64_t begin =
          mlir::cast<IntegerAttr>(slice.getBegins()[i]).getInt();
      const int64_t step = mlir::cast<IntegerAttr>(slice.getStep()[i]).getInt();
      inFlat += (begin + index * step) * stride;
      stride *= inShape[i];
    }
    value = slice.getInput();
    flat = inFlat;
  }
}

// Whether `value` holds heads [first, first + heads) of every row of `qkv`
// [tokens, width], head_dim wide, in (token, head, element) order.
bool isHeadView(Value value, Value qkv, int64_t tokens, int64_t width,
                int64_t first, int64_t heads, int64_t headDim) {
  const int64_t rowElements = heads * headDim;
  if (mlir::cast<RankedTensorType>(value.getType()).getNumElements() !=
      tokens * rowElements) {
    return false;
  }
  for (int64_t i = 0; i < tokens * rowElements; ++i) {
    auto [source, flat] = sourceOf(value, i);
    if (source != qkv ||
        flat != i / rowElements * width + first * headDim + i % rowElements) {
      return false;
    }
  }
  return true;
}

// The ops between `value` and `qkv`, for erasing once unused.
void addViewOps(Value value, Value qkv, llvm::SetVector<Operation *> &ops) {
  while (value != qkv) {
    Operation *def = value.getDefiningOp();
    ops.insert(def);
    value = def->getOperand(0);
  }
}

// The fused kernel rounds as rms_norm and rotary_embedding do at HiFi4 with
// FP32 sums, which the O0 pipeline sets on ops without a config.
bool hasHiFi4FP32Sums(DeviceComputeKernelConfigAttr config) {
  if (!config) {
    return true;
  }
  if (config.getMathFidelity() &&
      *config.getMathFidelity() != MathFidelity::HiFi4) {
    return false;
  }
  return !config.getFp32DestAccEn() || config.getFp32DestAccEn().getValue();
}

// A tiled BF16 tensor; interleaved in DRAM if `dram`.
bool isBF16Tile(Value value, bool dram) {
  auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(value.getType()).getEncoding());
  return layout && layout.isTiled() &&
         layout.getDataType() == ttcore::DataType::BFloat16 &&
         (!dram || layout.hasInterleavedDRAMTensorMemoryLayout());
}

} // namespace

mlir::LogicalResult SplitQKVNormRopeDecodeFusing::matchAndRewrite(
    ConcatOp concat, mlir::PatternRewriter &rewriter) const {
  // concat(k, v) [B, Hkv, D] each, along the last dim.
  ArrayRef<int64_t> kvShape = shapeOf(concat.getResult());
  if (concat.getInputs().size() != 2 || kvShape.size() != 3 ||
      (concat.getDim() != 2 && concat.getDim() != -1) || kvShape[2] % 2 != 0) {
    return mlir::failure();
  }
  const int64_t tokens = kvShape[0];
  const int64_t numKVHeads = kvShape[1];
  const int64_t headDim = kvShape[2] / 2;
  Value kHeads = concat.getInputs()[0];
  Value vHeads = concat.getInputs()[1];
  if (tokens > 32 || numKVHeads > 32 || headDim % 64 != 0 ||
      shapeOf(kHeads) != ArrayRef<int64_t>{tokens, numKVHeads, headDim}) {
    return mlir::failure();
  }

  // k's rotary embedding and norm, and the projection the norm reads.
  auto kRope = throughReshapes(kHeads).getDefiningOp<RotaryEmbeddingOp>();
  if (!kRope) {
    return mlir::failure();
  }
  auto kNorm = throughReshapes(kRope.getInput()).getDefiningOp<RMSNormOp>();
  if (!kNorm) {
    return mlir::failure();
  }
  Value qkv = sourceOf(kNorm.getInput(), 0).first;
  // q's: the rotary embedding of the same cos and sin (which every layer
  // shares) whose norm reads the same projection.
  RotaryEmbeddingOp qRope;
  RMSNormOp qNorm;
  for (Operation *user : kRope.getCosCache().getUsers()) {
    auto rope = dyn_cast<RotaryEmbeddingOp>(user);
    if (!rope || rope == kRope || rope.getSinCache() != kRope.getSinCache() ||
        rope.getTokenIndex() != kRope.getTokenIndex()) {
      continue;
    }
    auto norm = throughReshapes(rope.getInput()).getDefiningOp<RMSNormOp>();
    if (!norm || sourceOf(norm.getInput(), 0).first != qkv) {
      continue;
    }
    if (qRope) {
      return mlir::failure();
    }
    qRope = rope;
    qNorm = norm;
  }
  if (!qRope || !qNorm.getWeight() || !kNorm.getWeight() || qNorm.getBias() ||
      kNorm.getBias() || qNorm.getEpsilonAttr() != kNorm.getEpsilonAttr()) {
    return mlir::failure();
  }
  for (DeviceComputeKernelConfigAttr config :
       {qNorm.getComputeConfigAttr(), kNorm.getComputeConfigAttr(),
        qRope.getComputeConfigAttr(), kRope.getComputeConfigAttr()}) {
    if (!hasHiFi4FP32Sums(config)) {
      return mlir::failure();
    }
  }

  // The heads are views of one projection [B, (Hq + 2 Hkv) D].
  auto qkvType = mlir::cast<RankedTensorType>(qkv.getType());
  const int64_t width = qkvType.getShape().back();
  const int64_t qElements =
      mlir::cast<RankedTensorType>(qNorm.getInput().getType()).getNumElements();
  if (qkvType.getNumElements() != tokens * width ||
      qElements % (tokens * headDim) != 0) {
    return mlir::failure();
  }
  const int64_t numQHeads = qElements / (tokens * headDim);
  if (width != (numQHeads + 2 * numKVHeads) * headDim ||
      !isHeadView(qNorm.getInput(), qkv, tokens, width, 0, numQHeads,
                  headDim) ||
      !isHeadView(kNorm.getInput(), qkv, tokens, width, numQHeads, numKVHeads,
                  headDim) ||
      !isHeadView(vHeads, qkv, tokens, width, numQHeads + numKVHeads,
                  numKVHeads, headDim)) {
    return mlir::failure();
  }

  // The rotary embeddings take [1, B, H, D] with token b's row of cos and
  // sin: row 0 by token_index 0 for one token, one row per token for several.
  Value cos = qRope.getCosCache();
  Value sin = qRope.getSinCache();
  const int64_t tableRows = tokens == 1 ? 1 : tokens;
  if (shapeOf(qRope.getInput()) !=
          ArrayRef<int64_t>{1, tokens, numQHeads, headDim} ||
      shapeOf(kRope.getInput()) !=
          ArrayRef<int64_t>{1, tokens, numKVHeads, headDim} ||
      shapeOf(cos) != ArrayRef<int64_t>{1, tableRows, 1, headDim} ||
      shapeOf(sin) != shapeOf(cos) ||
      (tokens == 1 ? qRope.getTokenIndex() != std::optional<uint32_t>(0)
                   : qRope.getTokenIndex().has_value())) {
    return mlir::failure();
  }
  Value qWeight = qNorm.getWeight();
  Value kWeight = kNorm.getWeight();
  for (Value operand : {qkv, qWeight, kWeight, cos, sin}) {
    if (!isBF16Tile(operand, /*dram=*/true)) {
      return mlir::failure();
    }
  }
  for (Value weight : {qWeight, kWeight}) {
    if (mlir::cast<RankedTensorType>(weight.getType()).getNumElements() !=
        headDim) {
      return mlir::failure();
    }
  }

  // Everything between the projection and the outputs goes, so each value on
  // the way must feed only the next op.
  llvm::SetVector<Operation *> chain;
  for (RMSNormOp norm : {qNorm, kNorm}) {
    chain.insert(norm);
    addViewOps(norm.getInput(), qkv, chain);
  }
  addViewOps(vHeads, qkv, chain);
  for (RotaryEmbeddingOp rope : {qRope, kRope}) {
    chain.insert(rope);
    for (Value view = rope.getInput(); view.getDefiningOp<ReshapeOp>();
         view = view.getDefiningOp()->getOperand(0)) {
      chain.insert(view.getDefiningOp());
    }
  }
  // q: reshapes, a zero pad of the last dim, reshapes to [1, B, Hq, q_width].
  auto pad = dyn_cast_or_null<PadOp>(
      getSingleNonDeallocUser(alongReshapes(qRope.getResult(), chain)));
  if (!pad || pad.getValue().convertToFloat() != 0.0f) {
    return mlir::failure();
  }
  ArrayRef<int32_t> padding = pad.getPadding();
  for (size_t i = 0; i + 1 < padding.size(); ++i) {
    if (padding[i] != 0) {
      return mlir::failure();
    }
  }
  chain.insert(pad);
  Value qOut = alongReshapes(pad.getResult(), chain);
  const int64_t qWidth = shapeOf(qOut).back();
  if (shapeOf(pad.getInput()).back() != headDim || qWidth % 32 != 0 ||
      shapeOf(qOut) != ArrayRef<int64_t>{1, tokens, numQHeads, qWidth} ||
      !isBF16Tile(qOut, /*dram=*/false)) {
    return mlir::failure();
  }
  if (alongReshapes(kRope.getResult(), chain) != kHeads) {
    return mlir::failure();
  }
  chain.insert(concat);
  // k and v of each token: the concat itself for one token, a slice of its
  // row otherwise, then reshapes to [1, 1, Hkv, 2 D].
  SmallVector<Value> kvOut(tokens);
  for (Operation *user : concat.getResult().getUsers()) {
    if (isa<DeallocateOp>(user)) {
      continue;
    }
    int64_t token = 0;
    Value row;
    if (auto slice = dyn_cast<SliceStaticOp>(user)) {
      auto attr = [](ArrayAttr array, int i) {
        return mlir::cast<IntegerAttr>(array[i]).getInt();
      };
      token = attr(slice.getBegins(), 0);
      if (attr(slice.getEnds(), 0) != token + 1 ||
          attr(slice.getBegins(), 1) != 0 || attr(slice.getBegins(), 2) != 0 ||
          attr(slice.getEnds(), 1) != numKVHeads ||
          attr(slice.getEnds(), 2) != 2 * headDim ||
          attr(slice.getStep(), 0) != 1 || attr(slice.getStep(), 1) != 1 ||
          attr(slice.getStep(), 2) != 1) {
        return mlir::failure();
      }
      row = slice.getResult();
    } else if (isa<ReshapeOp>(user) && tokens == 1) {
      row = user->getResult(0);
    } else {
      return mlir::failure();
    }
    chain.insert(user);
    if (token < 0 || token >= tokens || kvOut[token]) {
      return mlir::failure();
    }
    Value out = alongReshapes(row, chain);
    if (shapeOf(out) != ArrayRef<int64_t>{1, 1, numKVHeads, 2 * headDim} ||
        !isBF16Tile(out, /*dram=*/false)) {
      return mlir::failure();
    }
    kvOut[token] = out;
  }
  for (Value out : kvOut) {
    if (!out) {
      return mlir::failure();
    }
  }
  // Each value inside the chain feeds only ops of the chain.
  SmallVector<Value> outputs{qOut};
  outputs.append(kvOut.begin(), kvOut.end());
  for (Operation *op : chain) {
    for (Value result : op->getResults()) {
      if (llvm::is_contained(outputs, result)) {
        continue;
      }
      for (Operation *user : result.getUsers()) {
        if (!isa<DeallocateOp>(user) && !chain.contains(user)) {
          return mlir::failure();
        }
      }
    }
  }

  // Build the fused op before the first user of its results, after its
  // operands.
  Operation *first = nullptr;
  for (Value out : outputs) {
    for (Operation *user : out.getUsers()) {
      if (user->getBlock() != concat->getBlock()) {
        return mlir::failure();
      }
      if (!first || user->isBeforeInBlock(first)) {
        first = user;
      }
    }
  }
  if (!first) {
    return mlir::failure();
  }
  for (Value operand : {qkv, qWeight, kWeight, cos, sin}) {
    Operation *def = operand.getDefiningOp();
    if (def && (def->getBlock() != first->getBlock() ||
                !def->isBeforeInBlock(first))) {
      return mlir::failure();
    }
  }
  rewriter.setInsertionPoint(first);
  Value input = qkv;
  if (qkvType.getShape() != ArrayRef<int64_t>{tokens, width}) {
    input = ttir_to_ttnn::utils::generateReshape(
        mlir::cast<TypedValue<RankedTensorType>>(qkv),
        SmallVector<int64_t>{tokens, width}, rewriter,
        ttmlir::utils::appendLocationSuffix(qRope.getLoc(), "_qkv_rows"));
  }
  SmallVector<Type> kvTypes;
  for (Value out : kvOut) {
    kvTypes.push_back(out.getType());
  }
  auto fused = rewriter.create<SplitQKVNormRopeDecodeOp>(
      qRope.getLoc(), qOut.getType(), kvTypes, input, qWeight, kWeight, cos,
      sin, rewriter.getUI32IntegerAttr(numQHeads),
      rewriter.getUI32IntegerAttr(numKVHeads), qNorm.getEpsilonAttr());
  rewriter.replaceAllUsesWith(qOut, fused.getQuery());
  for (int64_t b = 0; b < tokens; ++b) {
    rewriter.replaceAllUsesWith(kvOut[b], fused.getKeyValues()[b]);
  }
  // Erase the chain from its ends: every op whose results are now unused but
  // by deallocations.
  SmallVector<Operation *> pending(chain.begin(), chain.end());
  for (bool erased = true; erased;) {
    erased = false;
    for (Operation *&op : pending) {
      if (!op || llvm::any_of(op->getUsers(), [](Operation *user) {
            return !isa<DeallocateOp>(user);
          })) {
        continue;
      }
      for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
        rewriter.eraseOp(user);
      }
      rewriter.eraseOp(op);
      op = nullptr;
      erased = true;
    }
  }
  return mlir::success();
}

} // namespace mlir::tt::ttnn::fusing

namespace mlir::tt::ttnn {

namespace {

// Writes the k and v of every token where the ops that read them want them:
// the workarounds, after fusing, move each into the paged cache update's L1
// shard.
class WriteKeyValuesInPlace
    : public mlir::OpRewritePattern<SplitQKVNormRopeDecodeOp> {
public:
  using OpRewritePattern<SplitQKVNormRopeDecodeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SplitQKVNormRopeDecodeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // One memory config for all of them, as the op takes.
    SmallVector<ToMemoryConfigOp> moves;
    for (Value keyValue : op.getKeyValues()) {
      auto move = keyValue.hasOneUse()
                      ? dyn_cast<ToMemoryConfigOp>(*keyValue.getUsers().begin())
                      : nullptr;
      if (!move || move.getType() == keyValue.getType() ||
          (!moves.empty() && move.getType() != moves.front().getType())) {
        return mlir::failure();
      }
      auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
          move.getType().getEncoding());
      if (!layout || !layout.isTiled() ||
          layout.getDataType() != ttcore::DataType::BFloat16) {
        return mlir::failure();
      }
      moves.push_back(move);
    }
    if (moves.empty()) {
      return mlir::failure();
    }
    rewriter.modifyOpInPlace(op, [&] {
      for (Value keyValue : op.getKeyValues()) {
        keyValue.setType(moves.front().getType());
      }
    });
    for (auto [keyValue, move] : llvm::zip(op.getKeyValues(), moves)) {
      rewriter.replaceOp(move, keyValue);
    }
    return mlir::success();
  }
};

} // namespace

void SplitQKVNormRopeDecodeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<WriteKeyValuesInPlace>(context);
}

} // namespace mlir::tt::ttnn
