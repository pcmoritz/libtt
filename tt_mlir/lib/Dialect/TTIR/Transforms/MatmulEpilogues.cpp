// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// ttir-matmul-epilogues: matmuls with the epilogues their TTNN kernels apply as
// they write the product.
//
// - The dequantization of SGLang-TorchTPU's FP8 linears: the FP32 product
//   times an [M, 1] row scale and a [1, N] (or one-element) column scale, and a
//   narrowing to BF16 (ttnn.scaled_matmul).
// - The decode MLP's SwiGLU of its merged projection's halves (ttnn.matmul or
//   ttnn.scaled_matmul, ttnn.fused_swiglu): JAX models' up * silu(gate) of
//   [up | gate] with BFP8 weights, and PyTorch's SiluAndMul of [gate | up],
//   bit for bit (ttnn.exact_swiglu).
// - An activation of a matmul or linear (silu, gelu, sigmoid), unless a binary
//   op takes it as an input activation (TTNNFusing's
//   TTNNBinaryOpInputsActivation, under the same options).
//
// Each becomes one ttir.fused_matmul, which lowers to the TTNN op. This runs
// after the TTIR fusions and ttir-propagate-weight-dtype (the JAX SwiGLU needs
// BFP8 weights), so that later passes that change shapes see one op where the
// kernels take one.

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {

namespace {

constexpr int64_t kTile = 32;

RankedTensorType typeOf(Value value) {
  return mlir::cast<RankedTensorType>(value.getType());
}

Type elementTypeOf(Value value) { return typeOf(value).getElementType(); }

bool isFloat8(Value value) {
  return mlir::isa<Float8E4M3FNType, Float8E5M2Type>(elementTypeOf(value));
}

// The defining op of type T of one of `op`'s operands, and the other operand.
template <typename T>
T factorOf(MultiplyOp op, Value &other) {
  if (auto def = op.getLhs().getDefiningOp<T>()) {
    other = op.getRhs();
    return def;
  }
  if (auto def = op.getRhs().getDefiningOp<T>()) {
    other = op.getLhs();
    return def;
  }
  return T();
}

// Whether `slice` takes columns [begin, begin + width) of all `rows` rows.
bool isColumns(SliceStaticOp slice, int64_t rows, int64_t begin,
               int64_t width) {
  auto ints = [](ArrayAttr array) {
    return llvm::to_vector(llvm::map_range(array, [](Attribute a) {
      return mlir::cast<IntegerAttr>(a).getInt();
    }));
  };
  return ints(slice.getBegins()) == SmallVector<int64_t>{0, begin} &&
         ints(slice.getEnds()) == SmallVector<int64_t>{rows, begin + width} &&
         ints(slice.getStep()) == SmallVector<int64_t>{1, 1};
}

// Whether `op` may write memory: an in-place update, say, of a matmul's
// operand.
bool mayWrite(Operation *op) {
  if (isMemoryEffectFree(op)) {
    return false;
  }
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  return !effects || effects.hasEffect<MemoryEffects::Write>();
}

// The fused matmul of `from`, the matmul or linear it replaces, made where
// `from` is (or after `insertAfter`): an in-place write after the matmul, such
// as a cache update, must not change what it reads.
FusedMatmulOp createFused(PatternRewriter &rewriter, Operation *from,
                          Type type, Value a, Value b, Value bias,
                          Value rowScale, Value colScale, bool transposeA,
                          bool transposeB, StringAttr activation,
                          bool swiglu = false, bool exact = false,
                          Operation *insertAfter = nullptr) {
  if (insertAfter) {
    rewriter.setInsertionPointAfter(insertAfter);
  } else {
    rewriter.setInsertionPoint(from);
  }
  auto fused = rewriter.create<FusedMatmulOp>(
      from->getLoc(), type, a, b, bias, rowScale, colScale,
      rewriter.getBoolAttr(transposeA), rewriter.getBoolAttr(transposeB),
      activation, swiglu ? rewriter.getUnitAttr() : nullptr,
      exact ? rewriter.getUnitAttr() : nullptr);
  if (auto weightDtype = from->getAttr("ttcore.weight_dtype")) {
    fused->setAttr("ttcore.weight_dtype", weightDtype);
  }
  return fused;
}

// (matmul(a, b) * row_scale) * col_scale in FP32, a and b float8, on one tile
// row of activations: a scaled matmul, which also takes a narrowing of the
// result to BF16 (its epilogue rounds as binary_ng's FP32 multiply does).
class OutputScales : public OpRewritePattern<MultiplyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiplyOp outer,
                                PatternRewriter &rewriter) const override {
    Value colScale, rowScale;
    auto inner = factorOf<MultiplyOp>(outer, colScale);
    if (!inner || !inner->hasOneUse()) {
      return failure();
    }
    auto matmul = factorOf<MatmulOp>(inner, rowScale);
    if (!matmul || !matmul->hasOneUse() || matmul.getTransposeA() ||
        matmul->hasAttr("ttcore.fp32_exact") || !isFloat8(matmul.getA()) ||
        !isFloat8(matmul.getB())) {
      return failure();
    }
    RankedTensorType product = matmul.getType();
    if (product.getRank() != 2 || !product.getElementType().isF32() ||
        !elementTypeOf(inner.getResult()).isF32() ||
        !elementTypeOf(outer.getResult()).isF32() ||
        !elementTypeOf(rowScale).isF32() || !elementTypeOf(colScale).isF32()) {
      return failure();
    }
    const int64_t rows = product.getDimSize(0);
    const int64_t cols = product.getDimSize(1);
    // The official checkpoints' per-channel scales, [N, 1], reshaped to
    // [1, N]: the epilogue takes the column as it is.
    auto colReshape = colScale.getDefiningOp<ReshapeOp>();
    if (colReshape &&
        typeOf(colReshape.getInput()).getShape() ==
            ArrayRef<int64_t>{cols, 1}) {
      colScale = colReshape.getInput();
    } else {
      colReshape = nullptr;
    }
    ArrayRef<int64_t> colShape = typeOf(colScale).getShape();
    if (rows > kTile || typeOf(rowScale).getShape() != ArrayRef<int64_t>{rows, 1} ||
        (!colReshape && typeOf(colScale).getNumElements() != 1 &&
         colShape != ArrayRef<int64_t>{1, cols})) {
      return failure();
    }
    // The scaled matmul goes where the matmul is, or after a scale made
    // later, with nothing between that may write memory.
    Operation *after = nullptr;
    for (Value scale : {rowScale, colScale}) {
      Operation *def = scale.getDefiningOp();
      if (def && def->getBlock() != matmul->getBlock()) {
        return failure();
      }
      if (def && matmul->isBeforeInBlock(def) &&
          (!after || after->isBeforeInBlock(def))) {
        after = def;
      }
    }
    for (Operation *op = matmul->getNextNode(); after && op != after->getNextNode();
         op = op->getNextNode()) {
      if (mayWrite(op)) {
        return failure();
      }
    }
    Operation *last = outer;
    if (outer->hasOneUse()) {
      auto cast = dyn_cast<TypecastOp>(*outer->getUsers().begin());
      if (cast && elementTypeOf(cast.getResult()).isBF16()) {
        last = cast;
      }
    }
    auto fused = createFused(
        rewriter, matmul, last->getResult(0).getType(), matmul.getA(),
        matmul.getB(), Value(), rowScale, colScale, false,
        matmul.getTransposeB(), nullptr, false, false, after);
    rewriter.replaceOp(last, fused.getResult());
    if (last != outer) {
      rewriter.eraseOp(outer);
    }
    rewriter.eraseOp(inner);
    rewriter.eraseOp(matmul);
    if (colReshape && colReshape->use_empty()) {
      rewriter.eraseOp(colReshape);
    }
    return success();
  }
};

// up * silu(gate) of a [M <= 32, 2 H] matmul's halves [up | gate], with BF16
// activations and output and BFP8 weights, K and H whole tiles: JAX models'
// decode MLP.
class SwiGLU : public OpRewritePattern<MultiplyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiplyOp multiply,
                                PatternRewriter &rewriter) const override {
    // Exactly one operand is a silu.
    auto lhsSilu = multiply.getLhs().getDefiningOp<SiluOp>();
    auto rhsSilu = multiply.getRhs().getDefiningOp<SiluOp>();
    if (!lhsSilu == !rhsSilu) {
      return failure();
    }
    SiluOp silu = lhsSilu ? lhsSilu : rhsSilu;
    Value other = lhsSilu ? multiply.getRhs() : multiply.getLhs();
    auto gate = silu.getInput().getDefiningOp<SliceStaticOp>();
    auto up = other.getDefiningOp<SliceStaticOp>();
    if (!gate || !up || !silu->hasOneUse() || !gate->hasOneUse() ||
        !up->hasOneUse()) {
      return failure();
    }
    auto matmul = gate.getInput().getDefiningOp<MatmulOp>();
    if (!matmul || up.getInput() != matmul.getResult() ||
        !matmul.getResult().hasNUses(2) || matmul.getTransposeA() ||
        matmul.getTransposeB()) {
      return failure();
    }
    auto weightDtype =
        matmul->getAttrOfType<StringAttr>("ttcore.weight_dtype");
    ArrayRef<int64_t> shape = matmul.getType().getShape();
    const int64_t k = typeOf(matmul.getA()).getShape().back();
    const int64_t half = typeOf(multiply.getResult()).getShape().back();
    if (!weightDtype || weightDtype.getValue() != "bfp_bf8" ||
        !elementTypeOf(matmul.getA()).isBF16() ||
        !elementTypeOf(multiply.getResult()).isBF16() || shape.size() != 2 ||
        shape[0] > kTile || k % kTile != 0 || half % kTile != 0 ||
        shape[1] != 2 * half || !isColumns(up, shape[0], 0, half) ||
        !isColumns(gate, shape[0], half, half)) {
      return failure();
    }
    auto fused = createFused(rewriter, matmul, multiply.getType(),
                             matmul.getA(), matmul.getB(), Value(), Value(),
                             Value(), false, false, nullptr, /*swiglu=*/true);
    rewriter.replaceOp(multiply, fused.getResult());
    rewriter.eraseOp(silu);
    rewriter.eraseOp(gate);
    rewriter.eraseOp(up);
    rewriter.eraseOp(matmul);
    return success();
  }
};

// PyTorch's SiluAndMul after a merged [gate | up] projection, as
// SGLang-TorchTPU's decode MLP runs it on one tile row: silu of the gate half
// in FP32, cast back to BF16, times the up half. The BF16 projection (a matmul
// or a scaled matmul) takes it as its epilogue, which reproduces every
// rounding of those ops.
class ExactSwiGLU : public OpRewritePattern<MultiplyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MultiplyOp multiply,
                                PatternRewriter &rewriter) const override {
    // typecast(silu(typecast(slice))) to BF16 by way of an FP32 silu, each op
    // used once; the gate slice last.
    auto siluOfSlice = [](Value v) -> SmallVector<Operation *> {
      auto narrow = v.getDefiningOp<TypecastOp>();
      auto silu = narrow ? narrow.getInput().getDefiningOp<SiluOp>() : nullptr;
      auto widen = silu ? silu.getInput().getDefiningOp<TypecastOp>() : nullptr;
      auto gate =
          widen ? widen.getInput().getDefiningOp<SliceStaticOp>() : nullptr;
      SmallVector<Operation *> chain = {narrow, silu, widen, gate};
      if (!gate || !elementTypeOf(silu.getResult()).isF32() ||
          !elementTypeOf(narrow.getResult()).isBF16() ||
          !llvm::all_of(chain, [](Operation *op) { return op->hasOneUse(); })) {
        return {};
      }
      return chain;
    };
    SmallVector<Operation *> chain = siluOfSlice(multiply.getLhs());
    Value other = multiply.getRhs();
    if (chain.empty()) {
      chain = siluOfSlice(multiply.getRhs());
      other = multiply.getLhs();
    }
    if (chain.empty()) {
      return failure();
    }
    auto gate = mlir::cast<SliceStaticOp>(chain.back());
    auto up = other.getDefiningOp<SliceStaticOp>();
    Operation *projection = gate.getInput().getDefiningOp();
    if (!up || !projection || up.getInput() != gate.getInput() ||
        !up->hasOneUse() || !gate.getInput().hasNUses(2)) {
      return failure();
    }
    // A matmul, or a scaled matmul without another epilogue.
    Value a, b, rowScale, colScale;
    bool transposeB = false;
    if (auto matmul = dyn_cast<MatmulOp>(projection)) {
      if (matmul.getTransposeA() || matmul->hasAttr("ttcore.fp32_exact")) {
        return failure();
      }
      a = matmul.getA();
      b = matmul.getB();
      transposeB = matmul.getTransposeB();
    } else if (auto scaled = dyn_cast<FusedMatmulOp>(projection)) {
      if (!scaled.getRowScale() || scaled.getBias() ||
          scaled.getTransposeA() || scaled.getActivation() ||
          scaled.getFusedSwiglu()) {
        return failure();
      }
      a = scaled.getA();
      b = scaled.getB();
      rowScale = scaled.getRowScale();
      colScale = scaled.getColScale();
      transposeB = scaled.getTransposeB();
    } else {
      return failure();
    }
    // A BF16 [M <= 32, 2 H] product, K and the halves whole tiles.
    RankedTensorType type = typeOf(projection->getResult(0));
    ArrayRef<int64_t> shape = type.getShape();
    const int64_t half = shape.back() / 2;
    const int64_t k = typeOf(a).getShape().back();
    if (shape.size() != 2 || shape[1] != 2 * half ||
        !type.getElementType().isBF16() ||
        !elementTypeOf(multiply.getResult()).isBF16() || shape[0] > kTile ||
        half % kTile != 0 || k % kTile != 0 ||
        !isColumns(gate, shape[0], 0, half) ||
        !isColumns(up, shape[0], half, half)) {
      return failure();
    }
    auto fused = createFused(rewriter, projection, multiply.getType(), a, b,
                             Value(), rowScale, colScale, false, transposeB,
                             nullptr, /*swiglu=*/true, /*exact=*/true);
    rewriter.replaceOp(multiply, fused.getResult());
    for (Operation *op : chain) {
      rewriter.eraseOp(op);
    }
    rewriter.eraseOp(up);
    rewriter.eraseOp(projection);
    return success();
  }
};

// An activation of a matmul or linear's only result, unless a binary op takes
// it as an input activation (claimActivations: TTNNFusing does with
// enable-eltwise-activation-fusion, or DRAM-sharded matmuls, as a binary op
// computes it as it unpacks the operand).
template <typename ActivationOp>
class Activation : public OpRewritePattern<ActivationOp> {
public:
  Activation(MLIRContext *context, bool claimActivations)
      : OpRewritePattern<ActivationOp>(context),
        claimActivations(claimActivations) {}

  LogicalResult matchAndRewrite(ActivationOp activation,
                                PatternRewriter &rewriter) const override {
    Value input = activation.getInput();
    Operation *producer = input.getDefiningOp();
    if (!producer || !isa<MatmulOp, LinearOp>(producer) ||
        !input.hasOneUse() || producer->hasAttr("ttcore.fp32_exact") ||
        (claimActivations && takenByBinaryOp(activation))) {
      return failure();
    }
    StringRef name = ActivationOp::getOperationName().rsplit('.').second;
    auto fuse = [&](auto op, Value bias) {
      return createFused(rewriter, op, activation.getType(), op.getA(),
                         op.getB(), bias, Value(), Value(), op.getTransposeA(),
                         op.getTransposeB(), rewriter.getStringAttr(name));
    };
    auto linear = dyn_cast<LinearOp>(producer);
    FusedMatmulOp fused = linear ? fuse(linear, linear.getBias())
                                 : fuse(cast<MatmulOp>(producer), Value());
    rewriter.replaceOp(activation, fused.getResult());
    rewriter.eraseOp(producer);
    return success();
  }

private:
  // Whether the activation's only use is a binary op with BF16 or FP32
  // operands and result, which takes it as an input activation.
  static bool takenByBinaryOp(ActivationOp activation) {
    if (!activation->hasOneUse()) {
      return false;
    }
    Operation *user = *activation->getUsers().begin();
    auto fusible = [](Value v) {
      Type type = elementTypeOf(v);
      return type.isBF16() || type.isF32();
    };
    return isa<ElementwiseBinary>(user) && !isa<RemainderOp>(user) &&
           fusible(activation.getInput()) &&
           fusible(activation.getResult()) &&
           llvm::all_of(user->getOperands(), fusible) &&
           llvm::all_of(user->getResults(), fusible);
  }

  bool claimActivations;
};

class TTIRMatmulEpiloguesPass
    : public PassWrapper<TTIRMatmulEpiloguesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRMatmulEpiloguesPass)

  explicit TTIRMatmulEpiloguesPass(bool claimActivations = false)
      : claimActivations(claimActivations) {}

  StringRef getArgument() const final { return "ttir-matmul-epilogues"; }
  StringRef getDescription() const final {
    return "Fuse matmuls with the epilogues their TTNN kernels apply";
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<OutputScales, SwiGLU, ExactSwiGLU>(&getContext());
    patterns.add<Activation<SiluOp>, Activation<GeluOp>, Activation<SigmoidOp>>(
        &getContext(), claimActivations);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  bool claimActivations;
};

} // namespace

std::unique_ptr<Pass> createTTIRMatmulEpiloguesPass(bool claimActivations) {
  return std::make_unique<TTIRMatmulEpiloguesPass>(claimActivations);
}

} // namespace mlir::tt::ttir
