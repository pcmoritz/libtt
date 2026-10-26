// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// A few vocabulary-sized rows, computed as [..., H, W / 32, 32].
//
// TTNN stores a tensor in 32 x 32 tiles of its last two dimensions. A decode
// batch of logits, [2, 151936], fills 2 of each tile's 32 rows, so every pass
// over it moves 16 times its data: an FP32 elementwise op takes about 110 us,
// against 11 us on the same data as [2, 4748, 32]. Sampling at temperature > 0
// makes some 25 such passes.
//
// Each elementwise op, broadcast, reduction of whole rows and random-bit draw
// on such a value becomes the op on the dense shape between a pack of its
// operands and an unpack of its result (libtt's generator draws each element
// from its row-major index, so the bits stay the same). StableHLO's
// simplification cancels a pack of an unpack, and an unpack that every branch
// of a conditional returns moves after it, so reshapes remain only where the
// dense computation meets other ops. The pass runs right before StableHLO
// becomes TTIR, while conditionals are still regions, and a function keeps the
// rewrite only when the passes it saves outweigh those reshapes. Loop bodies
// keep their shapes. LIBTT_DENSE_ROWS=0 turns the pass off.

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace mlir::tt {
namespace {

constexpr int64_t kTileWidth = 32;
// Vocabulary-sized rows only: hidden states and MLP activations keep their
// shapes, which the TTIR fusions (norms, SwiGLU) match.
constexpr int64_t kMinWidth = 32768;
// More rows fill enough of each tile.
constexpr int64_t kMaxRows = 16;
// An unpack writes every padded tile of the wide shape again.
constexpr int64_t kUnpackCost = 3;

// The dense type of a [..., H, W] value, or null.
RankedTensorType getDenseType(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() || tensor.getRank() == 0 ||
      tensor.getEncoding() || !tensor.getElementType().isIntOrFloat()) {
    return {};
  }
  ArrayRef<int64_t> shape = tensor.getShape();
  int64_t width = shape.back();
  int64_t rows = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
  if (width < kMinWidth || width % kTileWidth != 0 || rows > kMaxRows) {
    return {};
  }
  SmallVector<int64_t> dense(shape.drop_back());
  dense.push_back(width / kTileWidth);
  dense.push_back(kTileWidth);
  return RankedTensorType::get(dense, tensor.getElementType());
}

bool isCandidate(Value value) {
  return static_cast<bool>(getDenseType(value.getType()));
}

// Whether `op` can compute on dense rows: an elementwise op, select or clamp
// whose other operands are scalars, a broadcast that adds no rows to the last
// dimension, a reduction of whole rows, or random bits.
bool canBeDense(Operation *op) {
  if (op->getParentOfType<stablehlo::WhileOp>() ||
      llvm::any_of(op->getAttrs(), [](NamedAttribute attribute) {
        return attribute.getName().strref().contains("sharding");
      })) {
    return false;
  }
  if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op)) {
    Value input = reduce.getInputs()[0];
    auto rank = cast<RankedTensorType>(input.getType()).getRank();
    return reduce.getInputs().size() == 1 && isCandidate(input) &&
           llvm::is_contained(reduce.getDimensions(), rank - 1);
  }
  // The uint64 state conversion has lowered random bits to TTIR already.
  if (auto rng = dyn_cast<ttir::RngBitGeneratorOp>(op)) {
    return isCandidate(rng.getOutput());
  }
  if (op->getNumResults() != 1 || !isCandidate(op->getResult(0))) {
    return false;
  }
  auto type = cast<RankedTensorType>(op->getResult(0).getType());
  if (auto broadcast = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
    // Along the rows: the operand dimension that becomes the last one is 1.
    auto operand = cast<RankedTensorType>(broadcast.getOperand().getType());
    return llvm::all_of(
        llvm::enumerate(broadcast.getBroadcastDimensions()), [&](auto dim) {
          return dim.value() != type.getRank() - 1 ||
                 operand.getDimSize(dim.index()) == 1;
        });
  }
  bool elementwise = isa<stablehlo::SelectOp, stablehlo::ClampOp,
                         stablehlo::BitcastConvertOp>(op) ||
                     (op->hasTrait<OpTrait::Elementwise>() &&
                      isa<stablehlo::StablehloDialect>(op->getDialect()));
  return elementwise && op->getNumRegions() == 0 &&
         llvm::all_of(op->getOperands(), [&](Value operand) {
           auto operandType = cast<RankedTensorType>(operand.getType());
           return operandType.getRank() == 0 ||
                  (isCandidate(operand) &&
                   operandType.getShape() == type.getShape());
         });
}

Value reshape(PatternRewriter &rewriter, Value value, Type type) {
  return rewriter.create<stablehlo::ReshapeOp>(value.getLoc(), type, value);
}

// op(x, ...) -> unpack(op(pack(x), ...)).
class DenseOpPattern : public RewritePattern {
public:
  DenseOpPattern(MLIRContext *context, int64_t &passes)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        passes(passes) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!canBeDense(op)) {
      return failure();
    }
    auto pack = [&](Value value) {
      RankedTensorType type = getDenseType(value.getType());
      return type ? reshape(rewriter, value, type) : value;
    };
    Operation *dense;
    if (auto reduce = dyn_cast<stablehlo::ReduceOp>(op)) {
      int64_t rank = cast<RankedTensorType>(reduce.getInputs()[0].getType())
                         .getRank();
      SmallVector<int64_t> dims(reduce.getDimensions());
      dims.push_back(rank);
      auto newReduce = rewriter.create<stablehlo::ReduceOp>(
          reduce.getLoc(), reduce.getResultTypes(),
          ValueRange{pack(reduce.getInputs()[0])}, reduce.getInitValues(),
          rewriter.getDenseI64ArrayAttr(dims));
      rewriter.cloneRegionBefore(reduce.getBody(), newReduce.getBody(),
                                 newReduce.getBody().end());
      dense = newReduce;
    } else {
      // Elementwise ops, broadcasts and random bits keep their attributes.
      IRMapping mapping;
      for (Value operand : op->getOperands()) {
        mapping.map(operand, pack(operand));
      }
      dense = rewriter.clone(*op, mapping);
      for (OpResult result : dense->getResults()) {
        if (RankedTensorType type = getDenseType(result.getType())) {
          result.setType(type);
        }
      }
    }
    if (!isa<stablehlo::BroadcastInDimOp>(op)) {
      ++passes;
    }
    SmallVector<Value> results;
    for (auto [result, denseResult] :
         llvm::zip_equal(op->getResults(), dense->getResults())) {
      results.push_back(result.getType() == denseResult.getType()
                            ? denseResult
                            : reshape(rewriter, denseResult, result.getType()));
    }
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  int64_t &passes;
};

// A conditional's result that every branch returns as an unpack is returned
// dense and unpacked after the conditional.
class DenseCaseResultPattern : public OpRewritePattern<stablehlo::CaseOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CaseOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<stablehlo::WhileOp>()) {
      return failure();
    }
    auto unpacked = [&](Region &branch, unsigned i) -> Value {
      auto unpack = branch.front()
                        .getTerminator()
                        ->getOperand(i)
                        .getDefiningOp<stablehlo::ReshapeOp>();
      return unpack && unpack.getOperand().getType() ==
                           getDenseType(op.getResult(i).getType())
                 ? unpack.getOperand()
                 : Value();
    };
    SmallVector<Type> types(op.getResultTypes());
    SmallVector<unsigned> dense;
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      if (isCandidate(op.getResult(i)) &&
          llvm::all_of(op.getBranches(),
                       [&](Region &branch) { return unpacked(branch, i); })) {
        types[i] = getDenseType(types[i]);
        dense.push_back(i);
      }
    }
    if (dense.empty()) {
      return failure();
    }
    auto newCase = rewriter.create<stablehlo::CaseOp>(
        op.getLoc(), types, op.getIndex(), op.getBranches().size());
    for (auto [branch, newBranch] :
         llvm::zip_equal(op.getBranches(), newCase.getBranches())) {
      Operation *terminator = branch.front().getTerminator();
      for (unsigned i : dense) {
        Value value = unpacked(branch, i);
        rewriter.modifyOpInPlace(terminator,
                                 [&] { terminator->setOperand(i, value); });
      }
      rewriter.inlineRegionBefore(branch, newBranch, newBranch.end());
    }
    SmallVector<Value> results(newCase.getResults());
    for (unsigned i : dense) {
      results[i] = reshape(rewriter, results[i], op.getResult(i).getType());
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Whether `value` is computed from constants and iotas alone, by reshapes and
// broadcasts: the program computes it once, when it loads.
bool isConstantLike(Value value) {
  Operation *op = value.getDefiningOp();
  while (isa_and_nonnull<stablehlo::ReshapeOp, stablehlo::BroadcastInDimOp>(op)) {
    op = op->getOperand(0).getDefiningOp();
  }
  return isa_and_nonnull<stablehlo::ConstantOp, stablehlo::IotaOp>(op);
}

// The reshapes between the two shapes, in passes over the rows.
int64_t getReshapeCost(func::FuncOp function) {
  int64_t cost = 0;
  function.walk([&](stablehlo::ReshapeOp op) {
    if (isConstantLike(op.getOperand())) {
      return;
    }
    if (isCandidate(op.getOperand()) &&
        op.getType() == getDenseType(op.getOperand().getType())) {
      cost += 1;
    } else if (isCandidate(op.getResult()) &&
               op.getOperand().getType() == getDenseType(op.getType())) {
      cost += kUnpackCost;
    }
  });
  return cost;
}

struct DenseWideRowsPass
    : public PassWrapper<DenseWideRowsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DenseWideRowsPass)

  StringRef getArgument() const final { return "libtt-dense-wide-rows"; }
  StringRef getDescription() const final {
    return "Compute a few vocabulary-sized rows as [..., H, W / 32, 32]";
  }

  void runOnOperation() override {
    if (const char *setting = std::getenv("LIBTT_DENSE_ROWS");
        setting && std::strcmp(setting, "0") == 0) {
      return;
    }
    for (func::FuncOp function :
         llvm::to_vector(getOperation().getOps<func::FuncOp>())) {
      // Rewrite a copy, and keep it if it is worth its reshapes.
      func::FuncOp copy = function.clone();
      int64_t passes = 0;
      RewritePatternSet patterns(&getContext());
      patterns.add<DenseOpPattern>(&getContext(), passes);
      patterns.add<DenseCaseResultPattern>(&getContext());
      // Among them, reshape(reshape(x)) -> x. The IR is already simplified.
      stablehlo::populateStablehloCanonicalizationPatterns(&getContext(),
                                                           &patterns);
      (void)applyPatternsGreedily(copy.getBody(), std::move(patterns));
      IRRewriter rewriter(&getContext());
      DominanceInfo dominance(copy);
      eliminateCommonSubExpressions(rewriter, dominance, copy);
      if (passes > getReshapeCost(copy) - getReshapeCost(function)) {
        function.getBody().takeBody(copy.getBody());
      }
      copy.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDenseWideRowsPass() {
  return std::make_unique<DenseWideRowsPass>();
}

} // namespace mlir::tt
