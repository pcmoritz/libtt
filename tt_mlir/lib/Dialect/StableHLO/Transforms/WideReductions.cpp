// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Reductions over the rows of a tall, thin tensor become reductions over the
// columns of its transpose.
//
// A [H, W] tensor with W within one tile column has H / 32 tile rows and a
// single tile column. Reducing it over H serializes on that column (one core
// walks every tile row), and even the elementwise ops that feed the reduction
// run about three times slower on that layout than on [W, H]. JAX code that
// reduces repeatedly, such as the binary searches behind top-k and top-p
// masking, swaps the axes up front precisely to put the reduction on the
// leading axis, which is the right call on TPU and the wrong one here.
//
// The rewrite transposes the reduction's operand chain back: the chain of
// elementwise ops, broadcasts and constants feeding the reduce is rebuilt on
// [W, H], and the reduce goes over the last dimension. It applies only when
// every tensor at the leaves of that chain is itself a transpose, which the
// rebuilt chain then consumes directly, so no transpose is ever added, only
// removed. A leaf that is a loop-carried value the loop never changes is first
// replaced by the value the loop was entered with, which exposes the transpose
// behind it.
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::tt {
namespace {

namespace shlo = mlir::stablehlo;

// A tensor whose last dimension fits one tile column and whose first is long
// enough for the column to matter.
constexpr int64_t kThinWidth = 32;
constexpr int64_t kTallHeight = 1024;
// How far up the operand chain the rebuild follows elementwise ops.
constexpr unsigned kMaxChainDepth = 12;

bool isTallThin(RankedTensorType type) {
  return type && type.getRank() == 2 && type.hasStaticShape() &&
         type.getDimSize(1) <= kThinWidth && type.getDimSize(0) >= kTallHeight;
}

RankedTensorType transposedType(RankedTensorType type) {
  return RankedTensorType::get({type.getDimSize(1), type.getDimSize(0)},
                               type.getElementType());
}

bool isSwap(shlo::TransposeOp transpose) {
  auto permutation = transpose.getPermutation();
  return permutation.size() == 2 && permutation[0] == 1 && permutation[1] == 0;
}

// Whether `value` can be rebuilt on the transposed shape without adding a
// transpose: it is a transpose (consumed directly), a broadcast of something
// no larger than rank 2 (re-broadcast), a splat constant, a replicated
// sharding constraint on such a value (JAX leaves one after most ops under a
// shard_map; the constraint is kept, since replication is the same on either
// shape), or an elementwise op over such values.
bool isRebuildable(Value value, unsigned depth) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 2) {
    return false;
  }
  Operation *op = value.getDefiningOp();
  if (!op) {
    return false;
  }
  if (auto transpose = dyn_cast<shlo::TransposeOp>(op)) {
    return isSwap(transpose);
  }
  if (auto broadcast = dyn_cast<shlo::BroadcastInDimOp>(op)) {
    return cast<RankedTensorType>(broadcast.getOperand().getType()).getRank() <=
           2;
  }
  if (auto constant = dyn_cast<shlo::ConstantOp>(op)) {
    return constant.getValue().isSplat();
  }
  if (auto constraint = dyn_cast<sdy::ShardingConstraintOp>(op)) {
    return constraint.getSharding().isFullyReplicated() &&
           isRebuildable(constraint.getInput(), depth);
  }
  if (depth == 0) {
    return false;
  }
  bool elementwise = op->hasTrait<OpTrait::Elementwise>() ||
                     isa<shlo::CompareOp, shlo::SelectOp>(op);
  if (!elementwise || op->getNumResults() != 1 || op->getNumRegions() != 0) {
    return false;
  }
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    return cast<RankedTensorType>(operand.getType()).getShape() ==
               type.getShape() &&
           isRebuildable(operand, depth - 1);
  });
}

// Builds `value` on the transposed shape; see isRebuildable for the cases.
Value rebuildTransposed(PatternRewriter &rewriter, Value value) {
  Operation *op = value.getDefiningOp();
  auto type = cast<RankedTensorType>(value.getType());
  if (auto transpose = dyn_cast<shlo::TransposeOp>(op)) {
    return transpose.getOperand();
  }
  if (auto broadcast = dyn_cast<shlo::BroadcastInDimOp>(op)) {
    SmallVector<int64_t> dims;
    for (int64_t dim : broadcast.getBroadcastDimensions()) {
      dims.push_back(1 - dim);
    }
    return rewriter.create<shlo::BroadcastInDimOp>(
        op->getLoc(), transposedType(type), broadcast.getOperand(),
        rewriter.getDenseI64ArrayAttr(dims));
  }
  if (auto constant = dyn_cast<shlo::ConstantOp>(op)) {
    auto splat = cast<SplatElementsAttr>(constant.getValue());
    return rewriter.create<shlo::ConstantOp>(
        op->getLoc(),
        SplatElementsAttr::get(transposedType(type),
                               splat.getSplatValue<Attribute>()));
  }
  if (auto constraint = dyn_cast<sdy::ShardingConstraintOp>(op)) {
    return rewriter.create<sdy::ShardingConstraintOp>(
        op->getLoc(), rebuildTransposed(rewriter, constraint.getInput()),
        constraint.getSharding());
  }
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    operands.push_back(rebuildTransposed(rewriter, operand));
  }
  Operation *rebuilt = rewriter.clone(*op);
  rebuilt->setOperands(operands);
  rebuilt->getResult(0).setType(transposedType(type));
  return rebuilt->getResult(0);
}

// reduce(dims = [0]) of a tall, thin chain -> reduce(dims = [1]) of the chain
// rebuilt on the transposed shape.
class TallReductionToWide : public OpRewritePattern<shlo::ReduceOp> {
public:
  using OpRewritePattern<shlo::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shlo::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getInputs().size() != 1 || reduceOp.getDimensions().size() != 1 ||
        reduceOp.getDimensions()[0] != 0) {
      return failure();
    }
    Value input = reduceOp.getInputs().front();
    if (!isTallThin(dyn_cast<RankedTensorType>(input.getType())) ||
        !isRebuildable(input, kMaxChainDepth)) {
      return failure();
    }
    Value transposed = rebuildTransposed(rewriter, input);
    auto rebuilt = rewriter.create<shlo::ReduceOp>(
        reduceOp.getLoc(), reduceOp.getResultTypes(), ValueRange{transposed},
        reduceOp.getInitValues(), rewriter.getDenseI64ArrayAttr({1}));
    rewriter.inlineRegionBefore(reduceOp.getBody(), rebuilt.getBody(),
                                rebuilt.getBody().end());
    rewriter.replaceOp(reduceOp, rebuilt.getResults());
    return success();
  }
};

// A loop operand that the body returns unchanged is the same value on every
// iteration. Uses of its block arguments become uses of the value the loop was
// entered with, so what feeds it (for example a transpose) is visible inside
// the body.
class ReplaceInvariantCarries : public OpRewritePattern<shlo::WhileOp> {
public:
  using OpRewritePattern<shlo::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shlo::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    Block &body = whileOp.getBody().front();
    Block &cond = whileOp.getCond().front();
    auto returnOp = cast<shlo::ReturnOp>(body.getTerminator());
    bool changed = false;
    for (auto [index, init] : llvm::enumerate(whileOp.getOperands())) {
      BlockArgument bodyArg = body.getArgument(index);
      BlockArgument condArg = cond.getArgument(index);
      if (returnOp.getOperand(index) != bodyArg ||
          (bodyArg.use_empty() && condArg.use_empty())) {
        continue;
      }
      rewriter.replaceAllUsesExcept(bodyArg, init, returnOp);
      rewriter.replaceAllUsesWith(condArg, init);
      changed = true;
    }
    return success(changed);
  }
};

} // namespace

void populateWideReductionPatterns(RewritePatternSet &patterns) {
  patterns.add<TallReductionToWide, ReplaceInvariantCarries>(
      patterns.getContext());
}

} // namespace mlir::tt
