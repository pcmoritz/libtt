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
//
// [W, H] still fills only W of each tile's 32 rows. When H splits into chunks
// of C columns (C a multiple of 32), the chain is rebuilt on [W, H / C, C]
// instead, the same row-major data in full tiles, and the reduce goes over the
// last two dimensions. Each transpose at a leaf becomes a reshape of its
// operand where the transpose was, so a loop-invariant leaf is reshaped once
// before the loop. C is the widest such chunk up to about sqrt(H): TTNN
// reduces the two dimensions in turn, and each stage walks one of them on few
// cores. For [2, 151936] (C = 128) a search step's elementwise ops take about
// 13 us each instead of 105, and its reduction about 26 us.
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
// Dense chunks are multiples of a tile's width.
constexpr int64_t kTileWidth = 32;

bool isTallThin(RankedTensorType type) {
  return type && type.getRank() == 2 && type.hasStaticShape() &&
         type.getDimSize(1) <= kThinWidth && type.getDimSize(0) >= kTallHeight;
}

// The chunk width for rebuilding a [H, W] chain on [W, H / chunk, chunk], or
// 0 when H does not split into tile-wide chunks.
int64_t denseChunk(int64_t height) {
  int64_t chunk = 0;
  for (int64_t candidate = kTileWidth; candidate * candidate <= height;
       candidate *= 2) {
    if (height % candidate == 0) {
      chunk = candidate;
    }
  }
  return chunk;
}

// [H, W] -> [W, H], or [W, H / chunk, chunk] for a nonzero chunk.
RankedTensorType transposedType(RankedTensorType type, int64_t chunk) {
  int64_t height = type.getDimSize(0);
  int64_t width = type.getDimSize(1);
  if (chunk == 0) {
    return RankedTensorType::get({width, height}, type.getElementType());
  }
  return RankedTensorType::get({width, height / chunk, chunk},
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
// shape), or an elementwise op over such values. The chunked shape also needs
// every broadcast to be constant along H (a per-row threshold, say), since H
// is split there.
bool isRebuildable(Value value, unsigned depth, bool chunked) {
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
    if (chunked) {
      auto operandType = cast<RankedTensorType>(broadcast.getOperand().getType());
      return llvm::all_of(
          llvm::enumerate(broadcast.getBroadcastDimensions()), [&](auto dim) {
            return dim.value() == 1 ||
                   operandType.getDimSize(dim.index()) == 1;
          });
    }
    return cast<RankedTensorType>(broadcast.getOperand().getType()).getRank() <=
           2;
  }
  if (auto constant = dyn_cast<shlo::ConstantOp>(op)) {
    return constant.getValue().isSplat();
  }
  if (auto constraint = dyn_cast<sdy::ShardingConstraintOp>(op)) {
    return constraint.getSharding().isFullyReplicated() &&
           isRebuildable(constraint.getInput(), depth, chunked);
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
           isRebuildable(operand, depth - 1, chunked);
  });
}

// `input` reshaped to `type` right after `transpose`, reusing such a reshape
// that every use of the transpose already sees.
Value chunkedLeaf(PatternRewriter &rewriter, shlo::TransposeOp transpose,
                  RankedTensorType type) {
  Value input = transpose.getOperand();
  for (Operation *user : input.getUsers()) {
    auto reshape = dyn_cast<shlo::ReshapeOp>(user);
    if (reshape && reshape.getType() == type &&
        reshape->getBlock() == transpose->getBlock() &&
        (reshape->isBeforeInBlock(transpose) ||
         reshape == transpose->getNextNode())) {
      return reshape;
    }
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(transpose);
  return rewriter.create<shlo::ReshapeOp>(transpose.getLoc(), type, input);
}

// Builds `value` on the transposed shape, chunked when `chunk` is nonzero;
// see isRebuildable for the cases.
Value rebuildTransposed(PatternRewriter &rewriter, Value value, int64_t chunk) {
  Operation *op = value.getDefiningOp();
  auto type = cast<RankedTensorType>(value.getType());
  RankedTensorType rebuiltType = transposedType(type, chunk);
  if (auto transpose = dyn_cast<shlo::TransposeOp>(op)) {
    if (chunk == 0) {
      return transpose.getOperand();
    }
    return chunkedLeaf(rewriter, transpose, rebuiltType);
  }
  if (auto broadcast = dyn_cast<shlo::BroadcastInDimOp>(op)) {
    // W is dimension 1 of [H, W] and 0 of either rebuilt shape, H is 0 and 1
    // (when chunked, only operand dimensions of size 1 map to H).
    SmallVector<int64_t> dims;
    for (int64_t dim : broadcast.getBroadcastDimensions()) {
      dims.push_back(1 - dim);
    }
    return rewriter.create<shlo::BroadcastInDimOp>(
        op->getLoc(), rebuiltType, broadcast.getOperand(),
        rewriter.getDenseI64ArrayAttr(dims));
  }
  if (auto constant = dyn_cast<shlo::ConstantOp>(op)) {
    auto splat = cast<SplatElementsAttr>(constant.getValue());
    return rewriter.create<shlo::ConstantOp>(
        op->getLoc(),
        SplatElementsAttr::get(rebuiltType, splat.getSplatValue<Attribute>()));
  }
  if (auto constraint = dyn_cast<sdy::ShardingConstraintOp>(op)) {
    sdy::TensorShardingAttr sharding = constraint.getSharding();
    if (chunk != 0) {
      bool closed = llvm::all_of(sharding.getDimShardings(),
                                 [](sdy::DimensionShardingAttr dim) {
                                   return dim.getIsClosed();
                                 });
      sharding = sdy::TensorShardingAttr::getFullyReplicated(
          op->getContext(), rebuiltType.getRank(), sharding.getMeshOrRef(),
          closed);
    }
    return rewriter.create<sdy::ShardingConstraintOp>(
        op->getLoc(), rebuildTransposed(rewriter, constraint.getInput(), chunk),
        sharding);
  }
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    operands.push_back(rebuildTransposed(rewriter, operand, chunk));
  }
  Operation *rebuilt = rewriter.clone(*op);
  rebuilt->setOperands(operands);
  rebuilt->getResult(0).setType(rebuiltType);
  return rebuilt->getResult(0);
}

// reduce(dims = [0]) of a tall, thin chain -> reduce(dims = [1, 2]) of the
// chain rebuilt on the chunked transposed shape, or reduce(dims = [1]) of it
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
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!isTallThin(inputType)) {
      return failure();
    }
    int64_t chunk = denseChunk(inputType.getDimSize(0));
    if (chunk != 0 && !isRebuildable(input, kMaxChainDepth, true)) {
      chunk = 0;
    }
    if (chunk == 0 && !isRebuildable(input, kMaxChainDepth, false)) {
      return failure();
    }
    Value transposed = rebuildTransposed(rewriter, input, chunk);
    SmallVector<int64_t> dims = {1};
    if (chunk != 0) {
      dims.push_back(2);
    }
    auto rebuilt = rewriter.create<shlo::ReduceOp>(
        reduceOp.getLoc(), reduceOp.getResultTypes(), ValueRange{transposed},
        reduceOp.getInitValues(), rewriter.getDenseI64ArrayAttr(dims));
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
