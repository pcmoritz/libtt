// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Reductions over the rows of a tall, thin tensor become reductions over full
// tiles of its transpose.
//
// A [H, W] tensor with W within one tile column has H / 32 tile rows and a
// single tile column. Reducing it over H serializes on that column (one core
// walks every tile row), and even the elementwise ops that feed the reduction
// run on that one column. JAX code that reduces repeatedly, such as the binary
// searches behind top-k and top-p masking, swaps the axes up front precisely
// to put the reduction on the leading axis, which is the right call on TPU and
// the wrong one here.
//
// The rewrite rebuilds the chain of elementwise ops, broadcasts and constants
// feeding such a reduce on [W, H / C, C]: the transposed data, row-major, in
// full tiles (plain [W, H] would fill only W of each tile's 32 rows). The
// reduce goes over the last two dimensions. Each transpose at a leaf of the
// chain becomes a reshape of its operand where the transpose was, so a
// loop-invariant leaf is reshaped once before the loop. C is the widest
// 32 * 2^k that divides H up to about sqrt(H): TTNN reduces the two dimensions
// in turn, and each stage walks one of them on few cores. For [2, 151936]
// (C = 128) a search step's elementwise ops take about 13 us each instead of
// 105, and its reduction about 26 us. Chains whose H does not split are left
// alone.
//
// StableHLO's import canonicalizations run alongside: they turn a loop carry
// that the body returns unchanged into a capture of the value the loop was
// entered with, which is what exposes the transpose behind a loop-invariant
// leaf.
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include <cstdint>

namespace mlir::tt {
namespace {

namespace shlo = mlir::stablehlo;

// A tensor whose last dimension fits one tile column and whose first is long
// enough for the column to matter.
constexpr int64_t kThinWidth = 32;
constexpr int64_t kTallHeight = 1024;
// How far up the operand chain the rewrite follows elementwise ops.
constexpr unsigned kMaxChainDepth = 12;
// Chunks are multiples of a tile's width.
constexpr int64_t kTileWidth = 32;

bool isTallThin(RankedTensorType type) {
  return type && type.getRank() == 2 && type.hasStaticShape() &&
         type.getDimSize(1) <= kThinWidth && type.getDimSize(0) >= kTallHeight;
}

// The chunk width C for [W, H / C, C], or 0 when H does not split into
// tile-wide chunks.
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

// [H, W] -> [W, H / chunk, chunk].
RankedTensorType chunkedType(RankedTensorType type, int64_t chunk) {
  return RankedTensorType::get(
      {type.getDimSize(1), type.getDimSize(0) / chunk, chunk},
      type.getElementType());
}

bool isReplicatedConstraint(sdy::ShardingConstraintOp constraint) {
  sdy::TensorShardingAttr sharding = constraint.getSharding();
  return sharding.isFullyReplicated() && sharding.isFullyClosed() &&
         sharding.getUnreducedAxes().empty();
}

// Collects the values of the chain feeding a reduce, operands before users,
// each once. The chain may hold swap-transposes (its leaves), broadcasts that
// are constant along H (a per-row threshold, say, since H is split), splat
// constants, closed replicated sharding constraints (JAX leaves one after most
// ops under a shard_map), and elementwise ops over such values. Returns false
// for anything else.
bool collectChain(Value value, unsigned depth, llvm::SetVector<Value> &chain) {
  if (chain.contains(value)) {
    return true;
  }
  auto type = dyn_cast<RankedTensorType>(value.getType());
  Operation *op = value.getDefiningOp();
  if (!type || type.getRank() != 2 || !op) {
    return false;
  }
  if (auto transpose = dyn_cast<shlo::TransposeOp>(op)) {
    if (!llvm::equal(transpose.getPermutation(), ArrayRef<int64_t>{1, 0})) {
      return false;
    }
  } else if (auto broadcast = dyn_cast<shlo::BroadcastInDimOp>(op)) {
    auto operandType = cast<RankedTensorType>(broadcast.getOperand().getType());
    for (auto [index, dim] :
         llvm::enumerate(broadcast.getBroadcastDimensions())) {
      if (dim == 0 && operandType.getDimSize(index) != 1) {
        return false;
      }
    }
  } else if (auto constant = dyn_cast<shlo::ConstantOp>(op)) {
    if (!constant.getValue().isSplat()) {
      return false;
    }
  } else if (auto constraint = dyn_cast<sdy::ShardingConstraintOp>(op)) {
    if (!isReplicatedConstraint(constraint) ||
        !collectChain(constraint.getInput(), depth, chain)) {
      return false;
    }
  } else {
    bool elementwise = op->hasTrait<OpTrait::Elementwise>() ||
                       isa<shlo::CompareOp, shlo::SelectOp>(op);
    if (depth == 0 || !elementwise || op->getNumResults() != 1 ||
        op->getNumRegions() != 0) {
      return false;
    }
    for (Value operand : op->getOperands()) {
      if (cast<RankedTensorType>(operand.getType()).getShape() !=
              type.getShape() ||
          !collectChain(operand, depth - 1, chain)) {
        return false;
      }
    }
  }
  chain.insert(value);
  return true;
}

// Builds `value` on the chunked shape, its operands already in `mapping`.
Value rebuildChunked(PatternRewriter &rewriter, Value value, int64_t chunk,
                     IRMapping &mapping) {
  Operation *op = value.getDefiningOp();
  RankedTensorType type =
      chunkedType(cast<RankedTensorType>(value.getType()), chunk);
  if (auto transpose = dyn_cast<shlo::TransposeOp>(op)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(transpose);
    return rewriter.create<shlo::ReshapeOp>(transpose.getLoc(), type,
                                            transpose.getOperand());
  }
  if (auto broadcast = dyn_cast<shlo::BroadcastInDimOp>(op)) {
    // W is dimension 1 of [H, W] and 0 of [W, H / C, C]; operand dimensions
    // of size 1 that went to H go to H / C.
    SmallVector<int64_t> dims;
    for (int64_t dim : broadcast.getBroadcastDimensions()) {
      dims.push_back(1 - dim);
    }
    return rewriter.create<shlo::BroadcastInDimOp>(
        op->getLoc(), type, broadcast.getOperand(),
        rewriter.getDenseI64ArrayAttr(dims));
  }
  if (auto constant = dyn_cast<shlo::ConstantOp>(op)) {
    auto splat = cast<SplatElementsAttr>(constant.getValue());
    return rewriter.create<shlo::ConstantOp>(
        op->getLoc(),
        SplatElementsAttr::get(type, splat.getSplatValue<Attribute>()));
  }
  if (auto constraint = dyn_cast<sdy::ShardingConstraintOp>(op)) {
    return rewriter.create<sdy::ShardingConstraintOp>(
        op->getLoc(), mapping.lookup(constraint.getInput()),
        sdy::TensorShardingAttr::getFullyReplicated(
            op->getContext(), type.getRank(),
            constraint.getSharding().getMeshOrRef(), /*isClosed=*/true));
  }
  Operation *rebuilt = rewriter.clone(*op, mapping);
  rebuilt->getResult(0).setType(type);
  return rebuilt->getResult(0);
}

// reduce(dims = [0]) of a [H, W] chain -> reduce(dims = [1, 2]) of the chain
// rebuilt on [W, H / C, C].
class TallReductionToWide : public OpRewritePattern<shlo::ReduceOp> {
public:
  using OpRewritePattern<shlo::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shlo::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getInputs().size() != 1 ||
        !llvm::equal(reduceOp.getDimensions(), ArrayRef<int64_t>{0})) {
      return failure();
    }
    Value input = reduceOp.getInputs().front();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!isTallThin(inputType)) {
      return failure();
    }
    int64_t chunk = denseChunk(inputType.getDimSize(0));
    llvm::SetVector<Value> chain;
    if (chunk == 0 || !collectChain(input, kMaxChainDepth, chain)) {
      return failure();
    }
    IRMapping mapping;
    for (Value value : chain) {
      mapping.map(value, rebuildChunked(rewriter, value, chunk, mapping));
    }
    auto rebuilt = rewriter.create<shlo::ReduceOp>(
        reduceOp.getLoc(), reduceOp.getResultTypes(),
        ValueRange{mapping.lookup(input)}, reduceOp.getInitValues(),
        rewriter.getDenseI64ArrayAttr({1, 2}));
    rewriter.inlineRegionBefore(reduceOp.getBody(), rebuilt.getBody(),
                                rebuilt.getBody().end());
    rewriter.replaceOp(reduceOp, rebuilt.getResults());
    return success();
  }
};

} // namespace

void populateWideReductionPatterns(RewritePatternSet &patterns) {
  patterns.add<TallReductionToWide>(patterns.getContext());
  mlir::stablehlo::populateStablehloHloImportCanonicalizationPatterns(
      patterns.getContext(), &patterns);
}

} // namespace mlir::tt
