// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// libtt's row merge (see ttmlir/D2MFusion.h): a D2M subgraph whose generics
// are all elementwise or last-dim reductions over one 2-D tile space becomes
// one generic that holds whole rows on each core, so an in-row reduction feeds
// its broadcast epilogue within one kernel. D2MGridSelection marks the
// candidates (annotateRowMergeCandidates) and D2MGenericFusion merges them
// (mergeRowGroups).

#include "ttmlir/D2MFusion.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/BlockFactorAnalysis.h"
#include "ttmlir/Dialect/D2M/Analysis/GridAnalysis.h"
#include "ttmlir/Dialect/D2M/Analysis/TopKShardingStrategy.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"
#include "ttmlir/Dialect/D2M/Utils/DstRegisterAnalysis.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/D2M/Utils/SpatialOpNormalizeUtil.h"
#include "ttmlir/Dialect/D2M/Utils/TopKUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>

namespace mlir::tt::d2m {
namespace {

// A function whose generics
// are all elementwise or last-dim reductions over one 2-D tile space, with
// at least one reduction, gets d2m.unsplit_dims = [1] on every generic so
// each core holds whole rows; GenericFusion then merges them into one.
static bool isRowMergeCandidate(GenericOp g) {
  if (!g.isComputeOnlyForm() || !g.hasPureTensorSemantics() ||
      g.getNumRegions() != 1 || !g.getRegion(0).hasOneBlock() ||
      !g.getAdditionalArgs().empty() || g.getIndexingMapsValue().empty() ||
      g.getIndexingMapsValue().front().getNumDims() != 2) {
    return false;
  }
  auto iters = g.getIteratorTypesValue();
  if (iters.size() != 2 || iters[0] != ttcore::IteratorType::Parallel) {
    return false;
  }
  return llvm::all_of(g.getRegion(0).front(), [](Operation &op) {
    return isa<tensor::EmptyOp, RemoteLoadOp, RemoteStoreOp, BlockIndexOp,
               linalg::GenericOp, YieldOp, arith::ConstantOp>(op);
  });
}

// Exp-based and similar tile ops (also matched by TTIR name in
// TTIRToD2M's rematerializeForSpreadRows).
static bool isTranscendental(StringRef name) {
  return llvm::is_contained({"d2m.tile_silu", "d2m.tile_sigmoid",
                             "d2m.tile_exp", "d2m.tile_tanh", "d2m.tile_gelu",
                             "d2m.tile_log"},
                            name);
}

static void annotateRowMergeCandidates(ModuleOp module) {
  module.walk([](func::FuncOp func) {
    SmallVector<GenericOp> generics;
    for (Operation &op : func.getBody().getOps()) {
      if (auto g = dyn_cast<GenericOp>(op)) {
        generics.push_back(g);
      }
    }
    if (generics.size() < 2 || !llvm::all_of(generics, isRowMergeCandidate)) {
      return;
    }
    bool hasReduction = llvm::any_of(generics, [](GenericOp g) {
      return g.getIteratorTypesValue()[1] == ttcore::IteratorType::Reduction;
    });
    if (!hasReduction) {
      return;
    }
    auto tiles = [](Value v) -> std::optional<std::pair<int64_t, int64_t>> {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getRank() != 4 ||
          !isa<ttcore::TileType>(t.getElementType())) {
        return std::nullopt;
      }
      auto s = t.getShape();
      return std::make_pair(s[0] * s[2], s[1] * s[3]);
    };
    int64_t rows = 1, cols = 1;
    for (GenericOp g : generics) {
      for (Value v : g->getOperands()) {
        auto t = tiles(v);
        if (!t) {
          return;
        }
        rows = std::max(rows, t->first);
        cols = std::max(cols, t->second);
      }
    }
    for (GenericOp g : generics) {
      for (Value v : g->getOperands()) {
        auto t = *tiles(v);
        if ((t.first != rows && t.first != 1) ||
            (t.second != cols && t.second != 1)) {
          return;
        }
      }
    }
    Builder b(func.getContext());
    // Fewer than four tile rows, or a transcendental epilogue (measured:
    // Qwen3.8's gated norm 12x128 12.0 -> 5.5 us, 70x64 7.2 -> 4.9 us,
    // 192x128 12.9 -> 6.1 us; Inkling 2's q clip over 6 tile rows got
    // slower, 5.9 -> 8.1 us):
    // only members that reduce over the columns (or feed a reduction) stay
    // on one core per row; the others spread over the columns, and the
    // merged kernel recomputes each row's reductions on every core of its
    // row (as hand-written row kernels split output tiles over cores).
    bool transcendental = false;
    for (GenericOp g : generics) {
      g->walk([&](Operation *op) {
        transcendental |= isTranscendental(op->getName().getStringRef());
      });
    }
    // A wide row reduced in chunks (ttir.chunk_allreduce): one chunk per
    // core, all members on whole chunk rows (d2m.chunk_rows: GridAnalysis
    // gives each core one tile row, on a virtual grid if needed).
    bool chunked = false;
    for (GenericOp g : generics) {
      g->walk([&](Operation *op) {
        chunked |= op->getName().getStringRef() == "d2m.tile_chunk_allreduce";
      });
    }
    const bool spread = !chunked && cols > 1 && (rows < 4 || transcendental);
    // Members feeding a reduction (directly or through other members, also
    // through views) must stay on whole rows.
    llvm::SmallPtrSet<Operation *, 16> feedsReduction;
    for (GenericOp g : llvm::reverse(generics)) {
      bool feeds = false;
      SmallVector<Value> work(g->getResults().begin(), g->getResults().end());
      llvm::SmallPtrSet<Operation *, 16> seen;
      while (!work.empty() && !feeds) {
        Value v = work.pop_back_val();
        for (Operation *user : v.getUsers()) {
          if (!seen.insert(user).second) {
            continue;
          }
          Operation *anc = func.getBody().front().findAncestorOpInBlock(*user);
          auto ug = anc ? dyn_cast<GenericOp>(anc) : GenericOp();
          if (ug && ug != g) {
            if (ug.getIteratorTypesValue()[1] ==
                    ttcore::IteratorType::Reduction ||
                feedsReduction.contains(ug)) {
              feeds = true;
            }
          } else if (!ug && user->getNumResults() > 0) {
            for (Value r : user->getResults()) {
              work.push_back(r);
            }
          }
        }
      }
      if (feeds) {
        feedsReduction.insert(g);
      }
    }
    for (GenericOp g : generics) {
      bool keepRow =
          !spread ||
          g.getIteratorTypesValue()[1] == ttcore::IteratorType::Reduction ||
          feedsReduction.contains(g) ||
          llvm::all_of(g->getOperands(),
                       [&](Value v) { return tiles(v)->second == 1; });
      if (chunked) {
        g->setAttr("d2m.chunk_rows", b.getUnitAttr());
      }
      if (keepRow) {
        g->setAttr("d2m.unsplit_dims", b.getI64ArrayAttr({1}));
      } else {
        g->setAttr("d2m.row_merge_member", b.getUnitAttr());
      }
    }
  });
}

// Merge the generics of one function that GridSelection kept on whole
// rows (d2m.unsplit_dims) into a single generic. Each original region's linalg
// ops are
// cloned in order; loads of values produced inside the group take the
// producer's tile tensor (so values with several consumers are fine), and only
// results used outside the group are stored. An in-row reduction then feeds
// its broadcast epilogue within one kernel.
static void mergeRowGroups(ModuleOp module) {
  SmallVector<func::FuncOp> funcs;
  module.walk([&](func::FuncOp f) { funcs.push_back(f); });
  for (func::FuncOp func : funcs) {
    SmallVector<GenericOp> generics;
    for (Operation &op : func.getBody().getOps()) {
      if (auto g = dyn_cast<GenericOp>(op)) {
        generics.push_back(g);
      }
    }
    if (generics.size() < 2 || !llvm::all_of(generics, [](GenericOp g) {
          return g->hasAttr("d2m.unsplit_dims") ||
                 g->hasAttr("d2m.row_merge_member");
        })) {
      continue;
    }
    // The merged grid is the largest member grid; members on one core per
    // row (grid [R, 1]) are recomputed on every core of their row.
    GenericOp first = generics.front();
    for (GenericOp g : generics) {
      if (g.getGrid().getGridVolume() > first.getGrid().getGridVolume()) {
        first = g;
      }
    }
    ArrayRef<int64_t> mergedGrid = first.getGrid().getShape();
    auto gridFits = [&](GenericOp g) {
      ArrayRef<int64_t> shape = g.getGrid().getShape();
      if (shape.size() != mergedGrid.size()) {
        return false;
      }
      for (auto [a, b] : llvm::zip(shape, mergedGrid)) {
        if (a != b && a != 1) {
          return false;
        }
      }
      return true;
    };
    bool uniform = llvm::all_of(generics, [&](GenericOp g) {
      return gridFits(g) &&
             (g.getGrid() != first.getGrid() ||
              g.getBlockFactors() == first.getBlockFactors()) &&
             g.getRegion(0).hasOneBlock() &&
             llvm::all_of(g.getRegion(0).front(), [](Operation &op) {
               return isa<tensor::EmptyOp, RemoteLoadOp, RemoteStoreOp,
                          BlockIndexOp, linalg::GenericOp, YieldOp,
                          arith::ConstantOp>(op);
             });
    });
    if (!uniform) {
      continue;
    }
    llvm::SmallPtrSet<Operation *, 16> group;
    for (GenericOp g : generics) {
      group.insert(g);
    }
    // A group result seen through views and layout casts (another member's
    // grid) is still internal: reading its buffer in the same kernel would
    // race the store.
    auto throughViews = [&](Value v) {
      while (Operation *def = v.getDefiningOp()) {
        if (auto view = dyn_cast<ViewLayoutOp>(def)) {
          v = view.getInput();
        } else if (auto cast = dyn_cast<ttir::TTNNMetalLayoutCastOp>(def)) {
          v = cast->getOperand(0);
        } else {
          break;
        }
      }
      return v;
    };
    auto inGroup = [&](Value v) {
      Operation *def = throughViews(v).getDefiningOp();
      return def && group.contains(def);
    };
    // Tiles per dim of a [grid0, grid1, shard0, shard1] operand.
    auto tiles = [](Value v) {
      auto s = mlir::cast<ShapedType>(v.getType()).getShape();
      return std::make_pair(s[0] * s[2], s[1] * s[3]);
    };
    int64_t rows = 1, cols = 1;
    for (GenericOp g : generics) {
      for (Value v : g->getOperands()) {
        rows = std::max(rows, tiles(v).first);
        cols = std::max(cols, tiles(v).second);
      }
    }
    MLIRContext *ctx = module.getContext();
    const bool spreadGroup = llvm::any_of(
        generics, [&](GenericOp g) { return g.getGrid() != first.getGrid(); });
    auto mapFor = [&](Value v) {
      AffineExpr zero = getAffineConstantExpr(0, ctx);
      if (spreadGroup) {
        // By the operand's own grid: a dim it does not split is block 0.
        auto shape = mlir::cast<ShapedType>(v.getType()).getShape();
        return AffineMap::get(
            2, 0,
            {shape[0] == mergedGrid[0] ? getAffineDimExpr(0, ctx) : zero,
             shape[1] == mergedGrid[1] ? getAffineDimExpr(1, ctx) : zero},
            ctx);
      }
      auto [r, c] = tiles(v);
      return AffineMap::get(2, 0,
                            {r == rows ? getAffineDimExpr(0, ctx) : zero,
                             c == cols ? getAffineDimExpr(1, ctx) : zero},
                            ctx);
    };

    SmallVector<Value> inputs;
    llvm::DenseMap<Value, unsigned> inputIndex;
    for (GenericOp g : generics) {
      for (Value v : g.getInputs()) {
        if (!inGroup(v) && !inputIndex.count(v)) {
          inputIndex[v] = inputs.size();
          inputs.push_back(v);
        }
      }
    }
    SmallVector<std::pair<GenericOp, unsigned>> escaping;
    for (GenericOp g : generics) {
      for (OpResult r : g->getResults()) {
        bool escapes = llvm::any_of(r.getUsers(), [&](Operation *user) {
          // Loads inside a group generic's region are internal uses.
          for (Operation *anc = user; anc; anc = anc->getParentOp()) {
            if (group.contains(anc)) {
              return false;
            }
          }
          // Leftover views and casts without uses are not real consumers;
          // views and casts read only by group members are internal.
          if (isa<ViewLayoutOp, ttir::TTNNMetalLayoutCastOp>(user)) {
            std::function<bool(Operation *)> internalOnly =
                [&](Operation *u) -> bool {
              return llvm::all_of(u->getUsers(), [&](Operation *uu) {
                for (Operation *anc = uu; anc; anc = anc->getParentOp()) {
                  if (group.contains(anc)) {
                    return true;
                  }
                }
                return isa<ViewLayoutOp, ttir::TTNNMetalLayoutCastOp>(uu) &&
                       internalOnly(uu);
              });
            };
            if (internalOnly(user)) {
              return false;
            }
          }
          return true;
        });
        if (escapes) {
          escaping.emplace_back(g, r.getResultNumber());
        }
      }
    }
    if (escaping.empty()) {
      continue;
    }
    // The merged generic sits after the last group member, so every outside
    // user of a group result must come after it too (a mask between two
    // members would otherwise use a value defined later).
    bool usersAfter = llvm::all_of(escaping, [&](auto gr) {
      return llvm::all_of(
          gr.first->getResult(gr.second).getUsers(), [&](Operation *user) {
            Operation *anc =
                func.getBody().front().findAncestorOpInBlock(*user);
            return !anc || group.contains(anc) ||
                   generics.back()->isBeforeInBlock(anc);
          });
    });
    if (!usersAfter) {
      continue;
    }
    SmallVector<Value> outputs;
    SmallVector<AffineMap> maps;
    for (Value v : inputs) {
      maps.push_back(mapFor(v));
    }
    for (auto [g, r] : escaping) {
      Value init = g.getDpsInitOperand(r)->get();
      outputs.push_back(init);
      maps.push_back(mapFor(init));
    }

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPointAfter(generics.back());
    Location loc = generics.back().getLoc();
    auto parallel =
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel);
    auto merged = rewriter.create<GenericOp>(
        loc, TypeRange(ValueRange(outputs)), inputs, outputs, ValueRange{},
        first.getGrid(), first.getBlockFactors(),
        rewriter.getAffineMapArrayAttr(maps),
        rewriter.getArrayAttr({parallel, parallel}), first.getThreads(),
        first.getFabricConnectionConfigAttr(), /*regions=*/1);
    if (Attribute chunkRows = first->getAttr("d2m.chunk_rows")) {
      merged->setAttr("d2m.chunk_rows", chunkRows);
    }
    if (Attribute unsplit = first->getAttr("d2m.unsplit_dims")) {
      merged->setAttr("d2m.unsplit_dims", unsplit);
    }

    Block &block = merged.getRegion(0).emplaceBlock();
    rewriter.setInsertionPointToStart(&block);
    Value block0 = rewriter.create<BlockIndexOp>(loc, static_cast<int64_t>(0));
    Value block1 = rewriter.create<BlockIndexOp>(loc, static_cast<int64_t>(1));
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto indicesFor = [&](AffineMap map) {
      SmallVector<Value> indices;
      for (AffineExpr e : map.getResults()) {
        auto d = mlir::dyn_cast<AffineDimExpr>(e);
        indices.push_back(!d ? c0 : (d.getPosition() == 0 ? block0 : block1));
      }
      return indices;
    };
    auto shardType = [](Value v) {
      auto t = mlir::cast<RankedTensorType>(v.getType());
      return RankedTensorType::get(t.getShape().drop_front(2),
                                   t.getElementType());
    };
    llvm::DenseMap<Value, Value> loaded;
    for (auto [i, v] : llvm::enumerate(inputs)) {
      RankedTensorType st = shardType(v);
      Value buffer = rewriter.create<tensor::EmptyOp>(loc, st.getShape(),
                                                      st.getElementType());
      loaded[v] =
          rewriter.create<RemoteLoadOp>(loc, st, buffer, v, indicesFor(maps[i]))
              .getResult();
    }

    bool mergeFailed = false;
    llvm::DenseMap<std::pair<Operation *, unsigned>, Value> produced;
    IRMapping irMap;
    for (GenericOp g : generics) {
      for (Operation &op : g.getRegion(0).front()) {
        if (auto bi = dyn_cast<BlockIndexOp>(op)) {
          // A member not split along a dim the merged grid splits sees block
          // 0 there (its whole extent on every core).
          int64_t dim = bi.getDim();
          bool split = g.getGrid().getShape()[dim] == mergedGrid[dim];
          irMap.map(bi.getResult(), !split ? c0 : (dim == 0 ? block0 : block1));
        } else if (auto load = dyn_cast<RemoteLoadOp>(op)) {
          Value src = load.getMemref();
          Value repl;
          if (inGroup(src)) {
            Value base = throughViews(src);
            repl = produced.lookup(
                {base.getDefiningOp(), cast<OpResult>(base).getResultNumber()});
          } else {
            repl = loaded.lookup(src);
          }
          if (!repl || repl.getType() != load.getResult().getType()) {
            mergeFailed = true;
            break;
          }
          irMap.map(load.getResult(), repl);
        } else if (auto store = dyn_cast<RemoteStoreOp>(op)) {
          auto outs = g.getOutputs();
          auto it = llvm::find(outs, store.getMemref());
          if (it == outs.end()) {
            mergeFailed = true;
            break;
          }
          produced[{g.getOperation(),
                    static_cast<unsigned>(std::distance(outs.begin(), it))}] =
              irMap.lookupOrDefault(store.getLocalBuffer());
        } else if (!isa<YieldOp>(op)) {
          rewriter.clone(op, irMap);
        }
      }
      if (mergeFailed) {
        break;
      }
    }
    SmallVector<Value> stored;
    if (!mergeFailed) {
      for (auto [k, gr] : llvm::enumerate(escaping)) {
        Value value = produced.lookup({gr.first.getOperation(), gr.second});
        if (!value) {
          mergeFailed = true;
          break;
        }
        stored.push_back(rewriter
                             .create<RemoteStoreOp>(
                                 loc, outputs[k].getType(), outputs[k],
                                 indicesFor(maps[inputs.size() + k]), value)
                             .getResult());
      }
    }
    if (mergeFailed) {
      rewriter.eraseOp(merged);
      continue;
    }
    rewriter.create<YieldOp>(loc, stored);

    // Fuse chains of all-parallel linalg ops inside the merged region (as the
    // general-fusion branch did), so values stay in DST between operations
    // instead of round-tripping through L1 scratch.
    {
      bool changed = true;
      // Each fusion removes a consumer, but keeping a multi-use producer's
      // value as an extra result can re-create fusable pairs; cap the rounds.
      for (int round = 0; changed && round < 64; ++round) {
        changed = false;
        // Merge repeated inputs (a consumer of v * v), so fusing the
        // producer replaces both reads instead of keeping it for one.
        {
          RewritePatternSet dedupe(rewriter.getContext());
          linalg::populateEraseUnusedOperandsAndResultsPatterns(dedupe);
          SmallVector<Operation *> ops;
          for (auto g : block.getOps<linalg::GenericOp>()) {
            ops.push_back(g);
          }
          (void)applyOpPatternsGreedily(ops, std::move(dedupe));
        }
        for (auto consumer :
             llvm::make_early_inc_range(block.getOps<linalg::GenericOp>())) {
          for (OpOperand &operand : consumer->getOpOperands()) {
            auto producer = operand.get().getDefiningOp<linalg::GenericOp>();
            // A broadcast unpacks its operand from L1 (unary_bcast), so a
            // broadcast operand cannot become a DST intermediate.
            if (operand.getOperandNumber() < consumer.getNumDpsInputs() &&
                llvm::any_of(
                    consumer.getMatchingBlockArgument(&operand).getUsers(),
                    [](Operation *u) { return isa<TileBcastOp>(u); })) {
              continue;
            }
            // Only single-consumer producers: a shared value (e.g. the scaled
            // input read by several epilogue ops) is computed once and kept.
            if (!producer || producer.getNumReductionLoops() != 0 ||
                !llvm::all_of(producer->getUsers(),
                              [&](Operation *u) { return u == consumer; }) ||
                !linalg::areElementwiseOpsFusable(&operand)) {
              continue;
            }
            rewriter.setInsertionPoint(consumer);
            FailureOr<linalg::ElementwiseOpFusionResult> fusion =
                linalg::fuseElementwiseOps(rewriter, &operand);
            if (failed(fusion)) {
              continue;
            }
            // Keep one tile of the fused body within DST: the unscheduled
            // allocator neither spills nor bounds-checks (an overrun popped
            // an empty slice stack), so undo fusions whose DST slices exceed
            // the half-sync capacity of the body's widest type.
            {
              auto fusedOp = mlir::cast<linalg::GenericOp>(fusion->fusedOp);
              int64_t slices = utils::estimateDstSlicesPerTile(fusedOp);
              unsigned capacity = utils::getDstCapacityTilesForBody(fusedOp);
              if (slices > static_cast<int64_t>(capacity)) {
                rewriter.eraseOp(fusedOp);
                continue;
              }
            }
            for (auto [original, replacement] : fusion->replacements) {
              original.replaceAllUsesWith(replacement);
            }
            rewriter.eraseOp(consumer);
            if (producer->use_empty()) {
              rewriter.eraseOp(producer);
            }
            changed = true;
            break;
          }
          if (changed) {
            break;
          }
        }
      }
    }

    for (auto [k, gr] : llvm::enumerate(escaping)) {
      gr.first->getResult(gr.second).replaceUsesWithIf(
          merged->getResult(k), [&](OpOperand &use) {
            for (Operation *anc = use.getOwner(); anc;
                 anc = anc->getParentOp()) {
              if (group.contains(anc)) {
                return false;
              }
            }
            return true;
          });
    }
    for (GenericOp g : llvm::reverse(generics)) {
      g->dropAllUses();
      rewriter.eraseOp(g);
    }
    // Views and casts that only connected members are dead now (some lost
    // their operand to the erased members).
    for (bool erased = true; erased;) {
      erased = false;
      for (Operation &op : llvm::make_early_inc_range(func.getBody().front())) {
        if (isa<ViewLayoutOp, ttir::TTNNMetalLayoutCastOp>(op) &&
            op.use_empty()) {
          op.erase();
          erased = true;
        }
      }
    }
  }
}

} // namespace
} // namespace mlir::tt::d2m

namespace mlir::tt::d2m_fusion {

void annotateRowMergeCandidates(ModuleOp module) {
  d2m::annotateRowMergeCandidates(module);
}

void mergeRowGroups(ModuleOp module) { d2m::mergeRowGroups(module); }

} // namespace mlir::tt::d2m_fusion
