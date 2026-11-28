// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// libtt's in-kernel all-reduce over the cores of a D2M program (see
// ttmlir/D2MFusion.h, ttir.chunk_allreduce). A wide-row reduction split in k
// chunks runs as one row-merged kernel with one chunk per core; each core's
// row sums (one FP32 tile, column 0) are packed by one compute phase and read
// by a later one at a libtt_chunk_allreduce_tile marker (one exchange per
// marker, in order). Here, after
// D2MToTTNN, that pack goes to a new one-tile CB P, which the reader kernel
// takes (libtt_dm_chunk_allreduce): it writes the row sums into row `me` of
// the root core's new gather CB R (ceil(k / 32) tiles) and bumps the root's
// semaphore; the root multicasts R to the others once all rows arrived, and
// every core pushes R. The consuming phase reads R instead, adds its tiles
// and, at the marker, sums the rows and turns the row into the columns
// (libtt_sfpu::add_gather, gather_combine: only the partial's rows).
// tt-metal's sharded layernorm kernels gather and multicast partials the
// same way.

#include "ttmlir/D2MFusion.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::d2m_fusion {
namespace {

namespace ttk = mlir::tt::ttkernel;

// The constant index operand of `get_common_arg_val(%cN)`, if any.
std::optional<int64_t> argIndex(Value v) {
  auto get = v.getDefiningOp<ttk::GetCommonArgValOp>();
  if (!get) {
    return std::nullopt;
  }
  auto c = get.getArgIndex().getDefiningOp<arith::ConstantIndexOp>();
  if (!c) {
    return std::nullopt;
  }
  return c.value();
}

// A new `get_common_arg_val(index)` of `type` at the start of `func`.
Value newArg(func::FuncOp func, int64_t index, Type type) {
  OpBuilder b(func.getContext());
  b.setInsertionPointToStart(&func.getBody().front());
  Location loc = func.getLoc();
  Value i = b.create<arith::ConstantIndexOp>(loc, index);
  return b.create<ttk::GetCommonArgValOp>(loc, type, i);
}

// The marker's operand: the tile copied from CB `cb` at index `index` into
// its DST slot, packed there by `pack` in an earlier phase.
struct Partial {
  ttk::CopyTileOp copy;
  ttk::PackTileOp pack;
};

bool sameIndex(Value a, Value b) {
  if (a == b) {
    return true;
  }
  auto ca = a.getDefiningOp<arith::ConstantIndexOp>();
  auto cb = b.getDefiningOp<arith::ConstantIndexOp>();
  return ca && cb && ca.value() == cb.value();
}

FailureOr<Partial> findPartial(func::FuncOp compute,
                               ttk::LibttChunkAllReduceTileOp marker) {
  Partial p;
  for (Operation *op = marker->getPrevNode(); op && !p.copy;
       op = op->getPrevNode()) {
    if (auto c = dyn_cast<ttk::CopyTileOp>(op);
        c && sameIndex(c.getTileIndexDst(), marker.getTileIndex())) {
      p.copy = c;
    }
  }
  if (!p.copy) {
    return marker.emitError("libtt: chunk all-reduce without a CB copy");
  }
  // The last pack to that CB page before the copy.
  compute.walk([&](Operation *op) {
    if (op == p.copy) {
      return WalkResult::interrupt();
    }
    if (auto k = dyn_cast<ttk::PackTileOp>(op);
        k && k.getOutCb() == p.copy.getCb0() &&
        sameIndex(k.getOutIndex(), p.copy.getTileIndexCb())) {
      p.pack = k;
    }
    return WalkResult::advance();
  });
  if (!p.pack || p.pack->getParentOp() != compute) {
    return marker.emitError("libtt: chunk all-reduce partial not packed once");
  }
  return p;
}

// An SFPU helper call (libtt_sfpu::<fn>(slots..., rows)) before `at`.
void helperCall(OpBuilder &b, Operation *at, StringRef fn, ValueRange slots,
                int64_t rows) {
  b.setInsertionPoint(at);
  std::string call = "MATH((libtt_sfpu::" + fn.str() + "(";
  for (size_t i = 0; i < slots.size(); ++i) {
    call += "{}, ";
  }
  call += std::to_string(rows) + ")));";
  OperationState state(at->getLoc(), emitc::VerbatimOp::getOperationName());
  state.addOperands(slots);
  state.addAttribute("value", b.getStringAttr(call));
  state.addAttribute("libtt.sfpu_def",
                     b.getStringAttr("// libtt in-DST tile helpers"));
  b.create(state);
}

// The pack of the partial goes to `send`; the phase that read it reads R's
// tiles instead and adds them (the columns < rows), then combines the gather
// rows where the marker was (libtt_sfpu::gather_combine).
LogicalResult rewriteCompute(func::FuncOp compute,
                             ttk::LibttChunkAllReduceTileOp marker, Partial p,
                             Value send, Value recv, int64_t recvTiles,
                             int64_t rows) {
  Location loc = marker.getLoc();
  OpBuilder top(compute.getContext());
  top.setInsertionPointToStart(&compute.getBody().front());
  Value one = top.create<arith::ConstantIntOp>(loc, 1, 32);
  Value pages = top.create<arith::ConstantIntOp>(loc, recvTiles, 32);
  Value zero = top.create<arith::ConstantIndexOp>(loc, 0);

  OpBuilder b(p.pack);
  b.create<ttk::CBReserveBackOp>(loc, send, one);
  p.pack.getOutCbMutable().assign(send);
  p.pack.getOutIndexMutable().assign(zero);
  b.setInsertionPointAfter(p.pack);
  b.create<ttk::CBPushBackOp>(loc, send, one);

  // The unpacker setup for the copy (same format) and the copy: R.
  Value old = p.copy.getCb0();
  for (Operation *op = p.copy->getPrevNode();
       op && !isa<ttk::TileRegsAcquireOp>(op); op = op->getPrevNode()) {
    if (isa<ttk::ReconfigDataFormatSrcAOp, ttk::CopyTileInitOp>(op) &&
        op->getOperand(0) == old) {
      op->setOperand(0, recv);
    }
  }
  b.setInsertionPoint(p.copy);
  b.create<ttk::CBWaitFrontOp>(loc, recv, pages);
  p.copy.getCb0Mutable().assign(recv);
  p.copy.getTileIndexCbMutable().assign(zero);
  b.setInsertionPointAfter(marker);
  b.create<ttk::CBPopFrontOp>(loc, recv, pages);
  Operation *at = marker->getPrevNode();
  if (!isa_and_nonnull<ttk::LibttChunkAllReduceTileInitOp>(at)) {
    at = marker;
  }
  auto combine = [&]() {
    helperCall(b, marker, "gather_combine", marker.getTileIndex(), rows);
    if (at != marker) {
      at->erase();
    }
    marker->erase();
    return success();
  };
  if (recvTiles == 1) {
    return combine();
  }
  // A free DST slot for the other tiles: in a kernel kept in DST, the one
  // keepKernelInDst found free at the marker (libtt.tmp_slot); else one no
  // tile op of the marker's acquired section names (its indices are
  // constants; FP32 DST holds 4 tiles in half-sync mode).
  Operation *acquire = nullptr, *commit = nullptr;
  for (Operation *op = marker->getPrevNode(); op && !acquire;
       op = op->getPrevNode()) {
    if (isa<ttk::TileRegsAcquireOp>(op)) {
      acquire = op;
    }
  }
  for (Operation *op = marker->getNextNode(); op && !commit;
       op = op->getNextNode()) {
    if (isa<ttk::TileRegsCommitOp>(op)) {
      commit = op;
    }
  }
  if (!acquire || !commit) {
    return marker.emitError("libtt: chunk all-reduce outside a DST section");
  }
  llvm::SmallDenseSet<int64_t> used;
  for (Operation *op = acquire; op != commit; op = op->getNextNode()) {
    op->walk([&](Operation *nested) {
      for (Value v : nested->getOperands()) {
        if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
          used.insert(c.value());
        }
      }
    });
  }
  int64_t tmp = 1;
  while (used.contains(tmp)) {
    ++tmp;
  }
  if (auto slot = marker->getAttrOfType<IntegerAttr>("libtt.tmp_slot")) {
    tmp = slot.getInt();
  } else if (tmp >= 4) {
    return marker.emitError("libtt: chunk all-reduce: no free DST slot");
  }
  b.setInsertionPoint(at);
  Value tmpIdx = b.create<arith::ConstantIndexOp>(loc, tmp);
  for (int64_t i = 1; i < recvTiles; ++i) {
    Value tile = b.create<arith::ConstantIndexOp>(loc, i);
    b.create<ttk::CopyTileInitOp>(loc, recv);
    b.create<ttk::CopyTileOp>(loc, recv, tile, tmpIdx);
    helperCall(b, at, "add_gather", {marker.getTileIndex(), tmpIdx}, rows);
  }
  return combine();
}

// A whole-tile marker (libtt.full_tile): the partial's tiles packed into a
// CB (in a loop over them or once) and copied back, at the marker, in a loop
// over the same tiles (its induction variable the tile index) or once.
struct FullPartial {
  ttk::CopyTileOp copy;
  ttk::PackTileOp pack;
  Operation *packAt;   // the pack, or its loop (at the kernel's top level)
  Operation *markerAt; // the marker, or its loop
  Value index;         // the tile index in the marker's loop (or none)
  int64_t tiles;
};

// The trip count of an scf.for with constant bounds.
std::optional<int64_t> tripCount(scf::ForOp loop) {
  auto c = [](Value v) -> std::optional<int64_t> {
    if (auto k = v.getDefiningOp<arith::ConstantIndexOp>()) {
      return k.value();
    }
    return std::nullopt;
  };
  auto lb = c(loop.getLowerBound()), ub = c(loop.getUpperBound()),
       st = c(loop.getStep());
  if (!lb || !ub || !st || *st != 1) {
    return std::nullopt;
  }
  return *ub - *lb;
}

FailureOr<FullPartial> findFullPartial(func::FuncOp compute,
                                       ttk::LibttChunkAllReduceTileOp marker) {
  FullPartial p;
  for (Operation *op = marker->getPrevNode(); op && !p.copy;
       op = op->getPrevNode()) {
    if (auto c = dyn_cast<ttk::CopyTileOp>(op);
        c && sameIndex(c.getTileIndexDst(), marker.getTileIndex())) {
      p.copy = c;
    }
  }
  if (!p.copy) {
    return marker.emitError("libtt: chunk all-reduce without a CB copy");
  }
  Block &top = compute.getBody().front();
  p.markerAt = top.findAncestorOpInBlock(*marker);
  p.tiles = 1;
  if (p.markerAt != marker.getOperation()) {
    auto loop = dyn_cast<scf::ForOp>(p.markerAt);
    std::optional<int64_t> n = loop ? tripCount(loop) : std::nullopt;
    if (!n || p.copy.getTileIndexCb() != loop.getInductionVar()) {
      return marker.emitError("libtt: whole-tile chunk all-reduce not in a "
                              "loop over its tiles");
    }
    p.tiles = *n;
    p.index = loop.getInductionVar();
  }
  // The last pack into that CB before the copy.
  compute.walk([&](Operation *op) {
    if (op == p.copy) {
      return WalkResult::interrupt();
    }
    if (auto k = dyn_cast<ttk::PackTileOp>(op);
        k && k.getOutCb() == p.copy.getCb0()) {
      p.pack = k;
    }
    return WalkResult::advance();
  });
  if (!p.pack) {
    return marker.emitError("libtt: whole-tile chunk all-reduce: no pack");
  }
  p.packAt = top.findAncestorOpInBlock(*p.pack);
  auto packLoop = dyn_cast<scf::ForOp>(p.packAt);
  std::optional<int64_t> packTiles =
      p.packAt == p.pack.getOperation()
          ? std::optional<int64_t>(1)
          : packLoop ? tripCount(packLoop) : std::nullopt;
  if (!packTiles || *packTiles != p.tiles) {
    return marker.emitError("libtt: whole-tile chunk all-reduce: the "
                            "partial's packs and reads differ");
  }
  return p;
}

// The free DST slot for a section's extra tiles: one no tile op of the
// marker's section names (constant indices; FP32 DST holds 4 tiles in
// half-sync mode).
FailureOr<int64_t> freeSlot(ttk::LibttChunkAllReduceTileOp marker) {
  Operation *acquire = nullptr, *commit = nullptr;
  for (Operation *op = marker->getPrevNode(); op && !acquire;
       op = op->getPrevNode()) {
    if (isa<ttk::TileRegsAcquireOp>(op)) {
      acquire = op;
    }
  }
  for (Operation *op = marker->getNextNode(); op && !commit;
       op = op->getNextNode()) {
    if (isa<ttk::TileRegsCommitOp>(op)) {
      commit = op;
    }
  }
  if (!acquire || !commit) {
    return marker.emitError("libtt: chunk all-reduce outside a DST section");
  }
  llvm::SmallDenseSet<int64_t> used;
  for (Operation *op = acquire; op != commit; op = op->getNextNode()) {
    op->walk([&](Operation *nested) {
      for (Value v : nested->getOperands()) {
        if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
          used.insert(c.value());
        }
      }
    });
  }
  int64_t tmp = 1;
  while (used.contains(tmp)) {
    ++tmp;
  }
  if (tmp >= 4) {
    return marker.emitError("libtt: chunk all-reduce: no free DST slot");
  }
  return tmp;
}

// The partial's packs go to `send` (its tiles reserved and pushed around
// them); the reads take partial tile n from R's gather tiles n * g.. (waited
// for and popped around the reads), add the others and fold their rows
// where the marker was (libtt_sfpu::gather_fold).
LogicalResult rewriteFullCompute(ttk::LibttChunkAllReduceTileOp marker,
                                 FullPartial p, Value send, Value recv,
                                 int64_t g, int64_t slot) {
  Location loc = marker.getLoc();
  OpBuilder b(p.packAt);
  Value n32 = b.create<arith::ConstantIntOp>(loc, p.tiles, 32);
  b.create<ttk::CBReserveBackOp>(loc, send, n32);
  p.pack.getOutCbMutable().assign(send);
  b.setInsertionPointAfter(p.packAt);
  b.create<ttk::CBPushBackOp>(loc, send, n32);

  b.setInsertionPoint(p.markerAt);
  Value pages = b.create<arith::ConstantIntOp>(loc, p.tiles * g, 32);
  b.create<ttk::CBWaitFrontOp>(loc, recv, pages);
  b.setInsertionPointAfter(p.markerAt);
  b.create<ttk::CBPopFrontOp>(loc, recv, pages);

  Value old = p.copy.getCb0();
  for (Operation *op = p.copy->getPrevNode();
       op && !isa<ttk::TileRegsAcquireOp>(op); op = op->getPrevNode()) {
    if (isa<ttk::ReconfigDataFormatSrcAOp, ttk::CopyTileInitOp>(op) &&
        op->getOperand(0) == old) {
      op->setOperand(0, recv);
    }
  }
  b.setInsertionPoint(p.copy);
  Value first = b.create<arith::ConstantIndexOp>(loc, 0);
  if (p.index) {
    first = b.create<arith::MulIOp>(
        loc, p.index, b.create<arith::ConstantIndexOp>(loc, g));
  }
  p.copy.getCb0Mutable().assign(recv);
  p.copy.getTileIndexCbMutable().assign(first);

  Operation *at = marker->getPrevNode();
  if (!isa_and_nonnull<ttk::LibttChunkAllReduceTileInitOp>(at)) {
    at = marker;
  }
  if (g > 1) {
    FailureOr<int64_t> tmp = freeSlot(marker);
    if (failed(tmp)) {
      return failure();
    }
    b.setInsertionPoint(at);
    Value tmpIdx = b.create<arith::ConstantIndexOp>(loc, *tmp);
    for (int64_t i = 1; i < g; ++i) {
      Value tile = b.create<arith::AddIOp>(
          loc, first, b.create<arith::ConstantIndexOp>(loc, i));
      b.create<ttk::CopyTileInitOp>(loc, recv);
      b.create<ttk::CopyTileOp>(loc, recv, tile, tmpIdx);
      helperCall(b, at, "add_gather", {marker.getTileIndex(), tmpIdx}, 32);
    }
  }
  helperCall(b, marker, "gather_fold", marker.getTileIndex(), slot);
  if (at != marker) {
    at->erase();
  }
  marker->erase();
  return success();
}

// A whole-tile partial of a kernel unrolled and kept in DST
// (keepKernelInDst): its tiles packed one by one into CB X (pages 0..N-1,
// each at a section boundary) and read back by one marker each. Packs go to
// `send`, the reads to R's gather tiles n * g.. as in rewriteFullCompute;
// each release after a pack keeps DST.
//
// Two-level (inbox and red set; libtt_dm_chunk_allreduce_rows): before the
// first read, the core sums its inbox's g tiles (one tile row per core, of
// one row of one partial tile it reduces) and packs the column sums to red;
// the reads then take R's summed tiles as they are.
LogicalResult
rewriteUnrolledFull(func::FuncOp compute,
                    SmallVector<ttk::LibttChunkAllReduceTileOp> markers,
                    Value send, Value recv, int64_t g, int64_t slot,
                    Value inbox = nullptr, Value red = nullptr) {
  Location loc = markers.front()->getLoc();
  SmallVector<ttk::CopyTileOp> copies;
  for (ttk::LibttChunkAllReduceTileOp &marker : markers) {
    ttk::CopyTileOp copy;
    for (Operation *op = marker->getPrevNode(); op && !copy;
         op = op->getPrevNode()) {
      if (auto c = dyn_cast<ttk::CopyTileOp>(op);
          c && sameIndex(c.getTileIndexDst(), marker.getTileIndex())) {
        copy = c;
      }
    }
    if (!copy) {
      return marker.emitError("libtt: chunk all-reduce without a CB copy");
    }
    copies.push_back(copy);
  }
  Value x = copies.front().getCb0();
  const int64_t n = static_cast<int64_t>(markers.size());
  // The packs into X before the first read, by page.
  SmallVector<ttk::PackTileOp> packs(n);
  compute.walk([&](Operation *op) {
    if (op == copies.front()) {
      return WalkResult::interrupt();
    }
    if (auto k = dyn_cast<ttk::PackTileOp>(op); k && k.getOutCb() == x) {
      auto page = getConstantIntValue(k.getOutIndex());
      if (page && *page >= 0 && *page < n) {
        packs[*page] = k;
      }
    }
    return WalkResult::advance();
  });
  if (llvm::any_of(packs, [](ttk::PackTileOp k) { return !k; })) {
    return markers.front()->emitError(
        "libtt: whole-tile chunk all-reduce: a page not packed");
  }
  OpBuilder b(packs.front());
  Value n32 = b.create<arith::ConstantIntOp>(loc, n, 32);
  b.create<ttk::CBReserveBackOp>(loc, send, n32);
  for (ttk::PackTileOp k : packs) {
    k.getOutCbMutable().assign(send);
    for (Operation *op = k->getNextNode(); op; op = op->getNextNode()) {
      if (isa<ttk::TileRegsReleaseOp>(op)) {
        OpBuilder(op).create<emitc::CallOpaqueOp>(
            op->getLoc(), TypeRange{}, "libtt_tile_regs_release_keep",
            ValueRange{});
        op->erase();
        break;
      }
    }
  }
  b.setInsertionPointAfter(packs.back());
  b.create<ttk::CBPushBackOp>(loc, send, n32);
  b.setInsertionPoint(copies.front());
  if (inbox) {
    // The reduced row: in the first marker's slot (free until its read),
    // the inbox tiles added through the marker's spare slot.
    auto tmpAttr =
        markers.front()->getAttrOfType<IntegerAttr>("libtt.tmp_slot");
    FailureOr<int64_t> tmp = tmpAttr ? FailureOr<int64_t>(tmpAttr.getInt())
                                     : freeSlot(markers.front());
    if (failed(tmp)) {
      return failure();
    }
    Value acc = copies.front().getTileIndexDst();
    Value tmpIdx = b.create<arith::ConstantIndexOp>(loc, *tmp);
    Value gPages = b.create<arith::ConstantIntOp>(loc, g, 32);
    Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    b.create<ttk::CBWaitFrontOp>(loc, inbox, gPages);
    b.create<ttk::ReconfigDataFormatSrcAOp>(loc, inbox);
    b.create<ttk::CopyTileInitOp>(loc, inbox);
    b.create<ttk::CopyTileOp>(loc, inbox, zero, acc);
    for (int64_t i = 1; i < g; ++i) {
      b.create<ttk::CopyTileOp>(
          loc, inbox, b.create<arith::ConstantIndexOp>(loc, i), tmpIdx);
      helperCall(b, copies.front(), "add_gather", {acc, tmpIdx}, 32);
    }
    helperCall(b, copies.front(), "gather_fold", acc, 1);
    b.create<ttk::CBPopFrontOp>(loc, inbox, gPages);
    b.create<ttk::CBReserveBackOp>(loc, red, one);
    b.create<ttk::PackReconfigDataFormatOp>(loc, red);
    b.create<ttk::TileRegsCommitOp>(loc);
    b.create<ttk::TileRegsWaitOp>(loc);
    b.create<ttk::PackTileOp>(loc, acc, red, zero, b.getBoolAttr(true));
    b.create<emitc::CallOpaqueOp>(loc, TypeRange{},
                                  "libtt_tile_regs_release_keep", ValueRange{});
    b.create<ttk::TileRegsAcquireOp>(loc);
    b.create<ttk::CBPushBackOp>(loc, red, one);
  }
  Value pages = b.create<arith::ConstantIntOp>(loc, inbox ? n : n * g, 32);
  b.create<ttk::CBWaitFrontOp>(loc, recv, pages);
  b.setInsertionPointAfter(markers.back().getOperation());
  b.create<ttk::CBPopFrontOp>(loc, recv, pages);
  for (size_t m = 0; m < markers.size(); ++m) {
    ttk::LibttChunkAllReduceTileOp marker = markers[m];
    ttk::CopyTileOp copy = copies[m];
    auto page = getConstantIntValue(copy.getTileIndexCb());
    if (!page) {
      return marker.emitError("libtt: chunk all-reduce: a computed page");
    }
    for (Operation *op = copy->getPrevNode();
         op && !isa<ttk::TileRegsAcquireOp>(op); op = op->getPrevNode()) {
      if (isa<ttk::ReconfigDataFormatSrcAOp, ttk::CopyTileInitOp>(op) &&
          op->getOperand(0) == x) {
        op->setOperand(0, recv);
      }
    }
    b.setInsertionPoint(copy);
    copy.getCb0Mutable().assign(recv);
    copy.getTileIndexCbMutable().assign(
        b.create<arith::ConstantIndexOp>(loc, inbox ? *page : *page * g));
    Operation *at = marker->getPrevNode();
    if (!isa_and_nonnull<ttk::LibttChunkAllReduceTileInitOp>(at)) {
      at = marker;
    }
    if (inbox) {
      if (at != marker.getOperation()) {
        at->erase();
      }
      marker->erase();
      continue;
    }
    if (g > 1) {
      // The gather tiles' unpack setup: once (the adds are SFPU).
      b.setInsertionPoint(at);
      b.create<ttk::ReconfigDataFormatSrcAOp>(loc, recv);
      b.create<ttk::CopyTileInitOp>(loc, recv);
      auto tmpAttr = marker->getAttrOfType<IntegerAttr>("libtt.tmp_slot");
      FailureOr<int64_t> tmp =
          tmpAttr ? FailureOr<int64_t>(tmpAttr.getInt()) : freeSlot(marker);
      if (failed(tmp)) {
        return failure();
      }
      b.setInsertionPoint(at);
      Value tmpIdx = b.create<arith::ConstantIndexOp>(loc, *tmp);
      for (int64_t i = 1; i < g; ++i) {
        b.create<ttk::CopyTileOp>(
            loc, recv, b.create<arith::ConstantIndexOp>(loc, *page * g + i),
            tmpIdx);
        helperCall(b, at, "add_gather", {marker.getTileIndex(), tmpIdx}, 32);
      }
    }
    helperCall(b, marker, "gather_fold", marker.getTileIndex(), slot);
    if (at != marker.getOperation()) {
      at->erase();
    }
    marker->erase();
  }
  return success();
}

// The exchange in the reader (libtt_dm_chunk_allreduce), before the kernel's
// first wait for a compute result (its output writes come after the
// epilogue, which needs the exchange), else at its end.
void insertExchange(func::FuncOp reader, StringRef callee,
                    ArrayRef<Value> cbsAndSem, ttnn::CoreRangeAttr range,
                    ArrayRef<int64_t> extra, int64_t noc) {
  Block &body = reader.getBody().back();
  Operation *at = body.getTerminator();
  for (Operation &op : body) {
    if (isa<ttk::CBWaitFrontOp>(op)) {
      at = &op;
      break;
    }
  }
  OpBuilder b(at);
  Location loc = reader.getLoc();
  SmallVector<Value> args(cbsAndSem.begin(), cbsAndSem.end());
  SmallVector<int64_t> values{
      static_cast<int64_t>(range.getStartCoord().getX()),
      static_cast<int64_t>(range.getStartCoord().getY()),
      static_cast<int64_t>(range.getEndCoord().getX()),
      static_cast<int64_t>(range.getEndCoord().getY())};
  values.append(extra.begin(), extra.end());
  values.push_back(noc);
  for (int64_t v : values) {
    args.push_back(b.create<arith::ConstantIntOp>(loc, v, 32));
  }
  b.create<emitc::CallOpaqueOp>(loc, TypeRange{}, callee, args);
}

} // namespace

// The program's CBs no kernel reads any more (keepKernelInDst keeps their
// values in DST): not allocated. Wide-row kernels declared ~450 KB of CBs on
// 72 cores, most of them for values never in L1, past the L1 earlier programs
// used (where buffers allocated while a trace exists can sit).
void pruneUnusedCBs(ModuleOp module, SymbolTable &symbols) {
  MLIRContext *ctx = module.getContext();
  module.walk([&](ttnn::GenericOp generic) {
    auto program = generic->getAttrOfType<ttnn::ProgramAttr>("program");
    if (!program) {
      return;
    }
    llvm::DenseSet<uint32_t> used;
    for (Attribute kernel : program.getKernels()) {
      SymbolRefAttr symbol;
      ArrayRef<Attribute> rtArgs, ctArgs;
      if (auto c = dyn_cast<ttnn::ComputeKernelAttr>(kernel)) {
        symbol = c.getSymbolRef();
        rtArgs = c.getCommonRtArgs();
        ctArgs = c.getCtArgs();
      } else if (auto d = dyn_cast<ttnn::DataMovementKernelAttr>(kernel)) {
        symbol = d.getSymbolRef();
        rtArgs = d.getCommonRtArgs();
        ctArgs = d.getCtArgs();
      } else {
        return; // an unknown kernel kind: keep every CB
      }
      for (Attribute a : ctArgs) {
        if (auto cb = dyn_cast<ttnn::KernelArgCBBufferIndexAttr>(a)) {
          used.insert(cb.getBufferIndex());
        }
      }
      auto func = symbols.lookup<func::FuncOp>(symbol.getRootReference());
      if (!func) {
        return;
      }
      bool unknown = false;
      func.walk([&](ttk::GetCommonArgValOp get) {
        auto index = get.getArgIndex().getDefiningOp<arith::ConstantIndexOp>();
        if (!index || index.value() >= static_cast<int64_t>(rtArgs.size())) {
          unknown = true;
          return;
        }
        if (auto cb = dyn_cast<ttnn::KernelArgCBBufferIndexAttr>(
                rtArgs[index.value()]);
            cb && !get->use_empty()) {
          used.insert(cb.getBufferIndex());
        }
      });
      if (unknown) {
        return;
      }
    }
    SmallVector<ttnn::KernelCBAttr> cbs;
    for (ttnn::KernelCBAttr cb : program.getCbs()) {
      if (llvm::any_of(cb.getFormats(), [&](ttnn::KernelCBFormatAttr f) {
            return used.contains(f.getBufferIndex());
          })) {
        cbs.push_back(cb);
      }
    }
    if (cbs.size() != program.getCbs().size()) {
      generic->setAttr("program",
                       ttnn::ProgramAttr::get(ctx, program.getKernels(), cbs,
                                              program.getSemaphores()));
    }
  });
}

LogicalResult insertChunkAllReduce(ModuleOp module) {
  SymbolTable symbols(module);
  MLIRContext *ctx = module.getContext();
  LogicalResult result = success();
  module.walk([&](ttnn::GenericOp generic) {
    auto program = generic->getAttrOfType<ttnn::ProgramAttr>("program");
    if (!program) {
      return;
    }
    ttnn::ComputeKernelAttr compute;
    SmallVector<ttnn::DataMovementKernelAttr> movers;
    for (Attribute kernel : program.getKernels()) {
      if (auto c = dyn_cast<ttnn::ComputeKernelAttr>(kernel)) {
        compute = c;
      } else if (auto d = dyn_cast<ttnn::DataMovementKernelAttr>(kernel)) {
        movers.push_back(d);
      }
    }
    if (!compute) {
      return;
    }
    auto computeFunc =
        symbols.lookup<func::FuncOp>(compute.getSymbolRef().getRootReference());
    SmallVector<ttk::LibttChunkAllReduceTileOp> markers;
    computeFunc.walk(
        [&](ttk::LibttChunkAllReduceTileOp m) { markers.push_back(m); });
    if (markers.empty()) {
      return;
    }
    auto fail = [&](const Twine &why) {
      generic.emitError("libtt: chunk all-reduce: " + why);
      result = failure();
    };
    ArrayRef<ttnn::CoreRangeAttr> ranges =
        compute.getCoreRanges().getCoreRanges();
    if (ranges.size() != 1) {
      return fail("needs one core range");
    }
    ttnn::CoreRangeAttr range = ranges.front();
    const int64_t k =
        (range.getEndCoord().getX() - range.getStartCoord().getX() + 1) *
        (range.getEndCoord().getY() - range.getStartCoord().getY() + 1);
    // The reader: the data-movement kernel that fills compute's input CBs.
    ttnn::DataMovementKernelAttr reader;
    bool readerWaits = true;
    for (ttnn::DataMovementKernelAttr dm : movers) {
      auto f =
          symbols.lookup<func::FuncOp>(dm.getSymbolRef().getRootReference());
      bool pushes = false, waits = false;
      f.walk([&](ttk::CBPushBackOp) { pushes = true; });
      f.walk([&](ttk::CBWaitFrontOp) { waits = true; });
      if (pushes && (!reader || (readerWaits && !waits))) {
        reader = dm;
        readerWaits = waits;
      }
    }
    if (!reader) {
      return fail("no reader kernel");
    }
    auto readerFunc =
        symbols.lookup<func::FuncOp>(reader.getSymbolRef().getRootReference());
    // The rows the sums cover: the program's [.., T, N] tensors (T <= 32),
    // or more rows of its values (libtt.tile_rows: a partial with several
    // values per token, attention residuals).
    // (Operands of more rows are matmul weights.)
    int64_t rows = 1;
    for (Value v : generic->getOperands()) {
      if (auto t = dyn_cast<RankedTensorType>(v.getType());
          t && t.getRank() >= 2 && t.getDimSize(t.getRank() - 2) <= 32) {
        rows = std::max(rows, t.getDimSize(t.getRank() - 2));
      }
    }
    if (auto tileRows = computeFunc->getAttrOfType<IntegerAttr>("libtt.tile_rows")) {
      rows = std::max(rows, tileRows.getInt());
    }
    if (rows > 32) {
      return fail("rows over one tile");
    }
    uint32_t nextIndex = 0;
    for (ttnn::KernelCBAttr cb : program.getCbs()) {
      for (ttnn::KernelCBFormatAttr f : cb.getFormats()) {
        nextIndex = std::max<uint32_t>(nextIndex, f.getBufferIndex() + 1);
      }
    }
    // Gather slots: the partial's rows rounded up to a power of two columns,
    // 32 / slot cores per gather row (libtt_gather_slot).
    int64_t slot = 1;
    while (slot < rows) {
      slot *= 2;
    }
    const int64_t perRow = 32 / slot;
    const int64_t recvTiles = ((k + perRow - 1) / perRow + 31) / 32;
    const bool keepDst = computeFunc->hasAttr("libtt.dst_full_sync");
    auto cbArg = [&](uint32_t i) {
      return ttnn::KernelArgCBBufferIndexAttr::get(ctx, i);
    };
    auto i32 = IntegerType::get(ctx, 32);
    SmallVector<Attribute> computeArgs(compute.getCommonRtArgs().begin(),
                                       compute.getCommonRtArgs().end());
    SmallVector<Attribute> readerArgs(reader.getCommonRtArgs().begin(),
                                      reader.getCommonRtArgs().end());
    SmallVector<ttnn::ComputeKernelUnpackToDestMode> modes(
        compute.getUnpackToDestModes().begin(),
        compute.getUnpackToDestModes().end());
    SmallVector<ttnn::KernelCBAttr> cbs(program.getCbs().begin(),
                                        program.getCbs().end());
    SmallVector<ttnn::KernelSemaphoreAttr> sems(program.getSemaphores().begin(),
                                                program.getSemaphores().end());

    // Each marker in order: its partial (FP32 tiles) goes out through a new
    // CB P, the exchanged tiles come back in a new CB R, with a new
    // semaphore; the reader runs the exchanges in the same order. Row sums
    // (one tile, column 0) gather in slots of columns; whole tiles
    // (libtt.full_tile) in slots of rows: libtt_full_slot.
    // Whole-tile markers of an unrolled kernel (kept in DST), by the CB
    // their partial is read from: one exchange each.
    llvm::MapVector<Value, SmallVector<ttk::LibttChunkAllReduceTileOp>>
        unrolledFull;
    if (keepDst) {
      for (ttk::LibttChunkAllReduceTileOp marker : markers) {
        if (!marker->hasAttr("libtt.full_tile")) {
          continue;
        }
        for (Operation *op = marker->getPrevNode(); op; op = op->getPrevNode()) {
          if (auto c = dyn_cast<ttk::CopyTileOp>(op);
              c && sameIndex(c.getTileIndexDst(), marker.getTileIndex())) {
            unrolledFull[c.getCb0()].push_back(marker);
            break;
          }
        }
      }
    }
    // (The group's later markers are erased with its first: decided before.)
    llvm::DenseMap<Operation *, bool> fullTile;
    llvm::DenseSet<Operation *> later;
    for (ttk::LibttChunkAllReduceTileOp marker : markers) {
      fullTile[marker] = marker->hasAttr("libtt.full_tile");
    }
    for (auto &entry : unrolledFull) {
      for (auto marker : llvm::drop_begin(entry.second)) {
        later.insert(marker);
      }
    }
    for (ttk::LibttChunkAllReduceTileOp marker : markers) {
      if (later.contains(marker.getOperation())) {
        continue;
      }
      const bool full = fullTile[marker];
      FailureOr<Partial> partial;
      FailureOr<FullPartial> fullPartial;
      SmallVector<ttk::LibttChunkAllReduceTileOp> group;
      if (full && keepDst) {
        auto it = llvm::find_if(unrolledFull, [&](auto &entry) {
          return entry.second.front() == marker;
        });
        if (it == unrolledFull.end()) {
          return fail("a whole-tile partial not copied from a CB");
        }
        group = it->second;
      } else if (full) {
        fullPartial = findFullPartial(computeFunc, marker);
        if (failed(fullPartial)) {
          result = failure();
          return;
        }
      } else {
        partial = findPartial(computeFunc, marker);
        if (failed(partial)) {
          result = failure();
          return;
        }
      }
      ttk::CopyTileOp copy;
      if (!group.empty()) {
        for (Operation *op = marker->getPrevNode(); op && !copy;
             op = op->getPrevNode()) {
          if (auto c = dyn_cast<ttk::CopyTileOp>(op);
              c && sameIndex(c.getTileIndexDst(), marker.getTileIndex())) {
            copy = c;
          }
        }
      } else {
        copy = full ? fullPartial->copy : partial->copy;
      }
      auto xArg = argIndex(copy.getCb0());
      auto xBuffer = xArg ? dyn_cast<ttnn::KernelArgCBBufferIndexAttr>(
                                compute.getCommonRtArgs()[*xArg])
                          : nullptr;
      if (!xBuffer) {
        return fail("the partial's CB is not an argument");
      }
      ttnn::KernelCBFormatAttr xFormat;
      for (ttnn::KernelCBAttr cb : program.getCbs()) {
        for (ttnn::KernelCBFormatAttr f : cb.getFormats()) {
          if (f.getBufferIndex() == xBuffer.getBufferIndex()) {
            xFormat = f;
          }
        }
      }
      if (!xFormat || xFormat.getDtype() != ttcore::DataType::Float32 ||
          xFormat.getPageSize() != 4096) {
        return fail("the partial is not an FP32 tile");
      }
      // Whole tiles: rows per core rounded to a slot (1, or a power of two
      // of at least 4), 32 / slot cores per gather tile, g per partial tile.
      const int64_t sendTiles = !group.empty()
                                    ? static_cast<int64_t>(group.size())
                                    : full ? fullPartial->tiles
                                           : 1;
      const int64_t fullSlot = rows == 1 ? 1 : std::max<int64_t>(4, slot);
      // Partials of several rows (unrolled kernels): reduced in two levels,
      // each row of each partial tile on its own core (as one-row partials:
      // g tiles of one row per core) and the sums gathered on the root, which
      // multicasts only those. One level (every core summing every core's
      // rows) moved and added k * rows rows on every core: the whole cost of
      // a 12-row exchange on 72 cores (21 of 60 us).
      const bool twoLevel = !group.empty() && rows > 1 &&
                            rows * sendTiles <= k;
      const int64_t g = twoLevel ? (k + 31) / 32
                                 : (k + 32 / fullSlot - 1) / (32 / fullSlot);
      const int64_t markerRecvTiles =
          twoLevel ? sendTiles : full ? sendTiles * g : recvTiles;
      const uint32_t sendIndex = nextIndex++, recvIndex = nextIndex++;
      const uint32_t inboxIndex = twoLevel ? nextIndex++ : 0;
      const uint32_t redIndex = twoLevel ? nextIndex++ : 0;
      const size_t semId = sems.size();

      auto tile =
          mlir::cast<ttk::CBType>(copy.getCb0().getType()).getElementType();
      Value sendC = newArg(computeFunc, computeArgs.size(),
                           ttk::CBType::get(ctx, sendTiles, tile));
      Value recvC = newArg(computeFunc, computeArgs.size() + 1,
                           ttk::CBType::get(ctx, markerRecvTiles, tile));
      computeArgs.append({cbArg(sendIndex), cbArg(recvIndex)});
      Value inboxC, redC;
      if (twoLevel) {
        inboxC = newArg(computeFunc, computeArgs.size(),
                        ttk::CBType::get(ctx, g, tile));
        redC = newArg(computeFunc, computeArgs.size() + 1,
                      ttk::CBType::get(ctx, 1, tile));
        computeArgs.append({cbArg(inboxIndex), cbArg(redIndex)});
      }
      if (failed(!group.empty()
                     ? rewriteUnrolledFull(computeFunc, group, sendC, recvC, g,
                                           fullSlot, inboxC, redC)
                 : full ? rewriteFullCompute(marker, *fullPartial, sendC,
                                             recvC, g, fullSlot)
                        : rewriteCompute(computeFunc, marker, *partial, sendC,
                                         recvC, recvTiles, rows))) {
        result = failure();
        return;
      }
      // A kernel kept in DST keeps its values across the exchange: the
      // release after the partial's pack leaves DST as it is.
      if (keepDst && !full) {
        Operation *release = partial->pack->getNextNode();
        while (release && !isa<ttk::TileRegsReleaseOp>(release)) {
          release = release->getNextNode();
        }
        if (!release) {
          return fail("no release after the partial's pack");
        }
        OpBuilder(release).create<emitc::CallOpaqueOp>(
            release->getLoc(), TypeRange{}, "libtt_tile_regs_release_keep",
            ValueRange{});
        release->erase();
      }
      const int64_t base = readerArgs.size();
      SmallVector<Value> exchangeArgs{newArg(readerFunc, base, i32),
                                      newArg(readerFunc, base + 1, i32),
                                      newArg(readerFunc, base + 2, i32)};
      if (twoLevel) {
        exchangeArgs.append({newArg(readerFunc, base + 3, i32),
                             newArg(readerFunc, base + 4, i32),
                             newArg(readerFunc, base + 5, i32)});
        insertExchange(readerFunc, "libtt_dm_chunk_allreduce_rows",
                       exchangeArgs, range, {rows, sendTiles},
                       static_cast<int64_t>(reader.getNocIndex()));
      } else if (full) {
        insertExchange(readerFunc, "libtt_dm_chunk_allreduce_tiles",
                       exchangeArgs, range, {rows, sendTiles, g},
                       static_cast<int64_t>(reader.getNocIndex()));
      } else {
        insertExchange(readerFunc, "libtt_dm_chunk_allreduce", exchangeArgs,
                       range, {rows, recvTiles},
                       static_cast<int64_t>(reader.getNocIndex()));
      }
      if (twoLevel) {
        readerArgs.append({cbArg(sendIndex), cbArg(recvIndex),
                           cbArg(inboxIndex), cbArg(redIndex),
                           ttnn::KernelArgSemaphoreAtAttr::get(ctx, semId),
                           ttnn::KernelArgSemaphoreAtAttr::get(ctx, semId + 1)});
      } else {
        readerArgs.append({cbArg(sendIndex), cbArg(recvIndex),
                           ttnn::KernelArgSemaphoreAtAttr::get(ctx, semId)});
      }

      auto xMode = xBuffer.getBufferIndex() < modes.size()
                       ? modes[xBuffer.getBufferIndex()]
                       : ttnn::ComputeKernelUnpackToDestMode::Default;
      SmallVector<std::pair<uint32_t, int64_t>> newCbs{
          {sendIndex, sendTiles}, {recvIndex, markerRecvTiles}};
      if (twoLevel) {
        newCbs.append({{inboxIndex, g}, {redIndex, 1}});
      }
      for (auto [index, tiles] : newCbs) {
        while (modes.size() <= index) {
          modes.push_back(ttnn::ComputeKernelUnpackToDestMode::Default);
        }
        modes[index] = xMode;
      }
      for (auto [index, tiles] : newCbs) {
        cbs.push_back(ttnn::KernelCBAttr::get(
            ctx, tiles * 4096, compute.getCoreRanges(),
            {ttnn::KernelCBFormatAttr::get(ctx, index, xFormat.getDtype(),
                                           4096)},
            nullptr));
      }
      for (size_t i = 0; i < (twoLevel ? 2u : 1u); ++i) {
        sems.push_back(ttnn::KernelSemaphoreAttr::get(
            ctx, semId + i, ttnn::KernelCoreType::Worker,
            compute.getCoreRanges(), 0));
      }
    }

    // Program entry (libtt_dm_entry_signal/_wait): no exchange writes into a
    // core still running an earlier program.
    {
      const size_t entrySem = sems.size();
      sems.push_back(ttnn::KernelSemaphoreAttr::get(
          ctx, entrySem, ttnn::KernelCoreType::Worker, compute.getCoreRanges(),
          0));
      Value sem = newArg(readerFunc, readerArgs.size(), i32);
      readerArgs.push_back(ttnn::KernelArgSemaphoreAtAttr::get(ctx, entrySem));
      OpBuilder b(readerFunc.getContext());
      b.setInsertionPointAfter(sem.getDefiningOp());
      Location loc = readerFunc.getLoc();
      SmallVector<Value> args{sem};
      for (int64_t v :
           {static_cast<int64_t>(range.getStartCoord().getX()),
            static_cast<int64_t>(range.getStartCoord().getY()),
            static_cast<int64_t>(range.getEndCoord().getX()),
            static_cast<int64_t>(range.getEndCoord().getY()),
            static_cast<int64_t>(reader.getNocIndex())}) {
        args.push_back(b.create<arith::ConstantIntOp>(loc, v, 32));
      }
      b.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "libtt_dm_entry_signal",
                                    args);
      emitc::CallOpaqueOp first;
      readerFunc.walk([&](emitc::CallOpaqueOp call) {
        if (!first && call.getCallee().starts_with("libtt_dm_chunk_allreduce")) {
          first = call;
        }
      });
      if (!first) {
        return fail("no exchange in the reader");
      }
      b.setInsertionPoint(first);
      b.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "libtt_dm_entry_wait",
                                    args);
    }

    // The program: the new CBs, semaphores and kernel arguments.
    SmallVector<Attribute> kernels;
    for (Attribute kernel : program.getKernels()) {
      if (kernel == compute) {
        kernels.push_back(ttnn::ComputeKernelAttr::get(
            ctx, compute.getSymbolRef(), compute.getCoreRanges(),
            compute.getMathFidelity(), compute.getFp32DestAccEn(),
            compute.getDstFullSyncEn(), modes, compute.getBfp8PackPrecise(),
            compute.getMathApproxMode(), computeArgs, compute.getCtArgs()));
      } else if (kernel == reader) {
        kernels.push_back(ttnn::DataMovementKernelAttr::get(
            ctx, reader.getSymbolRef(), reader.getCoreRanges(),
            reader.getProcessor(), reader.getNocIndex(), reader.getNocMode(),
            readerArgs, reader.getCtArgs()));
      } else {
        kernels.push_back(kernel);
      }
    }
    generic->setAttr("program",
                     ttnn::ProgramAttr::get(ctx, kernels, cbs, sems));
  });
  if (succeeded(result)) {
    pruneUnusedCBs(module, symbols);
  }
  return result;
}

} // namespace mlir::tt::d2m_fusion
