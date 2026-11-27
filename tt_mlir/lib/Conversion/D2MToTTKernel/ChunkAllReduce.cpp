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
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

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

// The exchange in the reader (libtt_dm_chunk_allreduce), before the kernel's
// first wait for a compute result (its output writes come after the
// epilogue, which needs the exchange), else at its end.
void insertExchange(func::FuncOp reader, ArrayRef<Value> cbsAndSem,
                    ttnn::CoreRangeAttr range, int64_t rows, int64_t recvTiles,
                    int64_t noc) {
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
  for (int64_t v : {static_cast<int64_t>(range.getStartCoord().getX()),
                    static_cast<int64_t>(range.getStartCoord().getY()),
                    static_cast<int64_t>(range.getEndCoord().getX()),
                    static_cast<int64_t>(range.getEndCoord().getY()), rows,
                    recvTiles, noc}) {
    args.push_back(b.create<arith::ConstantIntOp>(loc, v, 32));
  }
  b.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "libtt_dm_chunk_allreduce",
                                args);
}

} // namespace

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
    int64_t rows = 1;
    for (Value v : generic->getOperands()) {
      if (auto t = dyn_cast<RankedTensorType>(v.getType());
          t && t.getRank() >= 2) {
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

    // Each marker in order: its partial (an FP32 tile) goes out through a
    // new one-tile CB P, the exchanged tiles come back in a new CB R, with a
    // new semaphore; the reader runs the exchanges in the same order.
    for (ttk::LibttChunkAllReduceTileOp marker : markers) {
      FailureOr<Partial> partial = findPartial(computeFunc, marker);
      if (failed(partial)) {
        result = failure();
        return;
      }
      auto xArg = argIndex(partial->copy.getCb0());
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
      const uint32_t sendIndex = nextIndex++, recvIndex = nextIndex++;
      const size_t semId = sems.size();

      auto tile = mlir::cast<ttk::CBType>(partial->copy.getCb0().getType())
                      .getElementType();
      Value sendC = newArg(computeFunc, computeArgs.size(),
                           ttk::CBType::get(ctx, 1, tile));
      Value recvC = newArg(computeFunc, computeArgs.size() + 1,
                           ttk::CBType::get(ctx, recvTiles, tile));
      computeArgs.append({cbArg(sendIndex), cbArg(recvIndex)});
      if (failed(rewriteCompute(computeFunc, marker, *partial, sendC, recvC,
                                recvTiles, rows))) {
        result = failure();
        return;
      }
      // A kernel kept in DST keeps its values across the exchange: the
      // release after the partial's pack leaves DST as it is.
      if (keepDst) {
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
      insertExchange(
          readerFunc,
          {newArg(readerFunc, base, i32), newArg(readerFunc, base + 1, i32),
           newArg(readerFunc, base + 2, i32)},
          range, rows, recvTiles, static_cast<int64_t>(reader.getNocIndex()));
      readerArgs.append({cbArg(sendIndex), cbArg(recvIndex),
                         ttnn::KernelArgSemaphoreAtAttr::get(ctx, semId)});

      auto xMode = xBuffer.getBufferIndex() < modes.size()
                       ? modes[xBuffer.getBufferIndex()]
                       : ttnn::ComputeKernelUnpackToDestMode::Default;
      while (modes.size() <= recvIndex) {
        modes.push_back(ttnn::ComputeKernelUnpackToDestMode::Default);
      }
      modes[sendIndex] = xMode;
      modes[recvIndex] = xMode;
      for (auto [index, tiles] : {std::pair<uint32_t, int64_t>{sendIndex, 1},
                                  {recvIndex, recvTiles}}) {
        cbs.push_back(ttnn::KernelCBAttr::get(
            ctx, tiles * 4096, compute.getCoreRanges(),
            {ttnn::KernelCBFormatAttr::get(ctx, index, xFormat.getDtype(),
                                           4096)},
            nullptr));
      }
      sems.push_back(ttnn::KernelSemaphoreAttr::get(
          ctx, semId, ttnn::KernelCoreType::Worker, compute.getCoreRanges(),
          0));
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
  return result;
}

} // namespace mlir::tt::d2m_fusion
