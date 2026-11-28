// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// libtt's additions to TTNNCreateD2MSubgraphs (see ttmlir/D2MFusion.h): TTNN
// rewrites that let D2M take Inkling 2's per-head clips whole
// (prepareForOutlining), and the fixes that make a fusion group with head
// reshapes one the head-major rewrite in TTNN->TTIR takes (fixHeadGroups).

#include "ttmlir/D2MFusion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Support/Logger.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <map>
#include <optional>
#include <utility>

namespace mlir::tt::d2m_fusion {

// Tiles of a tensor in TTNN's tile layout (the last two dims padded).
int64_t tileCount(RankedTensorType t) {
  if (t.getRank() == 0) {
    return 1;
  }
  int64_t rank = t.getRank();
  int64_t rows = rank >= 2 ? (t.getDimSize(rank - 2) + 31) / 32 : 1;
  for (int64_t d = 0; d + 2 < rank; ++d) {
    rows *= t.getDimSize(d);
  }
  return rows * ((t.getDimSize(rank - 1) + 31) / 32);
}

// Head widths the head-major rewrite handles (multiples of a tile) or
// the group-major one (16: two heads per tile).
bool headDimSupported(int64_t d) { return d % 32 == 0 || d == 16; }

// The product of a tensor's leading dims (all but the last two).
static int64_t batchOf(RankedTensorType t) {
  int64_t b = 1;
  for (int64_t d = 0; d + 2 < t.getRank(); ++d) {
    b *= t.getDimSize(d);
  }
  return b;
}

std::optional<int64_t> wideNormChunks(Operation *op, RankedTensorType t) {
  int64_t rank = t.getRank();
  if (rank < 2 || t.getDimSize(rank - 2) > 32 || t.getDimSize(rank - 1) % 32) {
    return std::nullopt;
  }
  // Up to 8 rows of rows (splitWideRows unrolls the batch).
  if (batchOf(t) > 8) {
    return std::nullopt;
  }
  const int64_t tiles = t.getDimSize(rank - 1) / 32;
  if (tiles <= 32) {
    return std::nullopt;
  }
  // One chunk per core, the cores a rectangle of the worker grid.
  ttcore::DeviceAttr device = ttcore::lookupDevice(op);
  if (!device) {
    return std::nullopt;
  }
  ArrayRef<int64_t> grid = device.getWorkerGrid().getShape();
  auto fits = [&](int64_t k) {
    for (int64_t y = 1; y <= std::min(k, grid[0]); ++y) {
      if (k % y == 0 && k / y <= grid[1]) {
        return true;
      }
    }
    return false;
  };
  // The fewest tiles per core on at most 72 cores (2304 wide, T = 1, with
  // four-phase kernels: 72 cores 6.13 us, 36 cores 6.77).
  for (int64_t chunk = 1; chunk <= tiles; ++chunk) {
    if (tiles % chunk == 0 && tiles / chunk <= 72 && fits(tiles / chunk)) {
      return tiles / chunk;
    }
  }
  return std::nullopt;
}

std::optional<int64_t> attnResChunks(Operation *op) {
  auto attn = dyn_cast<ttnn::AttnResOp>(op);
  if (!attn || attn.getSources().size() > 8) {
    return std::nullopt;
  }
  auto source = cast<RankedTensorType>(attn.getSources().front().getType());
  auto queries = cast<RankedTensorType>(attn.getQueries().getType());
  for (Value v : attn.getSources()) {
    if (v.getType() != source) {
      return std::nullopt;
    }
  }
  if (source.getRank() != 2 || source.getDimSize(0) != 1 ||
      queries.getRank() != 2 || queries.getDimSize(0) != 1 ||
      queries.getDimSize(1) != source.getDimSize(1)) {
    return std::nullopt;
  }
  return wideNormChunks(op, source);
}

bool isSplitRowType(RankedTensorType t) {
  return t.getRank() >= 2 && batchOf(t) <= 8 &&
         t.getDimSize(t.getRank() - 2) <= 32 && t.getShape().back() % 32 == 0 &&
         t.getShape().back() / 32 > 32;
}

bool isSplitRowOp(Operation *op) {
  // A decode value [B.., T, X]: T <= 32, B <= 8, X whole tiles, FP32 or BF16.
  auto tiled = [](Type type) {
    auto t = dyn_cast<RankedTensorType>(type);
    return t && t.getRank() >= 2 && t.getShape().back() % 32 == 0 &&
           batchOf(t) <= 8 &&
           (t.getElementType().isF32() || t.getElementType().isBF16());
  };
  auto rows = [&](Type type) {
    auto t = dyn_cast<RankedTensorType>(type);
    return tiled(type) && t.getDimSize(t.getRank() - 2) <= 32;
  };
  auto wide = [](int64_t n) { return n / 32 > 32; };
  if (auto mm = dyn_cast<ttnn::MatmulOp>(op)) {
    // Low-rank decode projections: one of K, N a wide row, the other at most
    // 4 tiles, the weight's batch the activation's (or 1).
    auto a = mm.getA().getType(), b = mm.getB().getType();
    if (mm.getTransposeA() || mm.getTransposeB() || a.getRank() != b.getRank() ||
        !rows(a) || !tiled(b) || b.getDimSize(b.getRank() - 2) % 32 != 0 ||
        !rows(mm.getResult().getType()) ||
        (batchOf(a) != batchOf(b) && batchOf(a) != 1)) {
      return false;
    }
    int64_t k = a.getShape().back(), n = b.getShape().back();
    return (wide(k) && n <= 128) || (wide(n) && k <= 128);
  }
  if (auto concat = dyn_cast<ttnn::ConcatOp>(op)) {
    auto out = concat.getResult().getType();
    int64_t dim = concat.getDim();
    return (dim == -1 || dim == out.getRank() - 1) &&
           llvm::all_of(concat.getInputs(),
                        [&](Value v) { return rows(v.getType()); });
  }
  // Leading dims only: repeats, slices (steps 1) and reshapes keeping the
  // rows and columns.
  auto lastTwo = [](RankedTensorType a, RankedTensorType b) {
    return a.getRank() >= 2 && b.getRank() >= 2 &&
           a.getShape().take_back(2) == b.getShape().take_back(2);
  };
  if (auto repeat = dyn_cast<ttnn::RepeatOp>(op)) {
    auto dims = repeat.getRepeatDims().getShape();
    return rows(repeat.getResult().getType()) && dims.size() >= 2 &&
           dims.take_back(2) == ArrayRef<int64_t>{1, 1};
  }
  if (auto slice = dyn_cast<ttnn::SliceStaticOp>(op)) {
    auto st = extractFromIntegerArrayAttr<int32_t>(slice.getStep());
    return rows(slice.getInput().getType()) &&
           lastTwo(slice.getInput().getType(), slice.getResult().getType()) &&
           llvm::all_of(st, [](int32_t s) { return s == 1; });
  }
  // (And one row [1.., 1, N] as the vector [N].)
  if (auto reshape = dyn_cast<ttnn::ReshapeOp>(op)) {
    auto in = reshape.getInput().getType(), out = reshape.getResult().getType();
    if (isPromotionReshape(reshape) || !rows(in)) {
      return false;
    }
    if (out.getRank() == 1) {
      return in.getNumElements() == out.getNumElements() &&
             in.getShape().back() == out.getDimSize(0) && in.getShape().back() % 32 == 0;
    }
    return rows(out) && lastTwo(in, out) && in.getRank() != out.getRank();
  }
  return false;
}

// A slice of whole rows of x [T, N] (T <= 32) at tile-aligned columns:
// D2M reads it in place (a tile-offset view of x).
bool isTileColumnSlice(ttnn::SliceStaticOp s) {
  auto in = s.getInput().getType();
  if (in.getRank() != 2 || in.getDimSize(0) > 32) {
    return false;
  }
  auto b = extractFromIntegerArrayAttr<int32_t>(s.getBegins());
  auto e = extractFromIntegerArrayAttr<int32_t>(s.getEnds());
  auto st = extractFromIntegerArrayAttr<int32_t>(s.getStep());
  return st[0] == 1 && st[1] == 1 && b[0] == 0 && e[0] == in.getDimSize(0) &&
         b[1] % 32 == 0 && (e[1] % 32 == 0 || e[1] == in.getDimSize(1));
}

// Heads [h0, h1) sliced from a fused projection x [T, N] viewed as
// [..., N / D, D] (the leading dims flatten to the T rows; every other dim
// whole, steps 1). Returns h0 and h1.
static std::optional<std::pair<int64_t, int64_t>>
headSlice(ttnn::SliceStaticOp s) {
  auto r = s.getInput().getDefiningOp<ttnn::ReshapeOp>();
  if (!r) {
    return std::nullopt;
  }
  auto x = r.getInput().getType();
  auto v = r.getResult().getType();
  int64_t k = v.getRank();
  if (x.getRank() != 2 || k < 2 || x.getDimSize(0) > 32) {
    return std::nullopt;
  }
  int64_t T = x.getDimSize(0), N = x.getDimSize(1), D = v.getDimSize(k - 1);
  int64_t lead = 1;
  for (int64_t d = 0; d + 2 < k; ++d) {
    lead *= v.getDimSize(d);
  }
  if (N % D != 0 || v.getDimSize(k - 2) != N / D || lead != T) {
    return std::nullopt;
  }
  auto b = extractFromIntegerArrayAttr<int32_t>(s.getBegins());
  auto e = extractFromIntegerArrayAttr<int32_t>(s.getEnds());
  auto st = extractFromIntegerArrayAttr<int32_t>(s.getStep());
  for (int64_t d = 0; d < k; ++d) {
    if (st[d] != 1 || (d != k - 2 && (b[d] != 0 || e[d] != v.getDimSize(d)))) {
      return std::nullopt;
    }
  }
  int64_t h0 = b[k - 2], h1 = e[k - 2];
  if ((h0 * D) % 32 != 0 || ((h1 * D) % 32 != 0 && h1 * D != N)) {
    return std::nullopt;
  }
  return std::make_pair(h0, h1);
}

} // namespace mlir::tt::d2m_fusion

namespace mlir::tt::ttnn {
namespace {

using d2m_fusion::classifyHeadReshape;
using d2m_fusion::HeadDims;
using d2m_fusion::headDimSupported;
using d2m_fusion::headMerge;
using d2m_fusion::HeadReshape;
using d2m_fusion::headSlice;
using d2m_fusion::isHeadValue;
using d2m_fusion::isPromotionReshape;

// Whether the head-major (or group-major) rewrite in TTNN->TTIR
// (finishD2MSubgraphs) will take this group, by the same checks. A group it
// does not take must not keep head reshapes.
static bool headMajorApplicable(ArrayRef<Operation *> group,
                                const llvm::DenseSet<Operation *> &members) {
  SmallVector<Value> outs;
  for (Operation *op : group) {
    for (Value r : op->getResults()) {
      if (llvm::any_of(r.getUsers(),
                       [&](Operation *u) { return !members.contains(u); })) {
        outs.push_back(r);
      }
    }
  }
  if (outs.size() != 1) {
    return false;
  }
  auto merge = outs.front().getDefiningOp<ReshapeOp>();
  std::optional<HeadDims> dims =
      merge ? headMerge(merge.getInput().getType().getShape(),
                        merge.getResult().getType().getShape())
            : std::nullopt;
  if (!dims) {
    return false;
  }
  // Rank-3 values from outside keep TTNN's layout (heads as rows), not
  // the rewrite's: only single elements may come in that way.
  for (Operation *op : group) {
    for (Value v : op->getOperands()) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!members.contains(v.getDefiningOp()) && t && t.getRank() == 3 &&
          t.getNumElements() != 1) {
        return false;
      }
    }
  }
  bool split = false;
  for (Operation *op : group) {
    if (op == merge.getOperation()) {
      continue;
    }
    if (auto r = dyn_cast<ReshapeOp>(op)) {
      if (isPromotionReshape(r)) {
        continue;
      }
      // Norm weights are norm operands here, not reshapes.
      std::optional<HeadReshape> kind =
          classifyHeadReshape(r.getInput().getType().getShape(),
                              r.getResult().getType().getShape(), *dims);
      if (!kind || *kind == HeadReshape::Row) {
        return false;
      }
      split |= *kind == HeadReshape::Split;
      continue;
    }
    if (op->getNumResults() != 1 || op->getNumRegions() != 0 ||
        isa<WhereOp, MaxOp, MinOp>(op)) {
      return false;
    }
    // A norm's row weight serves every head only with whole-tile heads.
    if (dims->D % 32 != 0 &&
        (isa<RMSNormGatedOp>(op) ||
         (isa<RMSNormOp>(op) && cast<RMSNormOp>(op).getWeight()))) {
      return false;
    }
    auto t = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!t || t.getRank() > 3 ||
        (t.getRank() == 3 && !isHeadValue(t.getShape(), *dims))) {
      return false;
    }
  }
  return split;
}

// [T, H*D] -> [T, H, D] with T <= 32 and a supported head width (the
// head-major rewrite's split).
static bool isHeadSplit(ReshapeOp r) {
  auto in = r.getInput().getType();
  auto out = r.getResult().getType();
  return in.getRank() == 2 && out.getRank() == 3 &&
         out.getDimSize(0) == in.getDimSize(0) && out.getDimSize(0) <= 32 &&
         headDimSupported(out.getDimSize(2)) &&
         in.getDimSize(1) == out.getDimSize(1) * out.getDimSize(2);
}

// A group with a head split whose one result [T, H, D] leaves only
// through reshapes to other shapes (Inkling 2's q into the decode
// attention's [1, T, H, D]) merges it to [T, H * D] first, which makes the
// group head-major; the reshape from [T, H * D] stays on TTNN.
static void addHeadMerge(SmallVector<Operation *> &group) {
  llvm::DenseSet<Operation *> members(group.begin(), group.end());
  if (llvm::none_of(group, [](Operation *op) {
        auto r = dyn_cast<ReshapeOp>(op);
        return r && isHeadSplit(r);
      })) {
    return;
  }
  Value out;
  for (Operation *op : group) {
    for (Value r : op->getResults()) {
      if (llvm::any_of(r.getUsers(),
                       [&](Operation *u) { return !members.contains(u); })) {
        if (out) {
          return;
        }
        out = r;
      }
    }
  }
  auto t = out ? dyn_cast<RankedTensorType>(out.getType()) : nullptr;
  if (!t || t.getRank() != 3 || t.getDimSize(0) > 32 ||
      !headDimSupported(t.getDimSize(2))) {
    return;
  }
  int64_t T = t.getDimSize(0), HD = t.getDimSize(1) * t.getDimSize(2);
  SmallVector<ReshapeOp> users;
  for (Operation *u : out.getUsers()) {
    if (members.contains(u)) {
      continue;
    }
    auto r = dyn_cast<ReshapeOp>(u);
    if (!r || r.getType().getShape() == ArrayRef<int64_t>{T, HD}) {
      return;
    }
    users.push_back(r);
  }
  SmallVector<int64_t> shape{T, HD};
  auto mergedType = RankedTensorType::get(
      shape, t.getElementType(),
      TTNNLayoutAttr::Builder(cast<TTNNLayoutAttr>(t.getEncoding()), shape)
          .build());
  OpBuilder b(out.getContext());
  b.setInsertionPointAfterValue(out);
  auto merge = b.create<ReshapeOp>(
      out.getLoc(), mergedType, out,
      b.getI32ArrayAttr({static_cast<int32_t>(T), static_cast<int32_t>(HD)}));
  for (ReshapeOp u : users) {
    u.getInputMutable().assign(merge.getResult());
  }
  group.push_back(merge);
  members.insert(merge);
  if (!headMajorApplicable(group, members)) {
    for (ReshapeOp u : users) {
      u.getInputMutable().assign(out);
    }
    group.pop_back();
    merge.erase();
  }
}

// A per-head scale reshaped [H * D] -> [1, H, D] outside the group
// (the frontend reshapes Inkling 2's clip scales far ahead of their use)
// joins it as a copy, so the head-major rewrite sees the flat input.
// Returns each copy with its original (for undoing).
static SmallVector<std::pair<ReshapeOp, ReshapeOp>>
pullInScaleReshapes(SmallVector<Operation *> &group) {
  llvm::DenseSet<Operation *> members(group.begin(), group.end());
  SmallVector<std::pair<ReshapeOp, ReshapeOp>> copies;
  for (Operation *op : SmallVector<Operation *>(group)) {
    for (OpOperand &use : op->getOpOperands()) {
      auto r = use.get().getDefiningOp<ReshapeOp>();
      if (!r || members.contains(r)) {
        continue;
      }
      auto in = r.getInput().getType();
      auto out = r.getResult().getType();
      if (out.getRank() != 3 || out.getDimSize(0) != 1 ||
          !headDimSupported(out.getDimSize(2)) || in.getRank() > 2 ||
          in.getShape().back() != out.getDimSize(1) * out.getDimSize(2) ||
          in.getNumElements() != in.getShape().back()) {
        continue;
      }
      OpBuilder b(op);
      auto copy = cast<ReshapeOp>(b.clone(*r));
      use.set(copy.getResult());
      group.push_back(copy);
      members.insert(copy);
      copies.emplace_back(copy, r);
    }
  }
  return copies;
}

// Keeps pullInScaleReshapes' copies if the group is head-major (their
// originals go when unused), else puts the originals back.
static void
settleScaleReshapes(SmallVector<Operation *> &group,
                    ArrayRef<std::pair<ReshapeOp, ReshapeOp>> copies) {
  if (copies.empty()) {
    return;
  }
  llvm::DenseSet<Operation *> members(group.begin(), group.end());
  bool keep = headMajorApplicable(group, members);
  llvm::SetVector<Operation *> originals;
  for (auto [copy, original] : copies) {
    if (keep) {
      originals.insert(original);
      continue;
    }
    llvm::erase(group, copy.getOperation());
    copy.getResult().replaceAllUsesWith(original.getResult());
    copy.erase();
  }
  for (Operation *original : originals) {
    if (original->use_empty()) {
      original->erase();
    }
  }
}

// A slice is a view D2M reads in place; a subgraph cannot return a
// view (it returned the sliced tensor whole). A slice whose result leaves
// the group stays on TTNN, with a copy in the group for its users there.
static void keepEscapingSlicesOnTTNN(SmallVector<Operation *> &group) {
  llvm::DenseSet<Operation *> members(group.begin(), group.end());
  SmallVector<Operation *> kept;
  for (Operation *op : group) {
    auto slice = dyn_cast<SliceStaticOp>(op);
    if (!slice || llvm::all_of(slice->getUsers(), [&](Operation *u) {
          return members.contains(u);
        })) {
      kept.push_back(op);
      continue;
    }
    SmallVector<OpOperand *> inside;
    for (OpOperand &use : slice->getUses()) {
      if (members.contains(use.getOwner())) {
        inside.push_back(&use);
      }
    }
    if (inside.empty()) {
      continue;
    }
    OpBuilder b(slice);
    Operation *copy = b.clone(*slice);
    for (OpOperand *use : inside) {
      use->set(copy->getResult(0));
    }
    kept.push_back(copy);
  }
  group = std::move(kept);
  if (group.size() == 1 &&
      !isa<RMSNormOp, RMSNormGatedOp, AttnResOp>(group[0])) {
    group.clear();
  }
}

static bool isSplitAnchor(Operation *op);

// A split group's escaping value escaping reshaped too (a norm's [T, N]
// output and its [1, T, N]): the reshape stays on TTNN (a view there), as
// the split would return one buffer twice under two types.
static void dropReshapedOutputs(SmallVector<Operation *> &group) {
  if (llvm::none_of(group, isSplitAnchor)) {
    return;
  }
  llvm::DenseSet<Operation *> members(group.begin(), group.end());
  llvm::erase_if(group, [&](Operation *op) {
    auto r = dyn_cast<ReshapeOp>(op);
    if (!r || !members.contains(r.getInput().getDefiningOp()) ||
        r->use_empty()) {
      return false;
    }
    Operation *first = nullptr;
    for (Operation *u : r->getUsers()) {
      if (members.contains(u) || u->getBlock() != r->getBlock()) {
        return false;
      }
      if (!first || u->isBeforeInBlock(first)) {
        first = u;
      }
    }
    if (llvm::none_of(r.getInput().getUsers(), [&](Operation *u) {
          return u != op && !members.contains(u);
        })) {
      return false;
    }
    // After the outlined group (it precedes the reshape's readers).
    members.erase(op);
    op->moveBefore(first);
    return true;
  });
}

// A split group's repeat read outside too (Inkling 2's normed loop
// destination, shared across passes): a copy on TTNN for those readers, so
// the group returns the repeated value, not the repeat (rows of rows the
// split packs, not B tile rows).
static void cloneEscapingRepeats(SmallVector<Operation *> &group) {
  if (llvm::none_of(group, isSplitAnchor)) {
    return;
  }
  llvm::DenseSet<Operation *> members(group.begin(), group.end());
  for (Operation *op : group) {
    auto repeat = dyn_cast<RepeatOp>(op);
    if (!repeat) {
      continue;
    }
    SmallVector<OpOperand *> outside;
    Operation *first = nullptr;
    for (OpOperand &use : repeat->getUses()) {
      Operation *u = use.getOwner();
      if (members.contains(u)) {
        continue;
      }
      if (u->getBlock() != repeat->getBlock()) {
        outside.clear();
        break;
      }
      outside.push_back(&use);
      if (!first || u->isBeforeInBlock(first)) {
        first = u;
      }
    }
    if (outside.empty()) {
      continue;
    }
    OpBuilder b(first);
    Operation *copy = b.clone(*repeat);
    for (OpOperand *use : outside) {
      use->set(copy->getResult(0));
    }
  }
}

// A group with a head split whose one [T, H, D] result is joined
// by other escaping values (Inkling 2's first global layer computes the
// per-token tau that later layers use too) cannot be head-major. Those
// values, and the members only feeding them, stay on TTNN; the group reads
// them as inputs.
static void dropSideOutputs(SmallVector<Operation *> &group) {
  llvm::DenseSet<Operation *> members(group.begin(), group.end());
  if (llvm::none_of(group, [](Operation *op) {
        auto r = dyn_cast<ReshapeOp>(op);
        return r && isHeadSplit(r);
      })) {
    return;
  }
  int heads = 0;
  llvm::DenseSet<Operation *> removed;
  for (Operation *op : group) {
    for (Value r : op->getResults()) {
      if (llvm::none_of(r.getUsers(),
                        [&](Operation *u) { return !members.contains(u); })) {
        continue;
      }
      if (cast<RankedTensorType>(r.getType()).getRank() == 3) {
        ++heads;
      } else {
        removed.insert(op);
      }
    }
  }
  if (heads != 1 || removed.empty()) {
    return;
  }
  for (bool changed = true; changed;) {
    changed = false;
    for (Operation *op : group) {
      if (removed.contains(op) || op->use_empty()) {
        continue;
      }
      bool feedsRemoved = false, feedsKept = false;
      for (Operation *u : op->getUsers()) {
        feedsRemoved |= removed.contains(u);
        feedsKept |= members.contains(u) && !removed.contains(u);
      }
      if (feedsRemoved && !feedsKept) {
        removed.insert(op);
        changed = true;
      }
    }
  }
  llvm::erase_if(group, [&](Operation *op) { return removed.contains(op); });
}

// A group with head splits also holding a per-token scalar chain
// (Inkling 2's first global layer computes tau inside its q and rel groups,
// joining them into one) leaves that chain on TTNN and becomes one group
// per connected part (each can be head-major).
static SmallVector<SmallVector<Operation *>>
splitHeadGroup(const SmallVector<Operation *> &group) {
  if (llvm::none_of(group, [](Operation *op) {
        auto r = dyn_cast<ReshapeOp>(op);
        return r && isHeadSplit(r);
      })) {
    return {group};
  }
  auto single = [](Operation *op, bool flatOnly) {
    return op->getNumResults() > 0 &&
           llvm::all_of(op->getResultTypes(), [&](Type t) {
             auto r = dyn_cast<RankedTensorType>(t);
             return r && r.getNumElements() == 1 &&
                    (!flatOnly || r.getRank() <= 1);
           });
  };
  // The scalar chain ([1] values) stays on TTNN.
  SmallVector<Operation *> kept;
  for (Operation *op : group) {
    if (!single(op, /*flatOnly=*/true)) {
      kept.push_back(op);
    }
  }
  if (kept.size() == group.size()) {
    return {group};
  }
  // Parts: connected over values with more than one element; a
  // single-element op ([1] -> [1, 1, 1]) used by several parts is copied.
  llvm::DenseSet<Operation *> wide;
  for (Operation *op : kept) {
    if (!single(op, /*flatOnly=*/false)) {
      wide.insert(op);
    }
  }
  llvm::DenseMap<Operation *, int> part;
  int parts = 0;
  for (Operation *root : kept) {
    if (!wide.contains(root) || part.count(root)) {
      continue;
    }
    SmallVector<Operation *> work{root};
    part[root] = parts;
    while (!work.empty()) {
      Operation *op = work.pop_back_val();
      SmallVector<Operation *> next;
      for (Value v : op->getOperands()) {
        if (Operation *d = v.getDefiningOp(); d && wide.contains(d)) {
          next.push_back(d);
        }
      }
      for (Operation *u : op->getUsers()) {
        if (wide.contains(u)) {
          next.push_back(u);
        }
      }
      for (Operation *n : next) {
        if (part.try_emplace(n, parts).second) {
          work.push_back(n);
        }
      }
    }
    ++parts;
  }
  SmallVector<SmallVector<Operation *>> result(parts);
  for (Operation *op : kept) {
    if (wide.contains(op)) {
      result[part[op]].push_back(op);
      continue;
    }
    // A single-element op: a copy in every part using it; the original
    // stays outside for its other users (the per-token tau of later
    // layers) or goes.
    for (int k = 0; k < parts; ++k) {
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : op->getUses()) {
        auto it = part.find(use.getOwner());
        if (it != part.end() && it->second == k) {
          uses.push_back(&use);
        }
      }
      if (uses.empty()) {
        continue;
      }
      OpBuilder b(op->getContext());
      b.setInsertionPointAfter(op);
      Operation *mine = b.clone(*op);
      for (OpOperand *use : uses) {
        use->set(mine->getResult(0));
      }
      result[k].push_back(mine);
    }
    if (op->use_empty()) {
      op->erase();
    }
  }
  for (auto &r : result) {
    llvm::sort(
        r, [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
  }
  return result;
}

// Heads sliced from a fused projection become the column slice
// x[:, h0 * D : h1 * D] (read in place by D2M) and the head split, instead
// of a reshape of x and a slice of whole heads.
static void rewriteHeadSlices(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  SmallVector<SliceStaticOp> slices;
  moduleOp.walk([&](SliceStaticOp s) {
    if (headSlice(s)) {
      slices.push_back(s);
    }
  });
  for (SliceStaticOp s : slices) {
    auto [h0, h1] = *headSlice(s);
    auto r = s.getInput().getDefiningOp<ReshapeOp>();
    Value x = r.getInput();
    auto xType = cast<RankedTensorType>(x.getType());
    int64_t T = xType.getDimSize(0);
    int64_t D = r.getResult().getType().getShape().back();
    SmallVector<int64_t> shape{T, (h1 - h0) * D};
    auto colType = RankedTensorType::get(
        shape, xType.getElementType(),
        TTNNLayoutAttr::Builder(cast<TTNNLayoutAttr>(xType.getEncoding()),
                                shape)
            .build());
    rw.setInsertionPoint(s);
    auto cols = rw.create<SliceStaticOp>(
        s.getLoc(), colType, x,
        rw.getI32ArrayAttr({0, static_cast<int32_t>(h0 * D)}),
        rw.getI32ArrayAttr(
            {static_cast<int32_t>(T), static_cast<int32_t>(h1 * D)}),
        rw.getI32ArrayAttr({1, 1}));
    for (OpOperand &use : llvm::make_early_inc_range(s->getUses())) {
      if (auto user = dyn_cast<ReshapeOp>(use.getOwner())) {
        rw.modifyOpInPlace(user, [&] { use.set(cols.getResult()); });
      }
    }
    if (!s->use_empty()) {
      auto back =
          rw.create<ReshapeOp>(s.getLoc(), s.getType(), cols.getResult(),
                               rw.getI32ArrayAttr(llvm::to_vector_of<int32_t>(
                                   s.getType().getShape())));
      rw.replaceAllUsesWith(s.getResult(), back.getResult());
    }
    rw.eraseOp(s);
    if (r->use_empty()) {
      rw.eraseOp(r);
    }
  }
}

// One rows of rows sliced out of their flattening (x [B.., T, N] reshaped to
// [B.. T N] or [B.. T, N], the slice exactly x[b], as torch's unbind of
// Inkling 2's loop-controller outputs lowers): the leading-dim slice of x
// and a reshape, which D2M's row splits take as one pick of the packed rows.
static void rewriteRowsOfRowsSlices(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  SmallVector<std::pair<SliceStaticOp, SmallVector<int64_t>>> slices;
  moduleOp.walk([&](SliceStaticOp s) {
    auto r = s.getInput().getDefiningOp<ReshapeOp>();
    if (!r) {
      return;
    }
    auto x = cast<RankedTensorType>(r.getInput().getType());
    auto flat = r.getResult().getType();
    const int64_t n = x.getRank();
    if (n < 3 || !isa_and_nonnull<TTNNLayoutAttr>(x.getEncoding())) {
      return;
    }
    const int64_t T = x.getDimSize(n - 2), N = x.getDimSize(n - 1);
    // Decode rows only (what the row splits take; prefill keeps its form).
    if (T > 32) {
      return;
    }
    auto b = extractFromIntegerArrayAttr<int32_t>(s.getBegins());
    auto e = extractFromIntegerArrayAttr<int32_t>(s.getEnds());
    auto st = extractFromIntegerArrayAttr<int32_t>(s.getStep());
    if (llvm::any_of(st, [](int32_t v) { return v != 1; })) {
      return;
    }
    int64_t index;
    if (flat.getRank() == 1) {
      if (b[0] % (T * N) != 0 || e[0] - b[0] != T * N) {
        return;
      }
      index = b[0] / (T * N);
    } else if (flat.getRank() == 2 && flat.getDimSize(1) == N) {
      if (b[0] % T != 0 || e[0] - b[0] != T || b[1] != 0 || e[1] != N) {
        return;
      }
      index = b[0] / T;
    } else {
      return;
    }
    SmallVector<int64_t> coords(n - 2);
    for (int64_t d = n - 3; d >= 0; --d) {
      coords[d] = index % x.getDimSize(d);
      index /= x.getDimSize(d);
    }
    slices.push_back({s, coords});
  });
  for (auto &[s, coords] : slices) {
    auto r = s.getInput().getDefiningOp<ReshapeOp>();
    Value x = r.getInput();
    auto xType = cast<RankedTensorType>(x.getType());
    const int64_t n = xType.getRank();
    SmallVector<int64_t> shape(n - 2, 1);
    shape.append({xType.getDimSize(n - 2), xType.getDimSize(n - 1)});
    auto pickType = RankedTensorType::get(
        shape, xType.getElementType(),
        TTNNLayoutAttr::Builder(cast<TTNNLayoutAttr>(xType.getEncoding()),
                                shape)
            .build());
    SmallVector<int32_t> begins, ends;
    for (int64_t c : coords) {
      begins.push_back(static_cast<int32_t>(c));
      ends.push_back(static_cast<int32_t>(c + 1));
    }
    begins.append({0, 0});
    ends.append({static_cast<int32_t>(shape[n - 2]),
                 static_cast<int32_t>(shape[n - 1])});
    rw.setInsertionPoint(s);
    auto pick = rw.create<SliceStaticOp>(
        s.getLoc(), pickType, x, rw.getI32ArrayAttr(begins),
        rw.getI32ArrayAttr(ends),
        rw.getI32ArrayAttr(SmallVector<int32_t>(n, 1)));
    rw.replaceOpWithNewOp<ReshapeOp>(
        s, s.getType(), pick.getResult(),
        rw.getI32ArrayAttr(llvm::to_vector_of<int32_t>(s.getType().getShape())));
    if (r->use_empty()) {
      rw.eraseOp(r);
    }
  }
}

// An RMS norm over a vector [N] (Inkling 2's bs1 unbind flattens each
// query's mix): over the row [1, N] instead, read from the vector's row source
// when it is a reshape of one ([1.., 1, N]), and reshaped back. D2M takes
// norms over rows, so the row splits take it; a vector norm stayed one TTNN
// program per query.
static void rowNormsOfVectors(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  SmallVector<RMSNormOp> norms;
  moduleOp.walk([&](RMSNormOp n) {
    auto in = n.getInput().getType();
    if (in.getRank() == 1 && n.getResult().getType().getRank() == 1 &&
        !n.getBias() && isa_and_nonnull<TTNNLayoutAttr>(in.getEncoding())) {
      norms.push_back(n);
    }
  });
  auto withShape = [](RankedTensorType t, ArrayRef<int64_t> shape) {
    return RankedTensorType::get(
        shape, t.getElementType(),
        TTNNLayoutAttr::Builder(cast<TTNNLayoutAttr>(t.getEncoding()), shape)
            .build());
  };
  for (RMSNormOp n : norms) {
    const int64_t N = n.getInput().getType().getDimSize(0);
    rw.setInsertionPoint(n);
    Value x = n.getInput();
    if (auto r = x.getDefiningOp<ReshapeOp>()) {
      auto src = r.getInput().getType();
      if (src.getRank() >= 2 && src.getShape().back() == N &&
          src.getNumElements() == N) {
        x = r.getInput();
      }
    }
    auto xType = cast<RankedTensorType>(x.getType());
    if (xType.getRank() != 2) {
      auto rowType = withShape(xType, {1, N});
      x = rw.create<ReshapeOp>(n.getLoc(), rowType, x,
                               rw.getI32ArrayAttr({1, static_cast<int32_t>(N)}));
    }
    auto rowOut = withShape(n.getResult().getType(), {1, N});
    auto row = cast<RMSNormOp>(rw.clone(*n));
    row->setOperand(0, x);
    row.getResult().setType(rowOut);
    rw.replaceOpWithNewOp<ReshapeOp>(
        n, n.getResult().getType(), row.getResult(),
        rw.getI32ArrayAttr({static_cast<int32_t>(N)}));
  }
}

// A widening typecast whose only use is a reshape (Inkling 2's
// attention output before its gate) widens after the reshape instead, next
// to its consumer, where D2M takes it.
static void sinkWideningCasts(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  SmallVector<TypecastOp> casts;
  moduleOp.walk([&](TypecastOp castOp) {
    auto in = castOp.getInput().getType();
    auto out = castOp.getResult().getType();
    if (castOp->hasOneUse() && isa<ReshapeOp>(*castOp->getUsers().begin()) &&
        in.getElementType().isBF16() && out.getElementType().isF32() &&
        isa<TTNNLayoutAttr>(in.getEncoding())) {
      casts.push_back(castOp);
    }
  });
  for (TypecastOp castOp : casts) {
    auto r = cast<ReshapeOp>(*castOp->getUsers().begin());
    auto in = castOp.getInput().getType();
    auto out = r.getResult().getType();
    auto narrowType = RankedTensorType::get(
        out.getShape(), in.getElementType(),
        TTNNLayoutAttr::Builder(cast<TTNNLayoutAttr>(in.getEncoding()),
                                out.getShape())
            .build());
    rw.setInsertionPoint(r);
    Operation *reshape = rw.clone(*r);
    reshape->setOperand(0, castOp.getInput());
    reshape->getResult(0).setType(narrowType);
    Operation *widen = rw.clone(*castOp);
    widen->setOperand(0, reshape->getResult(0));
    widen->getResult(0).setType(out);
    rw.replaceOp(r, widen->getResults());
    rw.eraseOp(castOp);
  }
}

// A head split followed by a narrowing typecast (which stays on
// TTNN) narrows first, so the split joins the subgraph: reshape(x) ->
// typecast becomes typecast(x) -> reshape.
static void narrowBeforeHeadSplits(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  SmallVector<std::pair<ReshapeOp, TypecastOp>> swaps;
  moduleOp.walk([&](ReshapeOp r) {
    if (!isHeadSplit(r) || !r->hasOneUse()) {
      return;
    }
    auto castOp = dyn_cast<TypecastOp>(*r->getUsers().begin());
    if (castOp && castOp->getNumOperands() == 1 &&
        r.getInput().getType().getElementType().isF32() &&
        castOp.getResult().getType().getElementType().isBF16() &&
        isa<TTNNLayoutAttr>(castOp.getResult().getType().getEncoding())) {
      swaps.emplace_back(r, castOp);
    }
  });
  for (auto [r, castOp] : swaps) {
    auto in = r.getInput().getType();
    auto out = castOp.getResult().getType();
    auto narrowType = RankedTensorType::get(
        in.getShape(), out.getElementType(),
        TTNNLayoutAttr::Builder(cast<TTNNLayoutAttr>(out.getEncoding()),
                                in.getShape())
            .build());
    rw.setInsertionPoint(castOp);
    Operation *narrow = rw.clone(*castOp);
    narrow->setOperand(0, r.getInput());
    narrow->getResult(0).setType(narrowType);
    Operation *split = rw.clone(*r);
    split->setOperand(0, narrow->getResult(0));
    split->getResult(0).setType(out);
    rw.replaceOp(castOp, split->getResults());
    rw.eraseOp(r);
  }
}

// An FP32 binary op, a head merge, then a narrowing typecast becomes
// a BF16-result binary op and the merge (as TTNN does when the cast directly
// follows the op): the narrowing joins the subgraph instead of a separate
// TTNN typecast.
static void foldNarrowingCasts(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  SmallVector<TypecastOp> casts;
  moduleOp.walk([&](TypecastOp castOp) {
    auto merge = castOp->getOperand(0).getDefiningOp<ReshapeOp>();
    if (!merge || !merge->hasOneUse() ||
        merge.getInput().getType().getRank() != 3 ||
        !castOp.getResult().getType().getElementType().isBF16() ||
        !merge.getInput().getType().getElementType().isF32()) {
      return;
    }
    Operation *op = merge.getInput().getDefiningOp();
    if (op && op->hasOneUse() && isa<MultiplyOp, AddOp, SubtractOp>(op)) {
      casts.push_back(castOp);
    }
  });
  for (TypecastOp castOp : casts) {
    auto merge = castOp->getOperand(0).getDefiningOp<ReshapeOp>();
    Operation *op = merge.getInput().getDefiningOp();
    auto opType = cast<RankedTensorType>(op->getResult(0).getType());
    auto narrowType = RankedTensorType::get(
        opType.getShape(), castOp.getResult().getType().getElementType(),
        TTNNLayoutAttr::Builder(opType)
            .setElementType(castOp.getResult().getType().getElementType())
            .setLayout(cast<TTNNLayoutAttr>(opType.getEncoding()).getLayout())
            .build());
    rw.modifyOpInPlace(op, [&] { op->getResult(0).setType(narrowType); });
    rw.modifyOpInPlace(merge, [&] {
      merge.getResult().setType(castOp.getResult().getType());
    });
    rw.replaceOp(castOp, merge.getResult());
  }
}

// Each wide norm (d2m_fusion::wideNormChunks) or attention residual alone,
// a norm with the reshape of its weight: their in-kernel exchanges need the
// op's whole row-merged kernel. (Inkling 2 chains three 2304-wide norms
// in one group: split into programs, the exchange met a partial from DRAM.)
// The rest of the group stays a group.
// The ops that make a group one split over cores whole (splitWideRows): a
// low-rank matmul or a wide norm over several rows of rows.
static bool isSplitAnchor(Operation *op) {
  if (isa<MatmulOp>(op)) {
    return true;
  }
  if (isa<RMSNormOp, RMSNormGatedOp>(op)) {
    auto t = cast<RankedTensorType>(op->getOperand(0).getType());
    return d2m_fusion::wideNormChunks(op, t) && d2m_fusion::batchOf(t) > 1;
  }
  return false;
}

// Ops admitted only for splitting: the split-row ops and ops on batched wide
// rows past the outliner's L1 limit (d2m_fusion::isSplitRowType values; whole
// rows per core would not fit).
static bool splitOnly(Operation *op) {
  if (d2m_fusion::isSplitRowOp(op)) {
    return true;
  }
  return llvm::any_of(
      llvm::concat<Value>(op->getOperands(), op->getResults()), [](Value v) {
        auto t = dyn_cast<RankedTensorType>(v.getType());
        return t && t.getRank() > 2 && d2m_fusion::isSplitRowType(t) &&
               t.getDimSize(t.getRank() - 2) % 32 != 0 &&
               d2m_fusion::tileCount(t) > 128;
      });
}

// Without an anchor, the ops admitted only for splitting (splitOnly) leave a
// group at its boundary; a group with one inside is not outlined.
static void trimSplitOps(SmallVectorImpl<SmallVector<Operation *>> &groups) {
  for (auto &group : groups) {
    if (llvm::any_of(group, isSplitAnchor)) {
      continue;
    }
    llvm::SetVector<Operation *> members(group.begin(), group.end());
    for (bool changed = true; changed;) {
      changed = false;
      for (Operation *op : members.getArrayRef()) {
        if (!splitOnly(op)) {
          continue;
        }
        bool source = llvm::none_of(op->getOperands(), [&](Value v) {
          return members.contains(v.getDefiningOp());
        });
        bool sink = llvm::none_of(op->getUsers(), [&](Operation *u) {
          return members.contains(u);
        });
        if (source || sink) {
          members.remove(op);
          changed = true;
          break;
        }
      }
    }
    if (llvm::any_of(members, splitOnly)) {
      members.clear();
    }
    group = members.takeVector();
  }
}

// A group with an anchor splits its values over cores but returns only whole
// rows (batch 1): it takes in the consumers of its batched results (their
// groups, or ops admitted for splitting), if it stays schedulable (every
// input before every outside use of a result).
static void
absorbBatchedConsumers(SmallVectorImpl<SmallVector<Operation *>> &groups) {
  auto batched = [](Value v) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    return t && d2m_fusion::batchOf(t) > 1;
  };
  for (size_t g = 0; g < groups.size(); ++g) {
    if (!llvm::any_of(groups[g], isSplitAnchor)) {
      continue;
    }
    llvm::SetVector<Operation *> members(groups[g].begin(), groups[g].end());
    llvm::DenseSet<size_t> absorbed;
    bool stuck = false;
    for (bool changed = true; changed && !stuck;) {
      changed = false;
      for (Operation *op : members.getArrayRef()) {
        for (Value r : op->getResults()) {
          if (!batched(r)) {
            continue;
          }
          for (Operation *user : r.getUsers()) {
            if (members.contains(user)) {
              continue;
            }
            size_t other = groups.size();
            for (size_t h = 0; h < groups.size(); ++h) {
              if (h != g && !absorbed.contains(h) &&
                  llvm::is_contained(groups[h], user)) {
                other = h;
                break;
              }
            }
            if (other < groups.size()) {
              absorbed.insert(other);
              members.insert(groups[other].begin(), groups[other].end());
            } else if (d2m_fusion::isSplitRowOp(user)) {
              members.insert(user);
            } else {
              stuck = true;
              break;
            }
            changed = true;
            break;
          }
          if (changed || stuck) {
            break;
          }
        }
        if (changed || stuck) {
          break;
        }
      }
    }
    if (stuck || (absorbed.empty() && members.size() == groups[g].size())) {
      continue;
    }
    // Schedulable: the last input's def precedes every outside use.
    Block *block = groups[g].front()->getBlock();
    Operation *lastInput = nullptr;
    for (Operation *op : members) {
      for (Value v : op->getOperands()) {
        Operation *def = v.getDefiningOp();
        if (def && def->getBlock() == block && !members.contains(def) &&
            (!lastInput || lastInput->isBeforeInBlock(def))) {
          lastInput = def;
        }
      }
    }
    bool ok = llvm::all_of(members, [&](Operation *op) {
      return llvm::all_of(op->getUsers(), [&](Operation *user) {
        Operation *anc = block->findAncestorOpInBlock(*user);
        return !anc || members.contains(anc) || !lastInput ||
               lastInput->isBeforeInBlock(anc);
      });
    });
    if (!ok) {
      continue;
    }
    groups[g] = members.takeVector();
    for (size_t h : absorbed) {
      groups[h].clear();
    }
  }
  llvm::erase_if(groups, [](auto &group) { return group.empty(); });
}

static void isolateWideNorms(SmallVectorImpl<SmallVector<Operation *>> &groups) {
  SmallVector<SmallVector<Operation *>> out;
  for (auto &group : groups) {
    // A group split whole keeps its norms (one kernel, several exchanges)
    // but not its attention residuals (their own chunked kernels): cut at
    // each, in block order (the ops before one, it with its weight reshape,
    // the ops after; parts of a convex group in block order stay convex).
    if (llvm::any_of(group, isSplitAnchor)) {
      SmallVector<Operation *> sorted(group.begin(), group.end());
      llvm::sort(sorted, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      llvm::SmallPtrSet<Operation *, 4> weights;
      for (Operation *op : sorted) {
        if (d2m_fusion::attnResChunks(op) && op->getNumOperands() > 1) {
          auto weight = op->getOperand(1).getDefiningOp<ReshapeOp>();
          if (weight && llvm::is_contained(group, weight) &&
              weight->hasOneUse()) {
            weights.insert(weight);
          }
        }
      }
      SmallVector<Operation *> part;
      for (Operation *op : sorted) {
        if (weights.contains(op)) {
          continue;
        }
        if (!d2m_fusion::attnResChunks(op)) {
          part.push_back(op);
          continue;
        }
        if (!part.empty()) {
          out.push_back(std::exchange(part, {}));
        }
        SmallVector<Operation *> alone;
        if (auto weight = op->getOperand(1).getDefiningOp<ReshapeOp>();
            weight && weights.contains(weight)) {
          alone.push_back(weight);
        }
        alone.push_back(op);
        out.push_back(std::move(alone));
      }
      if (!part.empty()) {
        out.push_back(std::move(part));
      }
      continue;
    }
    llvm::SetVector<Operation *> rest(group.begin(), group.end());
    for (Operation *op : group) {
      bool wideNorm =
          isa<RMSNormOp, RMSNormGatedOp>(op) &&
          d2m_fusion::wideNormChunks(
              op, cast<RankedTensorType>(op->getOperand(0).getType()));
      if (!wideNorm && !d2m_fusion::attnResChunks(op)) {
        continue;
      }
      SmallVector<Operation *> alone;
      if (op->getNumOperands() > 1) {
        auto weight = op->getOperand(1).getDefiningOp<ReshapeOp>();
        if (weight && rest.contains(weight) && weight->hasOneUse()) {
          alone.push_back(weight);
          rest.remove(weight);
        }
      }
      alone.push_back(op);
      rest.remove(op);
      out.push_back(std::move(alone));
    }
    if (!rest.empty()) {
      out.push_back(rest.takeVector());
    }
  }
  groups.assign(out.begin(), out.end());
}

} // namespace
} // namespace mlir::tt::ttnn

namespace mlir::tt::d2m_fusion {

void prepareForOutlining(ModuleOp moduleOp) {
  ttnn::rewriteHeadSlices(moduleOp);
  ttnn::rewriteRowsOfRowsSlices(moduleOp);
  ttnn::rowNormsOfVectors(moduleOp);
  ttnn::sinkWideningCasts(moduleOp);
  ttnn::narrowBeforeHeadSplits(moduleOp);
  ttnn::foldNarrowingCasts(moduleOp);
}

void fixHeadGroups(SmallVectorImpl<SmallVector<Operation *>> &fusionGroups) {
  using namespace ttnn;
  absorbBatchedConsumers(fusionGroups);
  trimSplitOps(fusionGroups);
  isolateWideNorms(fusionGroups);
  // (Parts cut off a split group: split ops only at their boundaries.)
  trimSplitOps(fusionGroups);
  SmallVector<SmallVector<Operation *>> split;
  for (auto &group : fusionGroups) {
    for (auto &part : splitHeadGroup(group)) {
      // A composite norm or attention residual alone is worth a generic (the
      // outliner's rule).
      if (part.size() >= 2 ||
          (part.size() == 1 &&
           isa<RMSNormOp, RMSNormGatedOp, AttnResOp>(part[0]))) {
        split.push_back(std::move(part));
      }
    }
  }
  fusionGroups = std::move(split);
  // The head reshapes are views only after the head-major rewrite, which
  // needs the group's head split. Without one, boundary head reshapes stay
  // on TTNN; a group with one in the middle is not outlined.
  for (auto &group : fusionGroups) {
    keepEscapingSlicesOnTTNN(group);
    dropSideOutputs(group);
    dropReshapedOutputs(group);
    cloneEscapingRepeats(group);
    auto copies = pullInScaleReshapes(group);
    addHeadMerge(group);
    settleScaleReshapes(group, copies);
    // Outlining clones the group in order: keep it in block order.
    llvm::sort(group, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    // Its reshapes are of the rows of rows (splitWideRows), not heads.
    if (llvm::any_of(group, isSplitAnchor)) {
      continue;
    }
    llvm::DenseSet<Operation *> members(group.begin(), group.end());
    bool hasHeadReshape = llvm::any_of(group, [](Operation *op) {
      auto r = dyn_cast<ReshapeOp>(op);
      return r && !isPromotionReshape(r);
    });
    // A leading-dim broadcast of an operand with more than one row (a
    // [1, H, D] scale over [T, H, D]) is a view D2M only gets right after
    // the head-major rewrite (dims collapse past the per-index padding).
    bool slabBroadcast = llvm::any_of(group, [](Operation *op) {
      if (op->getNumResults() != 1) {
        return false;
      }
      auto out = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!out || out.getRank() < 3) {
        return false;
      }
      return llvm::any_of(op->getOperands(), [&](Value v) {
        auto in = dyn_cast<RankedTensorType>(v.getType());
        if (!in || in.getRank() < 2) {
          return false;
        }
        int64_t pad = out.getRank() - in.getRank();
        bool leading = false;
        for (int64_t d = 0; d + 2 < out.getRank(); ++d) {
          int64_t size = d < pad ? 1 : in.getDimSize(d - pad);
          leading |= size != out.getDimSize(d);
        }
        return leading && in.getDimSize(in.getRank() - 2) != 1;
      });
    });
    if (slabBroadcast && !headMajorApplicable(group, members)) {
      group.clear();
      continue;
    }
    if (!hasHeadReshape || headMajorApplicable(group, members)) {
      continue;
    }
    SmallVector<Operation *> kept;
    bool drop = false;
    for (Operation *op : group) {
      auto r = dyn_cast<ReshapeOp>(op);
      if (!r || isPromotionReshape(r)) {
        kept.push_back(op);
        continue;
      }
      bool source = llvm::none_of(op->getOperands(), [&](Value v) {
        return members.contains(v.getDefiningOp());
      });
      bool sink = llvm::none_of(
          op->getUsers(), [&](Operation *u) { return members.contains(u); });
      if (!source && !sink) {
        drop = true;
      }
    }
    if (drop) {
      group.clear();
    } else {
      group = kept;
    }
  }
}

} // namespace mlir::tt::d2m_fusion
