// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// ttir-physical-layouts: tensors stored without tile padding.
//
// TTNN pads each slice of a tiled tensor's last two dims to whole 32 x 32
// tiles, so a tensor with a small dim among them takes many times its size. A
// program's values keep their logical shapes; this pass gives each value a
// physical layout, the shape its elements are stored in on device, and
// rebuilds the program's ops on those shapes:
//
// - U, unit dims dropped: [T, 1, V] as [T, V], [S, N, H, 1] as [S, N, H].
// - F, folded: n elements as [n / 32, 32], for rank-1 buffers and matrices of
//   a few rows.
//
// Both hold the elements in the same row-major order, so a change of layout is
// one reshape. Values connected through ops that run in a kind (its op rules,
// kRules) form a cluster, which takes the kind when the time its ops save
// exceeds the time of the reshapes at its edges (worthIt). Public functions'
// arguments and results of the shapes the runtime stores compactly
// (boundaryShape, and compactShape in tt_mlir_physical_layouts.patch) take
// those shapes.
//
// Fusions that match across reshapes form before this pass, as XLA forms its
// library calls before it assigns layouts: the rotary embedding as a composite
// (ttcore.composite), the decode attention prologue as
// ttir.split_qkv_norm_rope_decode, and the matmul epilogues (output scales,
// SwiGLU, activations) as ttir.fused_matmul (ttir-matmul-epilogues). They are
// edges, taking their operands in their logical shapes, and a composite's
// decomposition is left as it is.
//
// LIBTT_PHYSICAL_LAYOUTS=0 turns it off, here and in the runtime.
// LIBTT_PHYSICAL_LAYOUTS_MIN_MB (default 1) is the least a cluster must save.

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>

namespace mlir::tt::ttir {

namespace {

constexpr int64_t kTile = 32;
// The runtime's storage rule (compactShape): rank-1 arrays of at least 2^20
// elements and matrices of at most 8 rows and at least 2^22, folded; [T, 1, V]
// tensors of at least 2^20 elements without the unit dim.
constexpr int64_t kMinRank1Elements = int64_t{1} << 20;
constexpr int64_t kMaxRows = 8;
constexpr int64_t kMinMatrixElements = int64_t{1} << 22;

bool enabled() {
  const char *value = std::getenv("LIBTT_PHYSICAL_LAYOUTS");
  return value == nullptr || std::strcmp(value, "0") != 0;
}

double minSavedBytes() {
  const char *value = std::getenv("LIBTT_PHYSICAL_LAYOUTS_MIN_MB");
  double mb = value && *value ? std::atof(value) : 1.0;
  return mb * 1024.0 * 1024.0;
}

enum class Kind { U, F };

RankedTensorType typeOf(Value value) {
  return mlir::dyn_cast<RankedTensorType>(value.getType());
}

RankedTensorType withShape(RankedTensorType type, ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, type.getElementType(),
                               type.getEncoding());
}

int64_t product(ArrayRef<int64_t> shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    n *= d;
  }
  return n;
}

int64_t roundUp(int64_t value) { return (value + kTile - 1) / kTile * kTile; }

// Times are counted in the bytes DRAM moves in that time, about 400 GB/s,
// as measured with traced programs on one Blackhole chip:
//
// - An op takes the time to move what it reads and writes, but at least the
//   time to dispatch it: a traced program advances every 2 to 5 us whatever
//   its size, about 1 MB's time. Padding a decode step's [1, N] rows costs
//   nothing.
// - A reshape that moves tiles (one that changes the last two dims) moves
//   about 12 GB/s of the tensor's data, 1/32 of DRAM's rate, whatever the
//   padding: [256, 4096] f32 to [32768, 32] takes 350 us.
constexpr double kOpFloorBytes = 1024.0 * 1024.0;
constexpr double kReshapeSlowdown = 32;
// Tensors of at most 16 rows and at least 32768 columns, a sampler's rows,
// take 16 x 32 tiles (top_half_tiles.patch).
constexpr int64_t kHalfTileMaxRows = 16;
constexpr int64_t kHalfTileMinColumns = 32768;

double elementBytes(Type elementType) {
  return elementType.isIntOrFloat()
             ? std::max(1u, elementType.getIntOrFloatBitWidth() / 8)
             : 4;
}

// What TTNN allocates for a tiled tensor of `shape`: every slice of the last
// two dims padded to whole tiles.
double allocation(ArrayRef<int64_t> shape, Type elementType) {
  int64_t rows = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
  int64_t columns = shape.empty() ? 1 : shape.back();
  int64_t batches = shape.size() > 2 ? product(shape.drop_back(2)) : int64_t{1};
  int64_t paddedRows =
      rows <= kHalfTileMaxRows && columns >= kHalfTileMinColumns
          ? kHalfTileMaxRows
          : roundUp(rows);
  return static_cast<double>(batches) * paddedRows * roundUp(columns) *
         elementBytes(elementType);
}

// The time an op takes to read or write a tensor of `shape`.
double opTime(ArrayRef<int64_t> shape, Type elementType) {
  return std::max(kOpFloorBytes, allocation(shape, elementType));
}

// Whether a TTNN reshape between the shapes moves no data: the same last two
// dims, as a rank-1 tensor is one row.
bool sameTiles(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  auto lastTwo = [](ArrayRef<int64_t> s) -> std::pair<int64_t, int64_t> {
    if (s.empty()) {
      return {1, 1};
    }
    if (s.size() == 1) {
      return {1, s[0]};
    }
    return {s[s.size() - 2], s.back()};
  };
  return product(a) == product(b) && lastTwo(a) == lastTwo(b);
}

double reshapeCost(ArrayRef<int64_t> from, ArrayRef<int64_t> to,
                   Type elementType) {
  if (sameTiles(from, to)) {
    return 0;
  }
  return std::max(kOpFloorBytes,
                  kReshapeSlowdown * product(from) * elementBytes(elementType));
}

SmallVector<int64_t> withoutUnitDims(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> result;
  for (int64_t d : shape) {
    if (d != 1) {
      result.push_back(d);
    }
  }
  return result;
}

// Each dim's index among the non-unit dims, or -1.
SmallVector<int64_t> keptIndex(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> index;
  int64_t next = 0;
  for (int64_t d : shape) {
    index.push_back(d == 1 ? -1 : next++);
  }
  return index;
}

bool sameUnitDims(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  return a.size() == b.size() &&
         llvm::all_of(llvm::zip_equal(a, b), [](auto dims) {
           return (std::get<0>(dims) == 1) == (std::get<1>(dims) == 1);
         });
}

// `shape` without the dims where `like` has a unit dim.
SmallVector<int64_t> dropUnitsOf(ArrayRef<int64_t> shape,
                                 ArrayRef<int64_t> like) {
  SmallVector<int64_t> result;
  for (auto [d, l] : llvm::zip_equal(shape, like)) {
    if (l != 1) {
      result.push_back(d);
    }
  }
  return result;
}

// Whether a value of `type` can take kind `kind`.
bool eligible(Kind kind, RankedTensorType type) {
  if (!type || !type.hasStaticShape() || type.getNumElements() == 0) {
    return false;
  }
  if (kind == Kind::U) {
    return llvm::any_of(type.getShape(), [](int64_t d) { return d != 1; });
  }
  return type.getNumElements() % kTile == 0;
}

SmallVector<int64_t> physicalShape(Kind kind, RankedTensorType type) {
  if (kind == Kind::U) {
    return withoutUnitDims(type.getShape());
  }
  return {type.getNumElements() / kTile, kTile};
}

int64_t normalized(int64_t dim, int64_t rank) {
  return dim < 0 ? dim + rank : dim;
}

SmallVector<int64_t> intsOf(ArrayAttr array) {
  SmallVector<int64_t> values;
  for (Attribute a : array) {
    values.push_back(mlir::cast<IntegerAttr>(a).getInt());
  }
  return values;
}

ArrayAttr i32Array(Builder &b, ArrayRef<int64_t> values) {
  return b.getI32ArrayAttr(SmallVector<int32_t>(values.begin(), values.end()));
}

Value reshapeTo(OpBuilder &b, Location loc, Value input,
                ArrayRef<int64_t> shape) {
  RankedTensorType type = typeOf(input);
  if (type.getShape() == shape) {
    return input;
  }
  return b.create<ReshapeOp>(loc, withShape(type, shape), input,
                             i32Array(b, shape));
}

// A value whose reshapes fold away (ReshapeOp::fold): a constant or a fill.
bool isFree(Value value) {
  Operation *producer = value.getDefiningOp();
  if (auto constant = mlir::dyn_cast_or_null<ConstantOp>(producer)) {
    return mlir::isa<DenseElementsAttr>(constant.getValue());
  }
  return mlir::isa_and_nonnull<FullOp, ZerosOp, OnesOp>(producer);
}

template <typename... Ops>
bool isOneOf(Operation *op) {
  return mlir::isa<Ops...>(op);
}

// The dim that `op`'s integer attribute `name` names in a shape of `rank`.
int64_t dimOf(Operation *op, StringRef name, int64_t rank) {
  return normalized(
      op->getAttrOfType<IntegerAttr>(name).getValue().getSExtValue(), rank);
}

// `op`'s dim attribute `name`, a dim of `shape`, renumbered among the dims U
// keeps.
NamedAttribute keptDim(OpBuilder &b, Operation *op, StringRef name,
                       ArrayRef<int64_t> shape) {
  int64_t dim = keptIndex(shape)[dimOf(op, name, shape.size())];
  return b.getNamedAttr(
      name,
      IntegerAttr::get(op->getAttrOfType<IntegerAttr>(name).getType(), dim));
}

// For a static slice with unit steps, its begins and sizes.
std::optional<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
unitStepBounds(SliceStaticOp slice) {
  SmallVector<int64_t> begins = intsOf(slice.getBegins());
  SmallVector<int64_t> ends = intsOf(slice.getEnds());
  SmallVector<int64_t> steps = intsOf(slice.getStep());
  if (llvm::any_of(steps, [](int64_t s) { return s != 1; })) {
    return std::nullopt;
  }
  SmallVector<int64_t> sizes;
  for (auto [b, e] : llvm::zip_equal(begins, ends)) {
    sizes.push_back(e - b);
  }
  return std::make_pair(begins, sizes);
}

// Rows [begin, end) of a fold.
Value foldRows(OpBuilder &b, Location loc, Value fold, int64_t begin,
               int64_t end) {
  RankedTensorType type = typeOf(fold);
  if (begin == 0 && end == type.getDimSize(0)) {
    return fold;
  }
  return b.create<SliceStaticOp>(
      loc, withShape(type, {end - begin, kTile}), fold, i32Array(b, {begin, 0}),
      i32Array(b, {end, kTile}), i32Array(b, {1, 1}));
}

Value stackRows(OpBuilder &b, Location loc, RankedTensorType type,
                ArrayRef<Value> pieces) {
  if (pieces.size() == 1) {
    return pieces.front();
  }
  return b.create<ConcatOp>(loc, type, pieces, b.getSI32IntegerAttr(0));
}

// `op` on `operands`, with a result of `type` and the attributes in
// `changed` replaced.
Value create(OpBuilder &b, Operation *op, ValueRange operands,
             RankedTensorType type, ArrayRef<NamedAttribute> changed = {}) {
  NamedAttrList attrs(op->getAttrs());
  for (NamedAttribute attr : changed) {
    attrs.set(attr.getName(), attr.getValue());
  }
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       TypeRange{type}, attrs);
  return b.create(state)->getResult(0);
}

struct Rule;

// How an op runs in a kind: its rule, which operands are stored like its
// result (members), and the shape each other operand (an edge) takes.
struct Plan {
  const Rule *rule = nullptr;
  // Per operand: nullopt for a member, else the edge's shape.
  SmallVector<std::optional<SmallVector<int64_t>>> operands;

  bool member(unsigned index) const { return !operands[index].has_value(); }
};

// One rule per op family and kind: `isa` picks the ops, `match` decides
// whether one runs in `kind` and which operands are members, and `build`
// emits it on physical operands. planFor has checked that the operands are
// static tensors and that the result can take the kind; choose checks the
// members.
struct Rule {
  Kind kind;
  bool (*isa)(Operation *op);
  std::optional<Plan> (*match)(Operation *op);
  Value (*build)(OpBuilder &b, Operation *op, ValueRange operands,
                 RankedTensorType type);
};

// The first operand is a member; any others keep their logical shape.
std::optional<Plan> oneMember(Operation *op) {
  Plan plan;
  plan.operands.push_back(std::nullopt);
  for (Value operand : op->getOperands().drop_front()) {
    plan.operands.push_back(SmallVector<int64_t>(typeOf(operand).getShape()));
  }
  return plan;
}

// The first operand is a member if it has the result's unit dims.
std::optional<Plan> sameUnitDimsMember(Operation *op) {
  if (!sameUnitDims(typeOf(op->getOperand(0)).getShape(),
                    typeOf(op->getResult(0)).getShape())) {
    return std::nullopt;
  }
  return oneMember(op);
}

Value buildPlain(OpBuilder &b, Operation *op, ValueRange operands,
                 RankedTensorType type) {
  return create(b, op, operands, type);
}

// Elementwise ops and broadcasts: operands of the result's shape are members
// (constants are edges, whose reshapes fold away); `other` gives the other
// operands' shapes, broadcast against the result.
std::optional<Plan> elementwisePlan(
    Operation *op, Kind kind,
    llvm::function_ref<std::optional<SmallVector<int64_t>>(RankedTensorType)>
        other) {
  ArrayRef<int64_t> result = typeOf(op->getResult(0)).getShape();
  Plan plan;
  for (Value operand : op->getOperands()) {
    RankedTensorType type = typeOf(operand);
    if (type.getShape() == result) {
      if (isFree(operand)) {
        plan.operands.push_back(physicalShape(kind, type));
      } else {
        plan.operands.push_back(std::nullopt);
      }
      continue;
    }
    std::optional<SmallVector<int64_t>> shape = other(type);
    if (!shape) {
      return std::nullopt;
    }
    plan.operands.push_back(std::move(shape));
  }
  return plan;
}

// In U, broadcast operands are edges without the result's unit dims.
std::optional<Plan> matchElementwiseU(Operation *op) {
  ArrayRef<int64_t> result = typeOf(op->getResult(0)).getShape();
  return elementwisePlan(
      op, Kind::U,
      [&](RankedTensorType type) -> std::optional<SmallVector<int64_t>> {
        if (type.getRank() != static_cast<int64_t>(result.size())) {
          return std::nullopt;
        }
        return dropUnitsOf(type.getShape(), result);
      });
}

// In F, they have one element.
std::optional<Plan> matchElementwiseF(Operation *op) {
  return elementwisePlan(
      op, Kind::F,
      [](RankedTensorType type) -> std::optional<SmallVector<int64_t>> {
        if (type.getNumElements() != 1) {
          return std::nullopt;
        }
        return SmallVector<int64_t>{1, 1};
      });
}

Value buildBroadcastU(OpBuilder &b, Operation *op, ValueRange operands,
                      RankedTensorType type) {
  SmallVector<int64_t> dims =
      dropUnitsOf(mlir::cast<BroadcastOp>(op).getBroadcastDimensions(),
                  typeOf(op->getResult(0)).getShape());
  return create(
      b, op, operands, type,
      {b.getNamedAttr("broadcast_dimensions", b.getDenseI64ArrayAttr(dims))});
}

Value buildBroadcastF(OpBuilder &b, Operation *op, ValueRange operands,
                      RankedTensorType type) {
  return create(b, op, operands, type,
                {b.getNamedAttr("broadcast_dimensions",
                                b.getDenseI64ArrayAttr(type.getShape()))});
}

// A reduction runs in U on the reduced non-unit dims, renumbered, if the
// result's non-unit dims are the input's that remain: those dims.
std::optional<SmallVector<int64_t>> reducedUnitDims(Operation *op) {
  auto dimArg = op->getAttrOfType<ArrayAttr>("dim_arg");
  if (!dimArg) {
    return std::nullopt;
  }
  RankedTensorType input = typeOf(op->getOperand(0));
  int64_t rank = input.getRank();
  SmallVector<int64_t> kept = keptIndex(input.getShape());
  SmallVector<bool> reduced(rank, false);
  for (int64_t d : intsOf(dimArg)) {
    reduced[normalized(d, rank)] = true;
  }
  SmallVector<int64_t> dims, remaining;
  for (int64_t d = 0; d < rank; ++d) {
    if (input.getDimSize(d) == 1) {
      continue;
    }
    if (reduced[d]) {
      dims.push_back(kept[d]);
    } else {
      remaining.push_back(input.getDimSize(d));
    }
  }
  if (dims.empty() ||
      remaining != withoutUnitDims(typeOf(op->getResult(0)).getShape())) {
    return std::nullopt;
  }
  return dims;
}

std::optional<Plan> matchReductionU(Operation *op) {
  if (!reducedUnitDims(op)) {
    return std::nullopt;
  }
  return oneMember(op);
}

Value buildReductionU(OpBuilder &b, Operation *op, ValueRange operands,
                      RankedTensorType type) {
  return create(b, op, operands, type,
                {b.getNamedAttr("dim_arg", i32Array(b, *reducedUnitDims(op))),
                 b.getNamedAttr("keep_dim", b.getBoolAttr(false))});
}

// Softmax runs in U along its dim, renumbered, if that isn't a unit dim.
std::optional<Plan> matchSoftmaxU(Operation *op) {
  ArrayRef<int64_t> input = typeOf(op->getOperand(0)).getShape();
  if (input[dimOf(op, "dimension", input.size())] == 1) {
    return std::nullopt;
  }
  return oneMember(op);
}

Value buildSoftmaxU(OpBuilder &b, Operation *op, ValueRange operands,
                    RankedTensorType type) {
  return create(
      b, op, operands, type,
      {keptDim(b, op, "dimension", typeOf(op->getOperand(0)).getShape())});
}

// A permute runs in U on the non-unit dims, and disappears when they keep
// their order.
Value buildPermuteU(OpBuilder &b, Operation *op, ValueRange operands,
                    RankedTensorType type) {
  SmallVector<int64_t> kept = keptIndex(typeOf(op->getOperand(0)).getShape());
  SmallVector<int64_t> permutation;
  for (int64_t d : mlir::cast<PermuteOp>(op).getPermutation()) {
    if (kept[d] >= 0) {
      permutation.push_back(kept[d]);
    }
  }
  if (llvm::is_sorted(permutation)) {
    return operands[0];
  }
  return create(
      b, op, operands, type,
      {b.getNamedAttr("permutation", b.getDenseI64ArrayAttr(permutation))});
}

// All-gather and reduce-scatter run in U along their dim, renumbered; in F
// only on rank-1 arrays.
StringRef gatherScatterDim(Operation *op) {
  return mlir::isa<AllGatherOp>(op) ? "all_gather_dim" : "scatter_dim";
}

std::optional<Plan> matchGatherScatterU(Operation *op) {
  ArrayRef<int64_t> input = typeOf(op->getOperand(0)).getShape();
  if (input[dimOf(op, gatherScatterDim(op), input.size())] == 1) {
    return std::nullopt;
  }
  return sameUnitDimsMember(op);
}

std::optional<Plan> matchGatherScatterF(Operation *op) {
  if (typeOf(op->getOperand(0)).getRank() != 1 ||
      typeOf(op->getResult(0)).getRank() != 1) {
    return std::nullopt;
  }
  return oneMember(op);
}

Value buildGatherScatterU(OpBuilder &b, Operation *op, ValueRange operands,
                          RankedTensorType type) {
  return create(b, op, operands, type,
                {keptDim(b, op, gatherScatterDim(op),
                         typeOf(op->getOperand(0)).getShape())});
}

// Slices run in U if they make no unit dims; in F a slice is whole folded
// rows, or a range of 32-element blocks in each row.
Value buildSliceU(OpBuilder &b, Operation *op, ValueRange operands,
                  RankedTensorType type) {
  auto slice = mlir::cast<SliceStaticOp>(op);
  ArrayRef<int64_t> input = typeOf(slice.getInput()).getShape();
  auto kept = [&](ArrayAttr bounds) {
    return i32Array(b, dropUnitsOf(intsOf(bounds), input));
  };
  return create(b, op, operands, type,
                {b.getNamedAttr("begins", kept(slice.getBegins())),
                 b.getNamedAttr("ends", kept(slice.getEnds())),
                 b.getNamedAttr("step", kept(slice.getStep()))});
}

std::optional<Plan> matchSliceF(Operation *op) {
  RankedTensorType input = typeOf(op->getOperand(0));
  auto bounds = unitStepBounds(mlir::cast<SliceStaticOp>(op));
  if (!bounds || input.getRank() > 2) {
    return std::nullopt;
  }
  auto &[begins, sizes] = *bounds;
  if (input.getRank() == 1) {
    if (begins[0] % kTile != 0) {
      return std::nullopt;
    }
    return oneMember(op);
  }
  // Whole rows, or columns on 32-element boundaries of every row.
  int64_t columns = input.getDimSize(1);
  if (columns % kTile != 0) {
    return std::nullopt;
  }
  if (sizes[1] == columns ||
      (sizes[0] == input.getDimSize(0) && begins[1] % kTile == 0 &&
       sizes[1] % kTile == 0)) {
    return oneMember(op);
  }
  return std::nullopt;
}

Value buildSliceF(OpBuilder &b, Operation *op, ValueRange operands,
                  RankedTensorType type) {
  Location loc = op->getLoc();
  auto slice = mlir::cast<SliceStaticOp>(op);
  auto [begins, sizes] = *unitStepBounds(slice);
  RankedTensorType input = typeOf(slice.getInput());
  if (input.getRank() == 1) {
    return foldRows(b, loc, operands[0], begins[0] / kTile,
                    (begins[0] + sizes[0]) / kTile);
  }
  // A row is columns / 32 rows of the fold.
  int64_t rowBlocks = input.getDimSize(1) / kTile;
  if (sizes[1] == input.getDimSize(1)) {
    return foldRows(b, loc, operands[0], begins[0] * rowBlocks,
                    (begins[0] + sizes[0]) * rowBlocks);
  }
  SmallVector<Value> pieces;
  for (int64_t row = 0; row < sizes[0]; ++row) {
    int64_t first = (begins[0] + row) * rowBlocks + begins[1] / kTile;
    pieces.push_back(
        foldRows(b, loc, operands[0], first, first + sizes[1] / kTile));
  }
  return stackRows(b, loc, type, pieces);
}

// Concats run in U along their dim, renumbered, if the inputs and the result
// have the same unit dims; in F they stack whole folded rows, side by side in
// each row for column concats.
std::optional<Plan> matchConcatU(Operation *op) {
  ArrayRef<int64_t> result = typeOf(op->getResult(0)).getShape();
  if (result[dimOf(op, "dim", result.size())] == 1) {
    return std::nullopt;
  }
  Plan plan;
  for (Value operand : op->getOperands()) {
    RankedTensorType type = typeOf(operand);
    if (!sameUnitDims(type.getShape(), result)) {
      return std::nullopt;
    }
    if (isFree(operand)) {
      plan.operands.push_back(physicalShape(Kind::U, type));
    } else {
      plan.operands.push_back(std::nullopt);
    }
  }
  return plan;
}

std::optional<Plan> matchConcatF(Operation *op) {
  RankedTensorType result = typeOf(op->getResult(0));
  if (result.getRank() > 2) {
    return std::nullopt;
  }
  Plan plan;
  for (Value operand : op->getOperands()) {
    RankedTensorType type = typeOf(operand);
    // Rank 1, or matrices whose rows are whole rows of the fold: stacked, or
    // side by side (each input's blocks of a row in turn).
    if (!eligible(Kind::F, type) || type.getRank() != result.getRank() ||
        (type.getRank() == 2 && type.getDimSize(1) % kTile != 0)) {
      return std::nullopt;
    }
    if (isFree(operand)) {
      plan.operands.push_back(physicalShape(Kind::F, type));
    } else {
      plan.operands.push_back(std::nullopt);
    }
  }
  return plan;
}

Value buildConcatU(OpBuilder &b, Operation *op, ValueRange operands,
                   RankedTensorType type) {
  return create(b, op, operands, type,
                {keptDim(b, op, "dim", typeOf(op->getResult(0)).getShape())});
}

Value buildConcatF(OpBuilder &b, Operation *op, ValueRange operands,
                   RankedTensorType type) {
  Location loc = op->getLoc();
  RankedTensorType result = typeOf(op->getResult(0));
  if (result.getRank() == 2 && dimOf(op, "dim", 2) == 1) {
    SmallVector<Value> pieces;
    for (int64_t row = 0; row < result.getDimSize(0); ++row) {
      for (auto [input, fold] : llvm::zip_equal(op->getOperands(), operands)) {
        int64_t rowBlocks = typeOf(input).getDimSize(1) / kTile;
        pieces.push_back(
            foldRows(b, loc, fold, row * rowBlocks, (row + 1) * rowBlocks));
      }
    }
    return stackRows(b, loc, type, pieces);
  }
  return create(b, op, operands, type,
                {b.getNamedAttr("dim", b.getSI32IntegerAttr(0))});
}

// Norms run in U when their normalized dims aren't unit dims.
std::optional<Plan> matchNormU(Operation *op) {
  ArrayRef<int64_t> input = typeOf(op->getOperand(0)).getShape();
  auto normalizedShape =
      op->getAttrOfType<DenseI64ArrayAttr>("normalized_shape");
  if (!normalizedShape ||
      llvm::is_contained(input.take_back(normalizedShape.size()), 1)) {
    return std::nullopt;
  }
  return oneMember(op);
}

const Rule kRules[] = {
    {Kind::U, isOneOf<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>,
     matchElementwiseU, buildPlain},
    {Kind::U, isOneOf<BroadcastOp>, matchElementwiseU, buildBroadcastU},
    {Kind::U,
     isOneOf<SumOp, MeanOp, MaxOp, MinOp, ProdOp, ArgMaxOp, ReduceAndOp,
             ReduceOrOp>,
     matchReductionU, buildReductionU},
    {Kind::U, isOneOf<SoftmaxOp>, matchSoftmaxU, buildSoftmaxU},
    {Kind::U, isOneOf<PermuteOp>, oneMember, buildPermuteU},
    {Kind::U, isOneOf<AllReduceOp>, sameUnitDimsMember, buildPlain},
    {Kind::U, isOneOf<AllGatherOp, ReduceScatterOp>, matchGatherScatterU,
     buildGatherScatterU},
    {Kind::U, isOneOf<SliceStaticOp>, sameUnitDimsMember, buildSliceU},
    {Kind::U, isOneOf<ConcatOp>, matchConcatU, buildConcatU},
    {Kind::U, isOneOf<RMSNormOp, LayerNormOp>, matchNormU, buildPlain},
    {Kind::F, isOneOf<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>,
     matchElementwiseF, buildPlain},
    {Kind::F, isOneOf<BroadcastOp>, matchElementwiseF, buildBroadcastF},
    {Kind::F, isOneOf<AllReduceOp>, oneMember, buildPlain},
    {Kind::F, isOneOf<AllGatherOp, ReduceScatterOp>, matchGatherScatterF,
     buildPlain},
    {Kind::F, isOneOf<SliceStaticOp>, matchSliceF, buildSliceF},
    {Kind::F, isOneOf<ConcatOp>, matchConcatF, buildConcatF},
};

// How `op` runs in `kind`, if it can.
std::optional<Plan> planFor(Operation *op, Kind kind) {
  const Rule *rule = llvm::find_if(kRules, [&](const Rule &rule) {
    return rule.kind == kind && rule.isa(op);
  });
  if (rule == std::end(kRules) || !eligible(kind, typeOf(op->getResult(0))) ||
      !llvm::all_of(op->getOperandTypes(), [](Type type) {
        auto tensor = mlir::dyn_cast<RankedTensorType>(type);
        return tensor && tensor.hasStaticShape();
      })) {
    return std::nullopt;
  }
  std::optional<Plan> plan = rule->match(op);
  if (plan) {
    plan->rule = rule;
  }
  return plan;
}

// The shape a public function's argument or result of `type` is stored in, if
// not its own: the runtime's rule (compactShape).
std::optional<SmallVector<int64_t>> boundaryShape(RankedTensorType type) {
  if (!type || !type.hasStaticShape() || type.getNumElements() == 0) {
    return std::nullopt;
  }
  int64_t n = type.getNumElements();
  if (n % kTile == 0 &&
      ((type.getRank() == 1 && n >= kMinRank1Elements) ||
       (type.getRank() == 2 && type.getDimSize(0) <= kMaxRows &&
        type.getDimSize(1) != kTile && n >= kMinMatrixElements))) {
    return physicalShape(Kind::F, type);
  }
  if (type.getRank() == 3 && type.getDimSize(1) == 1 &&
      type.getDimSize(0) > 1 && type.getDimSize(2) > 1 &&
      n >= kMinRank1Elements) {
    return physicalShape(Kind::U, type);
  }
  return std::nullopt;
}

// Whether a boundary's tensor is the device's own (unsharded, or presharded
// with its local shape): a mesh_shard would cut it otherwise.
bool wholeOnDevice(ttcore::LocalShapeAttr local, RankedTensorType type) {
  return !local || local.getLocalShape().getShape() == type.getShape();
}

// Gives a public function's arguments and results the runtime's storage
// shapes, each with a reshape to or from its logical shape.
bool viewBoundaries(func::FuncOp func) {
  MLIRContext *context = func.getContext();
  StringRef localShape = ttcore::LocalShapeAttr::name;
  OpBuilder builder(context);
  bool viewed = false;
  Block &entry = func.getBody().front();

  builder.setInsertionPointToStart(&entry);
  for (BlockArgument arg : entry.getArguments()) {
    unsigned index = arg.getArgNumber();
    RankedTensorType type = typeOf(arg);
    std::optional<SmallVector<int64_t>> shape = boundaryShape(type);
    auto local =
        func.getArgAttrOfType<ttcore::LocalShapeAttr>(index, localShape);
    if (!shape || !wholeOnDevice(local, type) ||
        llvm::any_of(arg.getUsers(), llvm::IsaPred<MeshShardOp>)) {
      continue;
    }
    viewed = true;
    RankedTensorType stored = withShape(type, *shape);
    arg.setType(stored);
    Value view = reshapeTo(builder, func.getLoc(), arg, type.getShape());
    arg.replaceAllUsesExcept(view, view.getDefiningOp());
    if (local) {
      func.setArgAttr(index, localShape,
                      ttcore::LocalShapeAttr::get(context, stored));
    }
  }

  auto ret = mlir::cast<func::ReturnOp>(entry.getTerminator());
  builder.setInsertionPoint(ret);
  for (OpOperand &operand : ret->getOpOperands()) {
    unsigned index = operand.getOperandNumber();
    RankedTensorType type = typeOf(operand.get());
    std::optional<SmallVector<int64_t>> shape = boundaryShape(type);
    auto local =
        func.getResultAttrOfType<ttcore::LocalShapeAttr>(index, localShape);
    if (!shape || !wholeOnDevice(local, type) ||
        operand.get().getDefiningOp<MeshShardOp>()) {
      continue;
    }
    viewed = true;
    operand.set(reshapeTo(builder, ret.getLoc(), operand.get(), *shape));
    if (local) {
      func.setResultAttr(
          index, localShape,
          ttcore::LocalShapeAttr::get(context, typeOf(operand.get())));
    }
  }
  func.setType(FunctionType::get(context, entry.getArgumentTypes(),
                                 ret.getOperandTypes()));
  return viewed;
}

// Layouts chosen for one function's values, and how to rebuild it.
class Layouts {
public:
  explicit Layouts(func::FuncOp func) : func(func) {}

  // Chooses clusters of `kind` among the values no earlier kind took.
  void choose(Kind kind) {
    Block &entry = func.getBody().front();
    llvm::EquivalenceClasses<Value> clusters;
    llvm::DenseMap<Operation *, Plan> plans;
    for (Operation &op : entry.without_terminator()) {
      std::optional<Plan> plan = planFor(&op, kind);
      if (!plan) {
        continue;
      }
      SmallVector<Value> members = {op.getResult(0)};
      for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
        if (plan->member(i)) {
          members.push_back(operand);
        }
      }
      if (llvm::any_of(members,
                       [&](Value v) {
                         return kinds.count(v) || !eligible(kind, typeOf(v));
                       }) ||
          llvm::none_of(members, [&](Value v) {
            return ArrayRef<int64_t>(physicalShape(kind, typeOf(v))) !=
                   typeOf(v).getShape();
          })) {
        continue;
      }
      for (Value member : members) {
        clusters.unionSets(members.front(), member);
      }
      plans[&op] = std::move(*plan);
    }

    // Each cluster, in the order of its first value.
    llvm::DenseSet<Value> decided;
    for (const auto *first : clusters) {
      Value leader = clusters.getLeaderValue(first->getData());
      if (!decided.insert(leader).second) {
        continue;
      }
      SmallVector<Value> values = llvm::to_vector(clusters.members(leader));
      if (worthIt(kind, values, plans)) {
        for (Value value : values) {
          kinds[value] = kind;
        }
      }
    }
    // The ops of the clusters taken: their results took the kind.
    for (auto &[op, plan] : plans) {
      if (kinds.contains(op->getResult(0))) {
        built.try_emplace(op, std::move(plan));
      }
    }
  }

  bool changed() const { return !kinds.empty(); }

  // Rebuilds the function's ops on their physical layouts.
  void rewrite() {
    Block &entry = func.getBody().front();
    MLIRContext *context = func.getContext();
    for (BlockArgument arg : entry.getArguments()) {
      if (auto kind = kindOf(arg)) {
        OpBuilder b = OpBuilder::atBlockBegin(&entry);
        physical[arg] =
            reshapeTo(b, arg.getLoc(), arg, physicalShape(*kind, typeOf(arg)));
        logical[arg] = arg;
      }
    }
    SmallVector<Operation *> ops;
    for (Operation &op : entry) {
      ops.push_back(&op);
    }
    SmallVector<Operation *> replaced;
    for (Operation *op : ops) {
      OpBuilder b(op);
      if (auto reshape = mlir::dyn_cast<ReshapeOp>(op)) {
        Value result = reshape.getResult();
        std::optional<Kind> kind = kindOf(result);
        if (!kind && !kindOf(reshape.getInput())) {
          continue;
        }
        Value input = physicalOf(reshape.getInput());
        Value out =
            reshapeTo(b, op->getLoc(), input,
                      kind ? physicalShape(*kind, typeOf(result))
                           : SmallVector<int64_t>(typeOf(result).getShape()));
        if (kind) {
          physical[result] = out;
        } else {
          result.replaceAllUsesWith(out);
        }
        replaced.push_back(op);
        continue;
      }
      if (auto it = built.find(op); it != built.end()) {
        const Plan &plan = it->second;
        SmallVector<Value> operands;
        for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
          if (plan.member(i)) {
            operands.push_back(physicalOf(operand));
          } else {
            operands.push_back(reshapeTo(b, op->getLoc(), physicalOf(operand),
                                         *plan.operands[i]));
          }
        }
        RankedTensorType result = typeOf(op->getResult(0));
        physical[op->getResult(0)] = plan.rule->build(
            b, op, operands,
            withShape(result, physicalShape(plan.rule->kind, result)));
        replaced.push_back(op);
        continue;
      }
      for (OpOperand &operand : op->getOpOperands()) {
        if (kindOf(operand.get())) {
          operand.set(logicalOf(operand.get()));
        }
      }
      for (Value result : op->getResults()) {
        if (auto kind = kindOf(result)) {
          OpBuilder after(context);
          after.setInsertionPointAfter(op);
          physical[result] = reshapeTo(after, op->getLoc(), result,
                                       physicalShape(*kind, typeOf(result)));
          logical[result] = result;
        }
      }
    }
    // Uses left (inside regions) take the logical shape.
    llvm::DenseSet<Operation *> erased(replaced.begin(), replaced.end());
    for (Operation *op : replaced) {
      for (Value result : op->getResults()) {
        if (kindOf(result) &&
            llvm::any_of(result.getUsers(),
                         [&](Operation *u) { return !erased.contains(u); })) {
          result.replaceUsesWithIf(logicalOf(result), [&](OpOperand &use) {
            return !erased.contains(use.getOwner());
          });
        }
      }
    }
    for (Operation *op : llvm::reverse(replaced)) {
      if (op->use_empty()) {
        op->erase();
      }
    }
  }

private:
  std::optional<Kind> kindOf(Value value) const {
    auto it = kinds.find(value);
    if (it == kinds.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  Value physicalOf(Value value) {
    auto it = physical.find(value);
    return it == physical.end() ? value : it->second;
  }

  // A value of a kind in its logical shape: one reshape, made once.
  Value logicalOf(Value value) {
    if (auto it = logical.find(value); it != logical.end()) {
      return it->second;
    }
    Value stored = physical.lookup(value);
    OpBuilder b(func.getContext());
    b.setInsertionPointAfterValue(stored);
    Value result =
        reshapeTo(b, stored.getLoc(), stored, typeOf(value).getShape());
    logical[value] = result;
    return result;
  }

  // The shape a value has where `kinds` doesn't give it one: its logical
  // shape, or a decided kind's.
  SmallVector<int64_t> currentShape(Value value) const {
    if (auto kind = kindOf(value)) {
      return physicalShape(*kind, typeOf(value));
    }
    return SmallVector<int64_t>(typeOf(value).getShape());
  }

  // Whether a cluster's ops save more time than the reshapes at its edges
  // take.
  bool worthIt(Kind kind, ArrayRef<Value> values,
               const llvm::DenseMap<Operation *, Plan> &plans) const {
    double saved = 0;
    double cost = 0;
    for (Value value : values) {
      RankedTensorType type = typeOf(value);
      Type element = type.getElementType();
      ArrayRef<int64_t> shape = type.getShape();
      SmallVector<int64_t> stored = physicalShape(kind, type);
      // Each op of the cluster that writes or reads it takes less time.
      double smaller = opTime(shape, element) - opTime(stored, element);
      // Making it.
      Operation *producer = value.getDefiningOp();
      if (producer && plans.count(producer)) {
        saved += smaller;
      } else if (auto reshape = mlir::dyn_cast_or_null<ReshapeOp>(producer)) {
        SmallVector<int64_t> from = currentShape(reshape.getInput());
        cost += reshapeCost(from, stored, element) -
                reshapeCost(from, shape, element);
      } else {
        cost += reshapeCost(shape, stored, element);
      }
      // Its uses.
      bool needsLogical = false;
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        auto plan = plans.find(user);
        if (plan != plans.end()) {
          if (plan->second.member(use.getOperandNumber())) {
            saved += smaller;
            continue;
          }
          ArrayRef<int64_t> edge =
              *plan->second.operands[use.getOperandNumber()];
          cost += reshapeCost(stored, edge, element) -
                  reshapeCost(shape, edge, element);
        } else if (auto reshape = mlir::dyn_cast<ReshapeOp>(user);
                   reshape && user->getBlock() == value.getParentBlock()) {
          SmallVector<int64_t> to = currentShape(reshape.getResult());
          cost += reshapeCost(stored, to, element) -
                  reshapeCost(shape, to, element);
        } else {
          needsLogical = true;
        }
      }
      if (needsLogical) {
        cost += reshapeCost(stored, shape, element);
      }
    }
    return saved - cost >= std::max(minSavedBytes(), 1.0);
  }

  func::FuncOp func;
  llvm::DenseMap<Value, Kind> kinds;
  llvm::DenseMap<Operation *, Plan> built;
  llvm::DenseMap<Value, Value> physical;
  llvm::DenseMap<Value, Value> logical;
};

class TTIRPhysicalLayoutsPass
    : public PassWrapper<TTIRPhysicalLayoutsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRPhysicalLayoutsPass)

  StringRef getArgument() const final { return "ttir-physical-layouts"; }
  StringRef getDescription() const final {
    return "Store tensors in physical layouts without tile padding";
  }

  void runOnOperation() final {
    if (!enabled()) {
      return;
    }
    // Composites' decompositions stay as their composites have them.
    llvm::DenseSet<StringRef> decompositions;
    getOperation().walk([&](Operation *op) {
      if (auto callee = op->getAttrOfType<FlatSymbolRefAttr>("decomposition")) {
        decompositions.insert(callee.getValue());
      }
    });
    SmallVector<func::FuncOp> changed;
    getOperation().walk([&](func::FuncOp func) {
      if (func.isExternal() || decompositions.contains(func.getName())) {
        return;
      }
      bool viewed = !func.isPrivate() && viewBoundaries(func);
      Layouts layouts(func);
      layouts.choose(Kind::U);
      layouts.choose(Kind::F);
      if (layouts.changed()) {
        layouts.rewrite();
      }
      if (viewed || layouts.changed()) {
        changed.push_back(func);
      }
    });
    // Folding alone (ReshapeOp::fold) merges consecutive reshapes, drops
    // reshapes to the same shape, and reshapes constants and fills.
    RewritePatternSet patterns(&getContext());
    FrozenRewritePatternSet folding(std::move(patterns));
    for (func::FuncOp func : changed) {
      if (failed(applyPatternsGreedily(func, folding))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createTTIRPhysicalLayoutsPass() {
  return std::make_unique<TTIRPhysicalLayoutsPass>();
}

} // namespace mlir::tt::ttir
