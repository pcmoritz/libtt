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
// planFor) form a cluster, which takes the kind when the time its ops save
// exceeds the time of the reshapes at its edges (worthIt). Public functions' arguments and results of the
// shapes the runtime stores compactly (boundaryShape, and compactShape in
// tt_mlir_physical_layouts.patch) take those shapes.
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
  int64_t batches =
      shape.size() > 2 ? product(shape.drop_back(2)) : int64_t{1};
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

// A value that is rebuilt in any shape at no cost: a constant or a fill.
bool isFree(Value value) {
  Operation *producer = value.getDefiningOp();
  if (auto constant = mlir::dyn_cast_or_null<ConstantOp>(producer)) {
    return mlir::isa<DenseElementsAttr>(constant.getValue());
  }
  return mlir::isa_and_nonnull<FullOp, ZerosOp, OnesOp>(producer);
}

Value rebuildFree(OpBuilder &b, Location loc, Value value,
                  ArrayRef<int64_t> shape) {
  RankedTensorType type = withShape(typeOf(value), shape);
  Operation *producer = value.getDefiningOp();
  if (auto constant = mlir::dyn_cast<ConstantOp>(producer)) {
    return b.create<ConstantOp>(
        loc, type,
        mlir::cast<DenseElementsAttr>(constant.getValue()).reshape(type));
  }
  SmallVector<NamedAttribute> attrs(producer->getAttrs());
  for (NamedAttribute &attr : attrs) {
    if (attr.getName() == "shape") {
      attr.setValue(b.getDenseI32ArrayAttr(
          SmallVector<int32_t>(shape.begin(), shape.end())));
    }
  }
  OperationState state(loc, producer->getName().getStringRef(), ValueRange{},
                       TypeRange{type}, attrs);
  return b.create(state)->getResult(0);
}

void setAttr(SmallVector<NamedAttribute> &attrs, StringRef name,
             Attribute value) {
  for (NamedAttribute &attr : attrs) {
    if (attr.getName() == name) {
      attr.setValue(value);
      return;
    }
  }
}

IntegerAttr withValue(Operation *op, StringRef name, int64_t value) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  return IntegerAttr::get(attr.getType(), value);
}

bool isElementwise(Operation *op) {
  return mlir::isa<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>(op);
}

bool isReduction(Operation *op) {
  return mlir::isa<SumOp, MeanOp, MaxOp, MinOp, ProdOp, ArgMaxOp, ReduceAndOp,
                   ReduceOrOp>(op);
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

// How an op runs in a kind: which operands are stored like its result
// (members), and the shape each other operand (an edge) takes.
struct Plan {
  // Per operand: nullopt for a member, else the edge's shape.
  SmallVector<std::optional<SmallVector<int64_t>>> operands;

  bool member(unsigned index) const { return !operands[index].has_value(); }
};

// Operands of the result's shape are members (constants are edges, rebuilt
// in the result's layout); `other` gives the other operands' shapes.
std::optional<Plan> elementwisePlan(
    Operation *op, Kind kind, RankedTensorType result,
    llvm::function_ref<std::optional<SmallVector<int64_t>>(RankedTensorType)>
        other) {
  Plan plan;
  for (Value operand : op->getOperands()) {
    RankedTensorType type = typeOf(operand);
    if (!type || !type.hasStaticShape()) {
      return std::nullopt;
    }
    if (type.getShape() == result.getShape()) {
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
    plan.operands.push_back(*shape);
  }
  return plan;
}

// The reduced dims of a reduction over U's dims, and the physical result's
// shape, if the reduction runs there.
std::optional<SmallVector<int64_t>> reducedUnitDims(Operation *op,
                                                    RankedTensorType input,
                                                    RankedTensorType result) {
  auto dimArg = op->getAttrOfType<ArrayAttr>("dim_arg");
  if (!dimArg) {
    return std::nullopt;
  }
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
  if (dims.empty() || remaining != withoutUnitDims(result.getShape())) {
    return std::nullopt;
  }
  return dims;
}

std::optional<Plan> planU(Operation *op) {
  RankedTensorType result = typeOf(op->getResult(0));
  if (!eligible(Kind::U, result)) {
    return std::nullopt;
  }
  auto oneMember = [&]() -> std::optional<Plan> {
    Plan plan;
    plan.operands.push_back(std::nullopt);
    for (unsigned i = 1; i < op->getNumOperands(); ++i) {
      plan.operands.push_back(SmallVector<int64_t>(
          typeOf(op->getOperand(i)).getShape()));
    }
    return plan;
  };
  if (isElementwise(op)) {
    return elementwisePlan(
        op, Kind::U, result,
        [&](RankedTensorType type) -> std::optional<SmallVector<int64_t>> {
          if (type.getRank() != result.getRank()) {
            return std::nullopt;
          }
          // A broadcast operand: 1 where the result has a unit dim.
          return dropUnitsOf(type.getShape(), result.getShape());
        });
  }
  RankedTensorType input =
      op->getNumOperands() ? typeOf(op->getOperand(0)) : RankedTensorType();
  if (!input || !input.hasStaticShape()) {
    return std::nullopt;
  }
  if (mlir::isa<BroadcastOp>(op)) {
    if (input.getRank() != result.getRank()) {
      return std::nullopt;
    }
    Plan plan;
    plan.operands.push_back(
        dropUnitsOf(input.getShape(), result.getShape()));
    return plan;
  }
  if (!eligible(Kind::U, input)) {
    return std::nullopt;
  }
  if (isReduction(op)) {
    if (!reducedUnitDims(op, input, result)) {
      return std::nullopt;
    }
    return oneMember();
  }
  if (auto softmax = mlir::dyn_cast<SoftmaxOp>(op)) {
    if (input.getDimSize(normalized(softmax.getDimension(), input.getRank())) ==
        1) {
      return std::nullopt;
    }
    return oneMember();
  }
  if (mlir::isa<PermuteOp>(op)) {
    return oneMember();
  }
  if (mlir::isa<SliceStaticOp, AllReduceOp>(op)) {
    if (!sameUnitDims(input.getShape(), result.getShape())) {
      return std::nullopt;
    }
    return oneMember();
  }
  if (mlir::isa<AllGatherOp, ReduceScatterOp>(op)) {
    StringRef name =
        mlir::isa<AllGatherOp>(op) ? "all_gather_dim" : "scatter_dim";
    int64_t dim = normalized(op->getAttrOfType<IntegerAttr>(name).getInt(),
                             input.getRank());
    if (!sameUnitDims(input.getShape(), result.getShape()) ||
        input.getDimSize(dim) == 1) {
      return std::nullopt;
    }
    return oneMember();
  }
  if (auto concat = mlir::dyn_cast<ConcatOp>(op)) {
    int64_t dim = normalized(concat.getDim(), result.getRank());
    if (result.getDimSize(dim) == 1) {
      return std::nullopt;
    }
    Plan plan;
    for (Value operand : op->getOperands()) {
      RankedTensorType type = typeOf(operand);
      if (!type || !type.hasStaticShape() ||
          !sameUnitDims(type.getShape(), result.getShape())) {
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
  if (mlir::isa<RMSNormOp, LayerNormOp>(op)) {
    auto normalizedShape =
        op->getAttrOfType<DenseI64ArrayAttr>("normalized_shape");
    if (!normalizedShape ||
        llvm::is_contained(
            input.getShape().take_back(normalizedShape.size()), 1)) {
      return std::nullopt;
    }
    return oneMember();
  }
  return std::nullopt;
}

std::optional<Plan> planF(Operation *op) {
  RankedTensorType result = typeOf(op->getResult(0));
  if (!eligible(Kind::F, result)) {
    return std::nullopt;
  }
  if (isElementwise(op)) {
    return elementwisePlan(
        op, Kind::F, result,
        [&](RankedTensorType type) -> std::optional<SmallVector<int64_t>> {
          if (type.getNumElements() != 1) {
            return std::nullopt;
          }
          return SmallVector<int64_t>{1, 1};
        });
  }
  RankedTensorType input =
      op->getNumOperands() ? typeOf(op->getOperand(0)) : RankedTensorType();
  if (!input || !input.hasStaticShape()) {
    return std::nullopt;
  }
  Plan member;
  member.operands.push_back(std::nullopt);
  if (mlir::isa<BroadcastOp>(op)) {
    if (input.getNumElements() != 1) {
      return std::nullopt;
    }
    Plan plan;
    plan.operands.push_back(SmallVector<int64_t>{1, 1});
    return plan;
  }
  if (!eligible(Kind::F, input)) {
    return std::nullopt;
  }
  if (mlir::isa<AllReduceOp>(op)) {
    return member;
  }
  if (mlir::isa<AllGatherOp, ReduceScatterOp>(op)) {
    if (input.getRank() != 1 || result.getRank() != 1) {
      return std::nullopt;
    }
    return member;
  }
  if (auto slice = mlir::dyn_cast<SliceStaticOp>(op)) {
    auto bounds = unitStepBounds(slice);
    if (!bounds || input.getRank() > 2) {
      return std::nullopt;
    }
    auto &[begins, sizes] = *bounds;
    if (input.getRank() == 1) {
      return begins[0] % kTile == 0 ? std::optional<Plan>(member)
                                    : std::nullopt;
    }
    // Whole rows, or columns on 32-element boundaries of every row.
    int64_t columns = input.getDimSize(1);
    if (columns % kTile != 0) {
      return std::nullopt;
    }
    if (sizes[1] == columns ||
        (sizes[0] == input.getDimSize(0) && begins[1] % kTile == 0 &&
         sizes[1] % kTile == 0)) {
      return member;
    }
    return std::nullopt;
  }
  if (auto concat = mlir::dyn_cast<ConcatOp>(op)) {
    if (result.getRank() > 2) {
      return std::nullopt;
    }
    Plan plan;
    for (Value operand : op->getOperands()) {
      RankedTensorType type = typeOf(operand);
      // Rank 1, or matrices whose rows are whole rows of the fold: stacked,
      // or side by side (each input's blocks of a row in turn).
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
  return std::nullopt;
}

// How `op` runs in `kind`, if it does.
std::optional<Plan> planFor(Operation *op, Kind kind) {
  if (op->getNumResults() != 1 || op->getNumRegions() != 0 ||
      !typeOf(op->getResult(0)) || mlir::isa<ReshapeOp>(op) ||
      isFree(op->getResult(0))) {
    return std::nullopt;
  }
  return kind == Kind::U ? planU(op) : planF(op);
}

// Rows [begin, end) of a fold.
Value foldRows(OpBuilder &b, Location loc, Value fold, int64_t begin,
               int64_t end) {
  RankedTensorType type = typeOf(fold);
  if (begin == 0 && end == type.getDimSize(0)) {
    return fold;
  }
  return b.create<SliceStaticOp>(loc, withShape(type, {end - begin, kTile}),
                                 fold, i32Array(b, {begin, 0}),
                                 i32Array(b, {end, kTile}),
                                 i32Array(b, {1, 1}));
}

Value stackRows(OpBuilder &b, Location loc, RankedTensorType type,
                ArrayRef<Value> pieces) {
  if (pieces.size() == 1) {
    return pieces.front();
  }
  return b.create<ConcatOp>(loc, type, pieces, b.getSI32IntegerAttr(0));
}

Value create(OpBuilder &b, Operation *op, ValueRange operands,
             RankedTensorType type, ArrayRef<NamedAttribute> attrs) {
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       TypeRange{type}, attrs);
  return b.create(state)->getResult(0);
}

// `op` on `operands`, stored in `kind`.
Value rebuild(OpBuilder &b, Operation *op, Kind kind, ValueRange operands) {
  Location loc = op->getLoc();
  RankedTensorType result = typeOf(op->getResult(0));
  RankedTensorType type = withShape(result, physicalShape(kind, result));
  SmallVector<NamedAttribute> attrs(op->getAttrs());
  if (kind == Kind::F) {
    if (auto broadcast = mlir::dyn_cast<BroadcastOp>(op)) {
      setAttr(attrs, "broadcast_dimensions",
              b.getDenseI64ArrayAttr(type.getShape()));
    } else if (auto slice = mlir::dyn_cast<SliceStaticOp>(op)) {
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
    } else if (auto concat = mlir::dyn_cast<ConcatOp>(op);
               concat && result.getRank() == 2 &&
               normalized(concat.getDim(), 2) == 1) {
      SmallVector<Value> pieces;
      for (int64_t row = 0; row < result.getDimSize(0); ++row) {
        for (auto [input, fold] :
             llvm::zip_equal(concat.getInputs(), operands)) {
          int64_t rowBlocks = typeOf(input).getDimSize(1) / kTile;
          pieces.push_back(foldRows(b, loc, fold, row * rowBlocks,
                                    (row + 1) * rowBlocks));
        }
      }
      return stackRows(b, loc, type, pieces);
    } else if (mlir::isa<ConcatOp>(op)) {
      setAttr(attrs, "dim", withValue(op, "dim", 0));
    }
    // Elementwise ops, all-reduces and rank-1 gathers keep their attributes.
    return create(b, op, operands, type, attrs);
  }

  RankedTensorType input = typeOf(op->getOperand(0));
  SmallVector<int64_t> kept =
      input ? keptIndex(input.getShape()) : SmallVector<int64_t>();
  if (auto broadcast = mlir::dyn_cast<BroadcastOp>(op)) {
    setAttr(attrs, "broadcast_dimensions",
            b.getDenseI64ArrayAttr(dropUnitsOf(
                broadcast.getBroadcastDimensions(), result.getShape())));
  } else if (isReduction(op)) {
    setAttr(attrs, "dim_arg",
            i32Array(b, *reducedUnitDims(op, input, result)));
    setAttr(attrs, "keep_dim", b.getBoolAttr(false));
  } else if (auto softmax = mlir::dyn_cast<SoftmaxOp>(op)) {
    setAttr(attrs, "dimension",
            withValue(op, "dimension",
                      kept[normalized(softmax.getDimension(),
                                      input.getRank())]));
  } else if (auto permute = mlir::dyn_cast<PermuteOp>(op)) {
    SmallVector<int64_t> permutation;
    for (int64_t d : permute.getPermutation()) {
      if (kept[d] >= 0) {
        permutation.push_back(kept[d]);
      }
    }
    if (llvm::is_sorted(permutation)) {
      return operands[0];
    }
    setAttr(attrs, "permutation", b.getDenseI64ArrayAttr(permutation));
  } else if (auto slice = mlir::dyn_cast<SliceStaticOp>(op)) {
    SmallVector<int64_t> begins = intsOf(slice.getBegins());
    SmallVector<int64_t> ends = intsOf(slice.getEnds());
    SmallVector<int64_t> steps = intsOf(slice.getStep());
    setAttr(attrs, "begins",
            i32Array(b, dropUnitsOf(begins, input.getShape())));
    setAttr(attrs, "ends", i32Array(b, dropUnitsOf(ends, input.getShape())));
    setAttr(attrs, "step", i32Array(b, dropUnitsOf(steps, input.getShape())));
  } else if (auto concat = mlir::dyn_cast<ConcatOp>(op)) {
    setAttr(attrs, "dim",
            withValue(op, "dim",
                      keptIndex(result.getShape())[normalized(
                          concat.getDim(), result.getRank())]));
  } else if (mlir::isa<AllGatherOp, ReduceScatterOp>(op)) {
    StringRef name =
        mlir::isa<AllGatherOp>(op) ? "all_gather_dim" : "scatter_dim";
    setAttr(attrs, name,
            withValue(op, name,
                      kept[normalized(op->getAttrOfType<IntegerAttr>(name)
                                          .getInt(),
                                      input.getRank())]));
  }
  // Elementwise ops, all-reduces and norms keep their attributes.
  return create(b, op, operands, type, attrs);
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

// Sets a boundary's local shape, if its attributes give one.
void setLocalShape(DictionaryAttr attrs, RankedTensorType type,
                   llvm::function_ref<void(StringRef, Attribute)> set) {
  if (!attrs || !attrs.get(ttcore::LocalShapeAttr::name)) {
    return;
  }
  set(ttcore::LocalShapeAttr::name,
      ttcore::LocalShapeAttr::get(type.getContext(), type));
}

// Whether the boundary's tensor is the device's own (unsharded, or presharded
// with its local shape): a mesh_shard would cut it otherwise.
bool wholeOnDevice(DictionaryAttr attrs, RankedTensorType type) {
  if (!attrs) {
    return true;
  }
  if (auto local = mlir::dyn_cast_or_null<ttcore::LocalShapeAttr>(
          attrs.get(ttcore::LocalShapeAttr::name))) {
    return local.getLocalShape().getShape() == type.getShape();
  }
  return true;
}

// Gives a public function's arguments and results the runtime's storage
// shapes, each with a reshape to or from its logical shape.
bool viewBoundaries(func::FuncOp func) {
  OpBuilder builder(func.getContext());
  bool viewed = false;
  Block &entry = func.getBody().front();
  SmallVector<Type> inputs(func.getArgumentTypes());
  SmallVector<Type> results(func.getResultTypes());

  builder.setInsertionPointToStart(&entry);
  for (BlockArgument arg : entry.getArguments()) {
    RankedTensorType type = typeOf(arg);
    std::optional<SmallVector<int64_t>> shape = boundaryShape(type);
    if (!shape || !wholeOnDevice(func.getArgAttrDict(arg.getArgNumber()), type) ||
        llvm::any_of(arg.getUsers(),
                     [](Operation *user) { return mlir::isa<MeshShardOp>(user); })) {
      continue;
    }
    RankedTensorType stored = withShape(type, *shape);
    viewed = true;
    arg.setType(stored);
    inputs[arg.getArgNumber()] = stored;
    Value view = reshapeTo(builder, func.getLoc(), arg, type.getShape());
    arg.replaceAllUsesExcept(view, view.getDefiningOp());
    setLocalShape(func.getArgAttrDict(arg.getArgNumber()), stored,
                  [&](StringRef name, Attribute value) {
                    func.setArgAttr(arg.getArgNumber(), name, value);
                  });
  }

  func.walk([&](func::ReturnOp ret) {
    builder.setInsertionPoint(ret);
    for (OpOperand &operand : ret->getOpOperands()) {
      unsigned index = operand.getOperandNumber();
      RankedTensorType type = typeOf(operand.get());
      std::optional<SmallVector<int64_t>> shape = boundaryShape(type);
      if (!shape || !wholeOnDevice(func.getResultAttrDict(index), type) ||
          operand.get().getDefiningOp<MeshShardOp>()) {
        continue;
      }
      viewed = true;
      operand.set(reshapeTo(builder, ret.getLoc(), operand.get(), *shape));
      results[index] = operand.get().getType();
      setLocalShape(func.getResultAttrDict(index), typeOf(operand.get()),
                    [&](StringRef name, Attribute value) {
                      func.setResultAttr(index, name, value);
                    });
    }
  });
  func.setType(builder.getFunctionType(inputs, results));
  return viewed;
}

class UnionFind {
public:
  unsigned id(Value value) {
    auto [it, inserted] = index.try_emplace(value, parent.size());
    if (inserted) {
      parent.push_back(parent.size());
    }
    return it->second;
  }
  unsigned find(unsigned i) {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  }
  unsigned find(Value value) { return find(id(value)); }
  void unite(Value a, Value b) { parent[find(a)] = find(b); }

private:
  llvm::DenseMap<Value, unsigned> index;
  SmallVector<unsigned> parent;
};

// Layouts chosen for one function's values, and how to rebuild it.
class Layouts {
public:
  explicit Layouts(func::FuncOp func) : func(func) {}

  // Chooses clusters of `kind` among the values no earlier kind took.
  void choose(Kind kind) {
    Block &entry = func.getBody().front();
    UnionFind clusters;
    llvm::DenseMap<Operation *, Plan> plans;
    SmallVector<Value> order;
    auto add = [&](Value value) {
      clusters.id(value);
      order.push_back(value);
    };
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
        add(member);
        clusters.unite(member, members.front());
      }
      plans[&op] = std::move(*plan);
    }

    llvm::DenseMap<unsigned, SmallVector<Value>> groups;
    SmallVector<unsigned> groupOrder;
    llvm::DenseSet<Value> seen;
    for (Value value : order) {
      if (!seen.insert(value).second) {
        continue;
      }
      unsigned root = clusters.find(value);
      if (!groups.count(root)) {
        groupOrder.push_back(root);
      }
      groups[root].push_back(value);
    }
    for (unsigned root : groupOrder) {
      ArrayRef<Value> values = groups[root];
      if (worthIt(kind, values, plans)) {
        for (Value value : values) {
          kinds[value] = kind;
        }
        for (auto &[op, plan] : plans) {
          if (clusters.find(op->getResult(0)) == root) {
            built.try_emplace(op, kind, plan);
          }
        }
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
        Value out = reshapeTo(
            b, op->getLoc(), input,
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
        auto &[kind, plan] = it->second;
        SmallVector<Value> operands;
        for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
          if (plan.member(i)) {
            operands.push_back(physicalOf(operand));
          } else if (isFree(operand)) {
            operands.push_back(
                rebuildFree(b, op->getLoc(), operand, *plan.operands[i]));
          } else {
            operands.push_back(reshapeTo(b, op->getLoc(), physicalOf(operand),
                                         *plan.operands[i]));
          }
        }
        physical[op->getResult(0)] = rebuild(b, op, kind, operands);
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
        if (kindOf(result) && llvm::any_of(result.getUsers(), [&](Operation *u) {
              return !erased.contains(u);
            })) {
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
    Value result = reshapeTo(b, stored.getLoc(), stored,
                             typeOf(value).getShape());
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
          ArrayRef<int64_t> edge = *plan->second.operands[use.getOperandNumber()];
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
  llvm::DenseMap<Operation *, std::pair<Kind, Plan>> built;
  llvm::DenseMap<Value, Value> physical;
  llvm::DenseMap<Value, Value> logical;
};

// reshape(reshape(x)) -> reshape(x), or x when the shapes match.
class FoldReshapes : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    Value input = reshape.getInput();
    if (typeOf(input).getShape() == typeOf(reshape.getResult()).getShape()) {
      rewriter.replaceOp(reshape, input);
      return success();
    }
    auto inner = input.getDefiningOp<ReshapeOp>();
    if (!inner) {
      return failure();
    }
    rewriter.replaceOp(reshape,
                       reshapeTo(rewriter, reshape.getLoc(), inner.getInput(),
                                 typeOf(reshape.getResult()).getShape()));
    return success();
  }
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
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldReshapes>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    for (func::FuncOp func : changed) {
      if (failed(applyPatternsGreedily(func, frozen))) {
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
