// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// ttir-physical-layouts: large tensors stored without tile padding.
//
// TTNN pads each slice of a tiled tensor's last two dims to 32 x 32 tiles, so
// a rank-1 array or a [T, 1, V] tensor takes 32 times its size. One rule
// (compactShape in ttmlir/PhysicalLayouts.h) gives large ones a compact shape
// holding the same elements in the same order: rank-1 arrays and matrices of a
// few rows folded to [n / 32, 32] (F), [T, 1, V] as [T, V] (U). The runtime
// stores buffers that way; this pass applies the rule to programs. Public
// functions take and return such tensors in their compact shapes, and each op
// with a rule for its compact tensor is rebuilt on compact operands, between
// reshapes from and to the logical shapes. Reshape folding removes the pairs
// between rebuilt ops. Other ops see the logical shapes.
//
// LIBTT_PHYSICAL_LAYOUTS=0 turns it off.

#include "ttmlir/PhysicalLayouts.h"
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>
#include <optional>

namespace mlir::tt::ttir {

namespace {

constexpr int64_t kTile = 32;

RankedTensorType typeOf(Value value) {
  return mlir::dyn_cast<RankedTensorType>(value.getType());
}

RankedTensorType withShape(RankedTensorType type, ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, type.getElementType(),
                               type.getEncoding());
}

// The shape the rule stores a tensor of `type` in, if not its own.
std::optional<std::array<int64_t, 2>> compactShape(RankedTensorType type) {
  if (!type || !type.hasStaticShape()) {
    return std::nullopt;
  }
  return ttmlir::physical_layouts::compactShape(type.getShape());
}

// A compact flat buffer or few-row matrix, stored folded (F).
bool isFolded(RankedTensorType type) {
  return compactShape(type) && type.getRank() <= 2;
}

// A compact [T, 1, V], stored as [T, V] (U).
bool isUnitRows(RankedTensorType type) {
  return compactShape(type) && type.getRank() == 3;
}

// Any tensor's elements as [n / 32, 32].
SmallVector<int64_t> foldShape(RankedTensorType type) {
  return {type.getNumElements() / kTile, kTile};
}

// A [T, 1, V]-like shape without its middle dim.
SmallVector<int64_t> withoutMiddle(ArrayRef<int64_t> shape) {
  return {shape[0], shape[2]};
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

// `operand` reshaped to `shape` where it is made, so that its padded shape
// lives no longer.
Value reshapeOperand(Value operand, ArrayRef<int64_t> shape) {
  OpBuilder after(operand.getContext());
  after.setInsertionPointAfterValue(operand);
  return reshapeTo(after, operand.getLoc(), operand, shape);
}

// `op` on its operands reshaped to `shapes`, with a result of `shape` and the
// attributes in `changed` replaced.
Value emit(OpBuilder &b, Operation *op, ArrayRef<SmallVector<int64_t>> shapes,
           ArrayRef<int64_t> shape, ArrayRef<NamedAttribute> changed = {}) {
  SmallVector<Value> operands;
  for (auto [operand, s] : llvm::zip_equal(op->getOperands(), shapes)) {
    operands.push_back(reshapeOperand(operand, s));
  }
  NamedAttrList attrs(op->getAttrs());
  for (NamedAttribute attr : changed) {
    attrs.set(attr.getName(), attr.getValue());
  }
  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       TypeRange{withShape(typeOf(op->getResult(0)), shape)},
                       attrs);
  return b.create(state)->getResult(0);
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

// Elementwise ops into a compact result. Folded (F): operands of its shape fold
// alike, one-element operands are [1, 1]. [T, 1, V] (U): every operand, of the
// result's shape or broadcast against it, without its middle dim.
Value rewriteElementwise(OpBuilder &b, Operation *op) {
  RankedTensorType result = typeOf(op->getResult(0));
  bool folded = isFolded(result);
  if (!folded && !isUnitRows(result)) {
    return nullptr;
  }
  SmallVector<SmallVector<int64_t>> shapes;
  for (Value operand : op->getOperands()) {
    RankedTensorType type = typeOf(operand);
    if (folded && type.getShape() == result.getShape()) {
      shapes.push_back(foldShape(type));
    } else if (folded && type.getNumElements() == 1) {
      shapes.push_back({1, 1});
    } else if (!folded && type.getRank() == 3 && type.getDimSize(1) == 1) {
      shapes.push_back(withoutMiddle(type.getShape()));
    } else {
      return nullptr;
    }
  }
  return emit(b, op, shapes,
              folded ? foldShape(result) : withoutMiddle(result.getShape()));
}

// Broadcasts of one element into a folded result.
Value rewriteBroadcast(OpBuilder &b, BroadcastOp broadcast) {
  RankedTensorType result = typeOf(broadcast.getResult());
  if (!isFolded(result) || typeOf(broadcast.getInput()).getNumElements() != 1) {
    return nullptr;
  }
  SmallVector<int64_t> fold = foldShape(result);
  return emit(
      b, broadcast, {SmallVector<int64_t>{1, 1}}, fold,
      {b.getNamedAttr("broadcast_dimensions", b.getDenseI64ArrayAttr(fold))});
}

// Non-empty slices of a compact input with unit steps. [T, 1, V]: the bounds
// of T and V. Folded: each sliced row's columns, from and to 32-element
// boundaries, are rows of the fold (a flat buffer is one row), stacked into a
// folded result.
Value rewriteSlice(OpBuilder &b, SliceStaticOp slice) {
  RankedTensorType input = typeOf(slice.getInput());
  RankedTensorType result = typeOf(slice.getResult());
  if (result.getNumElements() == 0) {
    return nullptr;
  }
  SmallVector<int64_t> begins = intsOf(slice.getBegins());
  SmallVector<int64_t> ends = intsOf(slice.getEnds());
  SmallVector<int64_t> steps = intsOf(slice.getStep());
  // Negative bounds count from the end of their own dim, as in TTIR.
  for (auto [i, size] : llvm::enumerate(input.getShape())) {
    begins[i] += begins[i] < 0 ? size : 0;
    ends[i] += ends[i] < 0 ? size : 0;
  }
  if (isUnitRows(input)) {
    auto outer = [&](ArrayRef<int64_t> bounds) {
      return i32Array(b, withoutMiddle(bounds));
    };
    return emit(b, slice, {withoutMiddle(input.getShape())},
                withoutMiddle(result.getShape()),
                {b.getNamedAttr("begins", outer(begins)),
                 b.getNamedAttr("ends", outer(ends)),
                 b.getNamedAttr("step", outer(steps))});
  }
  bool flat = input.getRank() == 1;
  int64_t columns = input.getShape().back();
  int64_t from = begins.back();
  int64_t to = ends.back();
  if (!isFolded(input) ||
      llvm::any_of(steps, [](int64_t s) { return s != 1; }) ||
      columns % kTile != 0 || from % kTile != 0 || to % kTile != 0) {
    return nullptr;
  }
  // The rows of the fold, as runs.
  SmallVector<std::pair<int64_t, int64_t>> runs;
  for (int64_t row = flat ? 0 : begins[0]; row < (flat ? 1 : ends[0]); ++row) {
    int64_t first = (row * columns + from) / kTile;
    int64_t last = (row * columns + to) / kTile;
    if (!runs.empty() && runs.back().second == first) {
      runs.back().second = last;
    } else {
      runs.push_back({first, last});
    }
  }
  Value fold = reshapeOperand(slice.getInput(), foldShape(input));
  SmallVector<Value> pieces;
  for (auto [first, last] : runs) {
    pieces.push_back(foldRows(b, slice.getLoc(), fold, first, last));
  }
  return stackRows(b, slice.getLoc(), withShape(result, foldShape(result)),
                   pieces);
}

// Concats into a compact result. [T, 1, V]: along T or V, the same on [T, V].
// Folded: inputs of any size folded alike and stacked, whole, or for matrices
// side by side, a row of each in turn.
Value rewriteConcat(OpBuilder &b, ConcatOp concat) {
  RankedTensorType result = typeOf(concat.getResult());
  int64_t rank = result.getRank();
  int64_t dim = normalized(concat.getDim(), rank);
  if (isUnitRows(result) && dim != 1) {
    SmallVector<SmallVector<int64_t>> shapes;
    for (Value input : concat.getInputs()) {
      shapes.push_back(withoutMiddle(typeOf(input).getShape()));
    }
    return emit(
        b, concat, shapes, withoutMiddle(result.getShape()),
        {b.getNamedAttr("dim", b.getSI32IntegerAttr(dim == 0 ? 0 : 1))});
  }
  if (!isFolded(result) || llvm::any_of(concat.getInputs(), [&](Value input) {
        RankedTensorType type = typeOf(input);
        return type.getRank() != rank || type.getShape().back() % kTile != 0;
      })) {
    return nullptr;
  }
  SmallVector<Value> folds;
  for (Value input : concat.getInputs()) {
    folds.push_back(reshapeOperand(input, foldShape(typeOf(input))));
  }
  int64_t parts = rank == 2 && dim == 1 ? result.getDimSize(0) : 1;
  SmallVector<Value> pieces;
  for (int64_t part = 0; part < parts; ++part) {
    for (Value fold : folds) {
      int64_t rows = typeOf(fold).getDimSize(0) / parts;
      pieces.push_back(
          foldRows(b, concat.getLoc(), fold, part * rows, (part + 1) * rows));
    }
  }
  return stackRows(b, concat.getLoc(), withShape(result, foldShape(result)),
                   pieces);
}

// All-gathers and reduce-scatters of flat buffers with either side folded.
Value rewriteGatherScatter(OpBuilder &b, Operation *op) {
  RankedTensorType input = typeOf(op->getOperand(0));
  RankedTensorType result = typeOf(op->getResult(0));
  if (input.getRank() != 1 || result.getRank() != 1 ||
      !(isFolded(input) || isFolded(result)) ||
      input.getNumElements() % kTile != 0 ||
      result.getNumElements() % kTile != 0) {
    return nullptr;
  }
  StringRef dim = mlir::isa<AllGatherOp>(op) ? "all_gather_dim" : "scatter_dim";
  return emit(b, op, {foldShape(input)}, foldShape(result),
              {b.getNamedAttr(dim, b.getSI32IntegerAttr(0))});
}

// Reductions of a [T, 1, V] input over T or V: the same on [T, V].
Value rewriteReduction(OpBuilder &b, Operation *op) {
  RankedTensorType input = typeOf(op->getOperand(0));
  auto dimArg = op->getAttrOfType<ArrayAttr>("dim_arg");
  if (!isUnitRows(input) || !dimArg) {
    return nullptr;
  }
  SmallVector<int64_t> dims;
  for (int64_t d : intsOf(dimArg)) {
    if (int64_t dim = normalized(d, 3); dim != 1) {
      dims.push_back(dim == 0 ? 0 : 1);
    }
  }
  if (dims.empty()) {
    return nullptr;
  }
  bool keep = op->getAttrOfType<BoolAttr>("keep_dim").getValue();
  SmallVector<int64_t> rows = withoutMiddle(input.getShape());
  SmallVector<int64_t> shape;
  for (auto [i, d] : llvm::enumerate(rows)) {
    if (!llvm::is_contained(dims, static_cast<int64_t>(i))) {
      shape.push_back(d);
    } else if (keep) {
      shape.push_back(1);
    }
  }
  return emit(b, op, {rows}, shape,
              {b.getNamedAttr("dim_arg", i32Array(b, dims))});
}

// `op` rebuilt on compact operands, if a rule takes it: its compact result.
Value rewrite(OpBuilder &b, Operation *op) {
  if (mlir::isa<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>(op)) {
    return rewriteElementwise(b, op);
  }
  if (auto broadcast = mlir::dyn_cast<BroadcastOp>(op)) {
    return rewriteBroadcast(b, broadcast);
  }
  if (auto slice = mlir::dyn_cast<SliceStaticOp>(op)) {
    return rewriteSlice(b, slice);
  }
  if (auto concat = mlir::dyn_cast<ConcatOp>(op)) {
    return rewriteConcat(b, concat);
  }
  if (mlir::isa<AllGatherOp, ReduceScatterOp>(op)) {
    return rewriteGatherScatter(b, op);
  }
  if (mlir::isa<SumOp, MeanOp, MaxOp, MinOp, ProdOp, ArgMaxOp, ReduceAndOp,
                ReduceOrOp>(op)) {
    return rewriteReduction(b, op);
  }
  return nullptr;
}

// Rebuilds each op a rule takes on compact operands, between reshapes from
// and to the logical shapes. Folding then drops the reshapes between rebuilt
// ops, and CSE merges the copies a value gets for several users.
bool rebuild(func::FuncOp func) {
  auto isStatic = [](Type type) {
    auto tensor = mlir::dyn_cast<RankedTensorType>(type);
    return tensor && tensor.hasStaticShape();
  };
  bool changed = false;
  for (Operation *op :
       llvm::to_vector(llvm::make_pointer_range(func.getBody().front()))) {
    if (op->getNumResults() != 1 ||
        !llvm::all_of(op->getOperandTypes(), isStatic) ||
        !llvm::all_of(op->getResultTypes(), isStatic)) {
      continue;
    }
    OpBuilder b(op);
    Value out = rewrite(b, op);
    if (!out) {
      continue;
    }
    changed = true;
    RankedTensorType type = typeOf(op->getResult(0));
    op->getResult(0).replaceAllUsesWith(
        reshapeTo(b, op->getLoc(), out, type.getShape()));
    op->erase();
  }
  return changed;
}

// Whether a boundary's tensor is the device's own (unsharded, or presharded
// with its local shape): a mesh_shard would cut it otherwise.
bool wholeOnDevice(ttcore::LocalShapeAttr local, RankedTensorType type) {
  return !local || local.getLocalShape().getShape() == type.getShape();
}

// Gives a public function's arguments and results their compact shapes, each
// with a reshape to or from its logical shape.
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
    std::optional<std::array<int64_t, 2>> shape = compactShape(type);
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
    std::optional<std::array<int64_t, 2>> shape = compactShape(type);
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

class TTIRPhysicalLayoutsPass
    : public PassWrapper<TTIRPhysicalLayoutsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRPhysicalLayoutsPass)

  StringRef getArgument() const final { return "ttir-physical-layouts"; }
  StringRef getDescription() const final {
    return "Store large tensors without tile padding";
  }

  void runOnOperation() final {
    if (!ttmlir::physical_layouts::enabled()) {
      return;
    }
    SmallVector<func::FuncOp> changed;
    getOperation().walk([&](func::FuncOp func) {
      if (func.isExternal()) {
        return;
      }
      bool viewed = !func.isPrivate() && viewBoundaries(func);
      if (rebuild(func) || viewed) {
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
