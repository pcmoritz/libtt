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

// The dim that `op`'s integer attribute `name` names in a shape of `rank`.
int64_t dimOf(Operation *op, StringRef name, int64_t rank) {
  return normalized(
      op->getAttrOfType<IntegerAttr>(name).getValue().getSExtValue(), rank);
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

template <typename... Ops>
bool isOneOf(Operation *op) {
  return mlir::isa<Ops...>(op);
}

struct Rule;

// How an op is rebuilt: its rule, the shape each operand is reshaped to, and
// the shape of its result.
struct Plan {
  const Rule *rule = nullptr;
  SmallVector<SmallVector<int64_t>> operands;
  SmallVector<int64_t> result;
};

// One rule per op family and layout: `match` decides from the op's compact
// tensor whether it is rebuilt and on which shapes, `build` emits it.
struct Rule {
  bool (*isa)(Operation *op);
  std::optional<Plan> (*match)(Operation *op);
  Value (*build)(OpBuilder &b, Operation *op, ValueRange operands,
                 RankedTensorType type);
};

Value buildPlain(OpBuilder &b, Operation *op, ValueRange operands,
                 RankedTensorType type) {
  return create(b, op, operands, type);
}

//===----------------------------------------------------------------------===//
// F: flat buffers and few-row matrices folded to [n / 32, 32].
//===----------------------------------------------------------------------===//

// Elementwise ops and broadcasts into a folded result: operands of its shape
// fold alike, one-element operands are [1, 1].
std::optional<Plan> matchElementwiseF(Operation *op) {
  RankedTensorType result = typeOf(op->getResult(0));
  if (!isFolded(result)) {
    return std::nullopt;
  }
  Plan plan;
  plan.result = foldShape(result);
  for (Value operand : op->getOperands()) {
    RankedTensorType type = typeOf(operand);
    if (type.getShape() == result.getShape()) {
      plan.operands.push_back(plan.result);
    } else if (type.getNumElements() == 1) {
      plan.operands.push_back({1, 1});
    } else {
      return std::nullopt;
    }
  }
  return plan;
}

Value buildBroadcastF(OpBuilder &b, Operation *op, ValueRange operands,
                      RankedTensorType type) {
  SmallVector<int64_t> multipliers;
  for (auto [out, in] :
       llvm::zip_equal(type.getShape(), typeOf(operands[0]).getShape())) {
    multipliers.push_back(out / in);
  }
  return create(b, op, operands, type,
                {b.getNamedAttr("broadcast_dimensions",
                                b.getDenseI64ArrayAttr(multipliers))});
}

// For a static slice with unit steps, its begins and sizes.
std::optional<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
unitStepBounds(SliceStaticOp slice) {
  SmallVector<int64_t> begins = intsOf(slice.getBegins());
  SmallVector<int64_t> ends = intsOf(slice.getEnds());
  if (llvm::any_of(intsOf(slice.getStep()), [](int64_t s) { return s != 1; })) {
    return std::nullopt;
  }
  SmallVector<int64_t> sizes;
  for (auto [b, e] : llvm::zip_equal(begins, ends)) {
    sizes.push_back(e - b);
  }
  return std::make_pair(begins, sizes);
}

// Slices of a folded input: 32-element runs of a flat buffer, whole rows of a
// matrix, or aligned column blocks of every row. The result is folded too,
// whatever its size.
std::optional<Plan> matchSliceF(Operation *op) {
  RankedTensorType input = typeOf(op->getOperand(0));
  RankedTensorType result = typeOf(op->getResult(0));
  auto bounds = unitStepBounds(mlir::cast<SliceStaticOp>(op));
  if (!isFolded(input) || !bounds || result.getNumElements() % kTile != 0) {
    return std::nullopt;
  }
  auto &[begins, sizes] = *bounds;
  if (input.getRank() == 1 && begins[0] % kTile != 0) {
    return std::nullopt;
  }
  if (input.getRank() == 2) {
    int64_t columns = input.getDimSize(1);
    bool rows = sizes[1] == columns;
    bool blocks = sizes[0] == input.getDimSize(0) && begins[1] % kTile == 0 &&
                  sizes[1] % kTile == 0;
    if (columns % kTile != 0 || (!rows && !blocks)) {
      return std::nullopt;
    }
  }
  Plan plan;
  plan.operands.push_back(foldShape(input));
  plan.result = foldShape(result);
  return plan;
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

// Concats into a folded result: flat buffers or rows stacked, or matrices
// side by side (each input's blocks of a row in turn). Inputs of any size
// fold alike.
std::optional<Plan> matchConcatF(Operation *op) {
  RankedTensorType result = typeOf(op->getResult(0));
  if (!isFolded(result)) {
    return std::nullopt;
  }
  Plan plan;
  plan.result = foldShape(result);
  for (Value operand : op->getOperands()) {
    RankedTensorType type = typeOf(operand);
    if (type.getRank() != result.getRank() ||
        type.getNumElements() % kTile != 0 ||
        (type.getRank() == 2 && type.getDimSize(1) % kTile != 0)) {
      return std::nullopt;
    }
    plan.operands.push_back(foldShape(type));
  }
  return plan;
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

// All-gathers and reduce-scatters of flat buffers with either side folded.
std::optional<Plan> matchGatherScatterF(Operation *op) {
  RankedTensorType input = typeOf(op->getOperand(0));
  RankedTensorType result = typeOf(op->getResult(0));
  if (input.getRank() != 1 || result.getRank() != 1 ||
      !(isFolded(input) || isFolded(result)) ||
      input.getNumElements() % kTile != 0 ||
      result.getNumElements() % kTile != 0) {
    return std::nullopt;
  }
  Plan plan;
  plan.operands.push_back(foldShape(input));
  plan.result = foldShape(result);
  return plan;
}

Value buildGatherScatterF(OpBuilder &b, Operation *op, ValueRange operands,
                          RankedTensorType type) {
  StringRef dim = mlir::isa<AllGatherOp>(op) ? "all_gather_dim" : "scatter_dim";
  return create(b, op, operands, type,
                {b.getNamedAttr(dim, b.getSI32IntegerAttr(0))});
}

//===----------------------------------------------------------------------===//
// U: [T, 1, V] as [T, V].
//===----------------------------------------------------------------------===//

// Elementwise ops into a [T, 1, V] result: every operand, of the result's
// shape or broadcast against it, without its middle dim.
std::optional<Plan> matchElementwiseU(Operation *op) {
  RankedTensorType result = typeOf(op->getResult(0));
  if (!isUnitRows(result)) {
    return std::nullopt;
  }
  Plan plan;
  plan.result = withoutMiddle(result.getShape());
  for (Value operand : op->getOperands()) {
    RankedTensorType type = typeOf(operand);
    if (type.getRank() != 3 || type.getDimSize(1) != 1) {
      return std::nullopt;
    }
    plan.operands.push_back(withoutMiddle(type.getShape()));
  }
  return plan;
}

// A reduction's dims of [T, 1, V] among those of [T, V].
SmallVector<int64_t> reducedDims(Operation *op) {
  SmallVector<int64_t> dims;
  for (int64_t d : intsOf(op->getAttrOfType<ArrayAttr>("dim_arg"))) {
    int64_t dim = normalized(d, 3);
    if (dim != 1) {
      dims.push_back(dim == 0 ? 0 : 1);
    }
  }
  return dims;
}

// Reductions of a [T, 1, V] input over T or V: the same on [T, V].
std::optional<Plan> matchReductionU(Operation *op) {
  RankedTensorType input = typeOf(op->getOperand(0));
  if (!isUnitRows(input) || !op->getAttrOfType<ArrayAttr>("dim_arg")) {
    return std::nullopt;
  }
  SmallVector<int64_t> dims = reducedDims(op);
  if (dims.empty()) {
    return std::nullopt;
  }
  bool keep = op->getAttrOfType<BoolAttr>("keep_dim").getValue();
  Plan plan;
  plan.operands.push_back(withoutMiddle(input.getShape()));
  for (auto [i, d] : llvm::enumerate(plan.operands.front())) {
    if (!llvm::is_contained(dims, static_cast<int64_t>(i))) {
      plan.result.push_back(d);
    } else if (keep) {
      plan.result.push_back(1);
    }
  }
  return plan;
}

Value buildReductionU(OpBuilder &b, Operation *op, ValueRange operands,
                      RankedTensorType type) {
  return create(b, op, operands, type,
                {b.getNamedAttr("dim_arg", i32Array(b, reducedDims(op)))});
}

// Slices of a [T, 1, V] input: the bounds of T and V.
std::optional<Plan> matchSliceU(Operation *op) {
  RankedTensorType input = typeOf(op->getOperand(0));
  if (!isUnitRows(input)) {
    return std::nullopt;
  }
  Plan plan;
  plan.operands.push_back(withoutMiddle(input.getShape()));
  plan.result = withoutMiddle(typeOf(op->getResult(0)).getShape());
  return plan;
}

Value buildSliceU(OpBuilder &b, Operation *op, ValueRange operands,
                  RankedTensorType type) {
  auto slice = mlir::cast<SliceStaticOp>(op);
  auto outer = [&](ArrayAttr bounds) {
    return i32Array(b, withoutMiddle(intsOf(bounds)));
  };
  return create(b, op, operands, type,
                {b.getNamedAttr("begins", outer(slice.getBegins())),
                 b.getNamedAttr("ends", outer(slice.getEnds())),
                 b.getNamedAttr("step", outer(slice.getStep()))});
}

// Concats into a [T, 1, V] result along T or V: the same on [T, V].
std::optional<Plan> matchConcatU(Operation *op) {
  RankedTensorType result = typeOf(op->getResult(0));
  if (!isUnitRows(result) || dimOf(op, "dim", 3) == 1) {
    return std::nullopt;
  }
  Plan plan;
  plan.result = withoutMiddle(result.getShape());
  for (Value operand : op->getOperands()) {
    plan.operands.push_back(withoutMiddle(typeOf(operand).getShape()));
  }
  return plan;
}

Value buildConcatU(OpBuilder &b, Operation *op, ValueRange operands,
                   RankedTensorType type) {
  int64_t dim = dimOf(op, "dim", 3) == 0 ? 0 : 1;
  return create(b, op, operands, type,
                {b.getNamedAttr("dim", b.getSI32IntegerAttr(dim))});
}

const Rule kRules[] = {
    {isOneOf<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>,
     matchElementwiseF, buildPlain},
    {isOneOf<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>,
     matchElementwiseU, buildPlain},
    {isOneOf<BroadcastOp>, matchElementwiseF, buildBroadcastF},
    {isOneOf<SliceStaticOp>, matchSliceF, buildSliceF},
    {isOneOf<SliceStaticOp>, matchSliceU, buildSliceU},
    {isOneOf<ConcatOp>, matchConcatF, buildConcatF},
    {isOneOf<ConcatOp>, matchConcatU, buildConcatU},
    {isOneOf<AllGatherOp, ReduceScatterOp>, matchGatherScatterF,
     buildGatherScatterF},
    {isOneOf<SumOp, MeanOp, MaxOp, MinOp, ProdOp, ArgMaxOp, ReduceAndOp,
             ReduceOrOp>,
     matchReductionU, buildReductionU},
};

// How `op` is rebuilt, if a rule takes it.
std::optional<Plan> planFor(Operation *op) {
  auto isStatic = [](Type type) {
    auto tensor = mlir::dyn_cast<RankedTensorType>(type);
    return tensor && tensor.hasStaticShape();
  };
  if (op->getNumResults() != 1 ||
      !llvm::all_of(op->getOperandTypes(), isStatic) ||
      !llvm::all_of(op->getResultTypes(), isStatic)) {
    return std::nullopt;
  }
  for (const Rule &rule : kRules) {
    if (!rule.isa(op)) {
      continue;
    }
    if (std::optional<Plan> plan = rule.match(op)) {
      plan->rule = &rule;
      return plan;
    }
  }
  return std::nullopt;
}

// Rebuilds each op a rule takes on its planned shapes, between reshapes from
// and to the logical shapes. Folding then drops the reshapes between rebuilt
// ops, and CSE merges the copies a value gets for several users.
bool rebuild(func::FuncOp func) {
  bool changed = false;
  for (Operation *op :
       llvm::to_vector(llvm::make_pointer_range(func.getBody().front()))) {
    std::optional<Plan> plan = planFor(op);
    if (!plan) {
      continue;
    }
    changed = true;
    OpBuilder b(op);
    SmallVector<Value> operands;
    for (auto [operand, shape] :
         llvm::zip_equal(op->getOperands(), plan->operands)) {
      // Where the operand is made, so that its padded shape lives no longer.
      OpBuilder after(op->getContext());
      after.setInsertionPointAfterValue(operand);
      operands.push_back(reshapeTo(after, operand.getLoc(), operand, shape));
    }
    RankedTensorType type = typeOf(op->getResult(0));
    Value out =
        plan->rule->build(b, op, operands, withShape(type, plan->result));
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
