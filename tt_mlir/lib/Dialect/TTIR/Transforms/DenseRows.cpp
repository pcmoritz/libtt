// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Computes wide, short tensors in dense tiles. A [..., H, W] tensor with few
// rows (a batch of decode rows over the vocabulary, say [2, 151936]) fills
// H / 32 of every 32 x 32 tile, so each elementwise op over it moves sixteen
// times its data through DRAM. The same values as [..., H, W / 32, 32] (each
// row split into 32-element chunks) fill their tiles.
//
// The pass groups such values into regions joined by the ops that can work on
// the chunked shape: elementwise ops (a per-row operand [..., h, 1] becomes
// [..., h, 1, 1]), broadcasts, constants, reductions over the last dimension
// (which reduce the last two), masked reductions with per-row thresholds
// (libtt's masked_reduce reduces each chunked row's [W / 32, 32] matrix), the
// native generator's bits, case operands and results with the branches''
// arguments and results, and loop-invariant loop operands with the body's
// arguments. A region is rewritten in the chunked shape when it saves more
// passes over wide tensors than the reshapes at its edges cost:
//
//   reshape [..., H, W] -> [..., H, W / 32, 32]  where a value enters it,
//   reshape back                                 where one leaves it.
//
// Both reshapes are row copies between tiles (ttnn::experimental::dense_rows).
// Ops in a loop body count once per iteration, ops in a case branch once per
// branch taken.
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string_view>

namespace mlir::tt::ttir {
namespace {

constexpr int64_t kChunk = 32;
// Rows at most this many fill at most half a tile.
constexpr int64_t kMaxRows = 16;
// Narrower rows are not worth the reshapes.
constexpr int64_t kMinWidth = 2048;
// A loop body runs many times; weigh it by about a search's trip count.
constexpr double kLoopWeight = 32;
// A region must save more than its edges cost (a reshape costs about what a
// wide op saves).
constexpr double kMinGain = 1;

bool isWide(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !tensor.hasStaticShape() || tensor.getRank() < 2 ||
      tensor.getEncoding()) {
    return false;
  }
  ArrayRef<int64_t> shape = tensor.getShape();
  const int64_t width = shape.back();
  const int64_t rows = shape[shape.size() - 2];
  return width % kChunk == 0 && width >= kMinWidth && rows <= kMaxRows &&
         rows > 0;
}

RankedTensorType chunkedType(Type type) {
  auto tensor = cast<RankedTensorType>(type);
  SmallVector<int64_t> shape(tensor.getShape());
  shape.back() /= kChunk;
  shape.push_back(kChunk);
  return RankedTensorType::get(shape, tensor.getElementType());
}

// [..., h, 1] -> [..., h, 1, 1]
RankedTensorType rowOperandType(Type type) {
  auto tensor = cast<RankedTensorType>(type);
  SmallVector<int64_t> shape(tensor.getShape());
  shape.push_back(1);
  return RankedTensorType::get(shape, tensor.getElementType());
}

Value createReshape(OpBuilder &builder, Location loc, Value input,
                    RankedTensorType type) {
  SmallVector<int32_t> shape(type.getShape().begin(), type.getShape().end());
  return builder.create<ReshapeOp>(loc, type, input,
                                   builder.getI32ArrayAttr(shape));
}

// Whether `operand`'s leading dimensions broadcast to `result`'s.
bool broadcastsTo(RankedTensorType operand, RankedTensorType result) {
  if (operand.getRank() != result.getRank()) {
    return false;
  }
  for (auto [dim, full] : llvm::zip(operand.getShape().drop_back(),
                                    result.getShape().drop_back())) {
    if (dim != 1 && dim != full) {
      return false;
    }
  }
  return true;
}

// How an operand of an elementwise op with the wide result `result` enters
// the chunked op: as a wide value of the result's width (chunked with it), as
// a per-row value [..., h, 1] (reshaped to [..., h, 1, 1]), or not at all.
enum class Operand { Wide, Row, Unsupported };
Operand elementwiseOperand(Type type, RankedTensorType result) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || !broadcastsTo(tensor, result)) {
    return Operand::Unsupported;
  }
  if (tensor.getShape().back() == 1) {
    return Operand::Row;
  }
  if (tensor.getShape().back() == result.getShape().back() && isWide(tensor)) {
    return Operand::Wide;
  }
  return Operand::Unsupported;
}

bool isChunkableElementwise(Operation *op) {
  if (!(isa<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>(op) ||
        isa<ClampScalarOp>(op)) ||
      op->getNumResults() != 1 || !isWide(op->getResult(0).getType())) {
    return false;
  }
  auto result = cast<RankedTensorType>(op->getResult(0).getType());
  return llvm::all_of(op->getOperandTypes(), [&](Type type) {
    return elementwiseOperand(type, result) != Operand::Unsupported;
  });
}

// Whether `op` is an elementwise op (whose result may have the chunked type
// by now).
bool isElementwiseOp(Operation *op) {
  return (isa<ElementwiseUnary, ElementwiseBinary, ElementwiseTernary>(op) ||
          isa<ClampScalarOp>(op)) &&
         op->getNumResults() == 1;
}

bool isChunkableBroadcast(Operation *op) {
  auto broadcast = dyn_cast<BroadcastOp>(op);
  if (!broadcast || !isWide(broadcast.getType())) {
    return false;
  }
  auto result = cast<RankedTensorType>(broadcast.getType());
  auto input = cast<RankedTensorType>(broadcast.getInput().getType());
  return input.getRank() == result.getRank() &&
         (input.getShape().back() == 1 ||
          (input.getShape().back() == result.getShape().back() &&
           isWide(input)));
}

// An argmax over the last dimension of a wide FP32 tensor: masked_reduce
// finds it in the chunked shape as the first element at the maximum.
bool isChunkableArgMax(Operation *op) {
  auto argmax = dyn_cast<ArgMaxOp>(op);
  if (!argmax || !isWide(argmax.getInput().getType()) ||
      !cast<RankedTensorType>(argmax.getInput().getType())
           .getElementType()
           .isF32()) {
    return false;
  }
  auto dims = argmax.getDimArgAttr();
  auto input = cast<RankedTensorType>(argmax.getInput().getType());
  if (!dims || dims.size() != 1) {
    return false;
  }
  const int64_t dim = cast<IntegerAttr>(dims[0]).getInt();
  return dim == input.getRank() - 1 || dim == -1;
}

// A mesh partition of a wide tensor's rows into wide shards whose chunks
// split evenly: the chunked tensor partitions along its chunks.
bool isChunkableMeshPartition(Operation *op) {
  auto partition = dyn_cast<MeshPartitionOp>(op);
  if (!partition || !isWide(partition.getInput().getType()) ||
      !isWide(partition.getResult().getType())) {
    return false;
  }
  auto input = cast<RankedTensorType>(partition.getInput().getType());
  const int64_t dim = partition.getDim();
  return dim == input.getRank() - 1 || dim == -1;
}

// The reductions TTNN runs over two dimensions (not prod).
bool isChunkableReduction(Operation *op) {
  if (!isa<SumOp, MaxOp, MinOp, MeanOp>(op) ||
      !isWide(op->getOperand(0).getType())) {
    return false;
  }
  auto dims = op->getAttrOfType<ArrayAttr>("dim_arg");
  auto input = cast<RankedTensorType>(op->getOperand(0).getType());
  if (!dims || dims.size() != 1) {
    return false;
  }
  const int64_t dim = cast<IntegerAttr>(dims[0]).getInt();
  return dim == input.getRank() - 1 || dim == -1;
}

// A masked reduction with a [..., H] threshold for its [..., H, W] input:
// libtt's masked_reduce reduces a [..., H, W / 32, 32] input with the same
// threshold per [W / 32, 32] matrix.
bool isChunkableMaskedReduce(Operation *op) {
  auto masked = dyn_cast<MaskedReduceOp>(op);
  if (!masked || !isWide(masked.getInput().getType())) {
    return false;
  }
  auto input = cast<RankedTensorType>(masked.getInput().getType());
  auto threshold = cast<RankedTensorType>(masked.getThreshold().getType());
  return threshold.getShape() == input.getShape().drop_back();
}

bool isChunkableConstant(Operation *op) {
  if (!isa<ConstantOp, FullOp, ZerosOp, OnesOp>(op) ||
      !isWide(op->getResult(0).getType())) {
    return false;
  }
  auto constant = dyn_cast<ConstantOp>(op);
  return !constant || isa<DenseElementsAttr>(constant.getValue());
}

// Whether `user` takes a chunked operand without a reshape.
bool takesChunked(Operation *user) {
  return isChunkableElementwise(user) || isChunkableBroadcast(user) ||
         isChunkableReduction(user) || isChunkableMaskedReduce(user) ||
         isChunkableArgMax(user) || isChunkableMeshPartition(user);
}

// How a region produces or consumes a value: unknown uses count as edges.
enum class Role { None, Chunked, Edge };

class DenseRows {
public:
  explicit DenseRows(ModuleOp module) : module(module) {}

  void run() {
    findCallees();
    module.walk([&](Operation *op) { classify(op); });
    module.walk([&](func::FuncOp func) {
      for (BlockArgument arg : func.getArguments()) {
        if (isWide(arg.getType()) && !linkedArgs.contains(arg)) {
          addValue(arg);
          setProducer(arg, Role::Edge);
        }
      }
    });
    rewrite();
  }

private:
  // Union-find over the wide values.
  Value find(Value value) {
    Value root = value;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[value] != root) {
      Value next = parent[value];
      parent[value] = root;
      value = next;
    }
    return root;
  }
  void addValue(Value value) { parent.try_emplace(value, value); }
  void unite(Value a, Value b) {
    addValue(a);
    addValue(b);
    a = find(a);
    b = find(b);
    if (a != b) {
      parent[b] = a;
    }
  }

  // A chunked role, once found, wins over an edge one (a case branch's
  // return op is visited after the case that links it, or before).
  void setProducer(Value value, Role role) {
    auto [it, inserted] = producers.try_emplace(value, role);
    if (!inserted && role == Role::Chunked) {
      it->second = role;
    }
  }
  void setConsumer(OpOperand &use, Role role) {
    auto [it, inserted] = consumers.try_emplace(&use, role);
    if (!inserted && role == Role::Chunked) {
      it->second = role;
    }
  }
  bool isChunkedUse(OpOperand &use) {
    return consumers.lookup(&use) == Role::Chunked;
  }

  // The case branches and loop bodies that have one caller, which this pass
  // may retype along with it, and how often each runs per run of its caller.
  void findCallees() {
    auto single = [&](FlatSymbolRefAttr symbol) -> func::FuncOp {
      auto func = module.lookupSymbol<func::FuncOp>(symbol);
      if (!func) {
        return nullptr;
      }
      std::optional<SymbolTable::UseRange> uses =
          SymbolTable::getSymbolUses(func, module);
      if (!uses || !llvm::hasSingleElement(*uses)) {
        return nullptr;
      }
      return func;
    };
    module.walk([&](Operation *op) {
      if (auto caseOp = dyn_cast<ttcore::CaseOp>(op)) {
        SmallVector<func::FuncOp> branches;
        for (Attribute attr : caseOp.getBranchPrograms()) {
          func::FuncOp branch = single(cast<FlatSymbolRefAttr>(attr));
          if (!branch ||
              branch.getNumArguments() != caseOp.getInputs().size() ||
              branch.getNumResults() != caseOp.getNumResults()) {
            return;
          }
          branches.push_back(branch);
        }
        for (func::FuncOp branch : branches) {
          callers[branch] = {op, 1.0 / branches.size()};
        }
        caseBranches[op] = branches;
      } else if (auto loop = dyn_cast<ttcore::WhileLoopOp>(op)) {
        if (loop.getConditionProgramAttr()) {
          return;
        }
        func::FuncOp body = single(loop.getBodyProgramAttr());
        if (body && body.getNumArguments() == loop.getInputs().size()) {
          callers[body] = {op, kLoopWeight};
          loopBody[op] = body;
        }
      }
    });
  }

  double weightOf(Operation *op) {
    auto func = op->getParentOfType<func::FuncOp>();
    double weight = 1;
    for (int depth = 0; func && depth < 16; ++depth) {
      auto it = callers.find(func);
      if (it == callers.end()) {
        break;
      }
      weight *= it->second.second;
      func = it->second.first->getParentOfType<func::FuncOp>();
    }
    return weight;
  }

  double weightAt(Value value) {
    if (Operation *def = value.getDefiningOp()) {
      return weightOf(def);
    }
    return weightOf(&value.getParentBlock()->front());
  }

  void classify(Operation *op) {
    if (isa<func::FuncOp, ModuleOp>(op)) {
      return;
    }
    if (isChunkableElementwise(op)) {
      Value out = op->getResult(0);
      addValue(out);
      setProducer(out, Role::Chunked);
      auto result = cast<RankedTensorType>(out.getType());
      for (OpOperand &use : op->getOpOperands()) {
        if (elementwiseOperand(use.get().getType(), result) == Operand::Wide) {
          unite(out, use.get());
          setConsumer(use, Role::Chunked);
        }
      }
      computeOps.push_back(op);
      return;
    }
    if (isChunkableBroadcast(op)) {
      Value out = op->getResult(0);
      addValue(out);
      setProducer(out, Role::Chunked);
      if (isWide(op->getOperand(0).getType())) {
        unite(out, op->getOperand(0));
        setConsumer(op->getOpOperand(0), Role::Chunked);
      }
      return;
    }
    if (isChunkableConstant(op)) {
      addValue(op->getResult(0));
      setProducer(op->getResult(0), Role::Chunked);
      return;
    }
    if (isChunkableMeshPartition(op)) {
      Value out = op->getResult(0);
      addValue(out);
      setProducer(out, Role::Chunked);
      unite(out, op->getOperand(0));
      setConsumer(op->getOpOperand(0), Role::Chunked);
      computeOps.push_back(op);
      return;
    }
    // libtt's native generator numbers the bits by their row-major index, so
    // any shape of the same elements holds the same bits.
    if (auto rng = dyn_cast<RngBitGeneratorOp>(op);
        rng && isWide(rng.getOutput().getType()) &&
        !isWide(rng.getInitialState().getType())) {
      addValue(rng.getOutput());
      setProducer(rng.getOutput(), Role::Chunked);
      markEdgeResults(op);
      return;
    }
    if (isChunkableReduction(op) || isChunkableMaskedReduce(op) ||
        isChunkableArgMax(op)) {
      addValue(op->getOperand(0));
      setConsumer(op->getOpOperand(0), Role::Chunked);
      computeOps.push_back(op);
      return;
    }
    if (auto caseOp = dyn_cast<ttcore::CaseOp>(op)) {
      // Edges for now: linkCases joins a case to its branches where both
      // sides of a link are chunked anyway.
      if (caseBranches.contains(op)) {
        cases.push_back(op);
      }
    }
    if (auto loop = dyn_cast<ttcore::WhileLoopOp>(op)) {
      auto it = loopBody.find(op);
      if (it != loopBody.end()) {
        llvm::DenseSet<int32_t> carried(loop.getOutputIndices().begin(),
                                        loop.getOutputIndices().end());
        for (OpOperand &use : op->getOpOperands()) {
          if (!isWide(use.get().getType())) {
            continue;
          }
          addValue(use.get());
          const unsigned index = use.getOperandNumber();
          BlockArgument arg = it->second.getArgument(index);
          // Only a loop-invariant input every use of which in the body takes
          // it chunked: a reshape in the body would run every iteration.
          if (carried.contains(index) ||
              !llvm::all_of(arg.getUses(), [](OpOperand &argUse) {
                return takesChunked(argUse.getOwner());
              })) {
            setConsumer(use, Role::Edge);
            continue;
          }
          unite(use.get(), arg);
          setConsumer(use, Role::Chunked);
          linkedArgs.insert(arg);
          setProducer(arg, Role::Chunked);
        }
        markEdgeResults(op);
        return;
      }
    }
    for (OpOperand &use : op->getOpOperands()) {
      if (isWide(use.get().getType())) {
        addValue(use.get());
        setConsumer(use, Role::Edge);
      }
    }
    markEdgeResults(op);
  }

  bool isChunked(Value value) { return chunkedRoots.contains(find(value)); }
  void uniteChunked(Value a, Value b) {
    const bool chunk = isChunked(a) || isChunked(b);
    unite(a, b);
    if (chunk) {
      chunkedRoots.insert(find(a));
    }
  }

  // A case passes a chunked operand on chunked (its branches' arguments join
  // the operand's region), and returns a result chunked when every branch
  // returns it chunked. Chunking flows only from a side that chose it, so an
  // unlikely branch's ops never pull reshapes onto the path around them.
  void linkCases() {
    for (bool changed = true; changed;) {
      changed = false;
      for (Operation *op : cases) {
        auto caseOp = cast<ttcore::CaseOp>(op);
        ArrayRef<func::FuncOp> branches = caseBranches[op];
        for (OpOperand &use : caseOp.getInputsMutable()) {
          if (!isWide(use.get().getType()) || isChunkedUse(use) ||
              !isChunked(use.get())) {
            continue;
          }
          const unsigned index =
              use.getOperandNumber() - caseOp.getInputs().getBeginOperandIndex();
          for (func::FuncOp branch : branches) {
            BlockArgument arg = branch.getArgument(index);
            uniteChunked(use.get(), arg);
            setProducer(arg, Role::Chunked);
          }
          setConsumer(use, Role::Chunked);
          changed = true;
        }
        for (OpResult result : caseOp->getResults()) {
          if (!isWide(result.getType()) ||
              producers.lookup(result) == Role::Chunked) {
            continue;
          }
          auto operandOf = [&](func::FuncOp branch) -> OpOperand & {
            auto ret = cast<func::ReturnOp>(branch.front().getTerminator());
            return ret->getOpOperand(result.getResultNumber());
          };
          if (!llvm::all_of(branches, [&](func::FuncOp branch) {
                return isChunked(operandOf(branch).get());
              })) {
            continue;
          }
          for (func::FuncOp branch : branches) {
            OpOperand &operand = operandOf(branch);
            uniteChunked(result, operand.get());
            setConsumer(operand, Role::Chunked);
          }
          setProducer(result, Role::Chunked);
          changed = true;
        }
      }
    }
  }

  void markEdgeResults(Operation *op) {
    for (Value result : op->getResults()) {
      if (isWide(result.getType())) {
        addValue(result);
        setProducer(result, Role::Edge);
      }
    }
  }

  void rewrite() {
    struct Region {
      SmallVector<Value> values;
      double benefit = 0;
      double cost = 0;
    };
    llvm::MapVector<Value, Region> regions;
    SmallVector<Value> all;
    for (auto &entry : parent) {
      all.push_back(entry.first);
    }
    for (Value value : all) {
      Region &region = regions[find(value)];
      region.values.push_back(value);
      const bool edge = producers.lookup(value) != Role::Chunked;
      // An edge value with a chunked use is reshaped into the region; a
      // member with a use outside it is reshaped back out.
      const bool reshaped = llvm::any_of(value.getUses(), [&](OpOperand &use) {
        return isChunkedUse(use) == edge;
      });
      if (reshaped) {
        region.cost += weightAt(value);
      }
    }
    for (Operation *op : computeOps) {
      Value anchor = isWide(op->getResult(0).getType()) ? op->getResult(0)
                                                         : op->getOperand(0);
      regions[find(anchor)].benefit += weightOf(op);
    }

    const char *debug = std::getenv("LIBTT_DENSE_ROWS_DEBUG");
    for (auto &[root, region] : regions) {
      const bool chunk = region.benefit > kMinGain * region.cost;
      if (debug && region.benefit > 0) {
        llvm::errs() << "ttir-dense-rows: region of " << region.values.size()
                     << " values at " << root.getLoc() << ": benefit "
                     << region.benefit << ", cost " << region.cost
                     << (chunk ? ", chunked\n" : ", kept\n");
      }
      if (chunk) {
        chunkedRoots.insert(root);
      }
    }
    linkCases();

    llvm::MapVector<Value, SmallVector<Value>> chunked;
    for (Value value : all) {
      Value root = find(value);
      if (chunkedRoots.contains(root)) {
        chunked[root].push_back(value);
      }
    }
    llvm::SetVector<func::FuncOp> retyped;
    for (auto &[root, values] : chunked) {
      chunkRegion(values, retyped);
    }
    for (func::FuncOp func : retyped) {
      auto ret = cast<func::ReturnOp>(func.front().getTerminator());
      func.setType(FunctionType::get(module.getContext(),
                                     func.front().getArgumentTypes(),
                                     ret.getOperandTypes()));
    }
  }

  void chunkRegion(ArrayRef<Value> values,
                   llvm::SetVector<func::FuncOp> &retyped) {
    OpBuilder builder(module.getContext());
    auto setAfterDef = [&](Value value) {
      if (Operation *def = value.getDefiningOp()) {
        builder.setInsertionPointAfter(def);
      } else {
        builder.setInsertionPointToStart(value.getParentBlock());
      }
    };

    // Values entering the region, reshaped once for their chunked uses.
    SmallVector<Value> members;
    llvm::SetVector<Operation *> users;
    for (Value value : values) {
      if (producers.lookup(value) == Role::Chunked) {
        members.push_back(value);
        continue;
      }
      SmallVector<OpOperand *> chunkedUses;
      for (OpOperand &use : value.getUses()) {
        if (isChunkedUse(use)) {
          chunkedUses.push_back(&use);
        }
      }
      if (chunkedUses.empty()) {
        continue;
      }
      setAfterDef(value);
      Value chunked = createReshape(builder, value.getLoc(), value,
                                    chunkedType(value.getType()));
      for (OpOperand *use : chunkedUses) {
        use->set(chunked);
        users.insert(use->getOwner());
      }
    }

    // The members take the chunked type; their uses outside the region a
    // reshape back.
    SmallVector<std::pair<Value, RankedTensorType>> wideTypes;
    for (Value value : members) {
      auto wide = cast<RankedTensorType>(value.getType());
      SmallVector<OpOperand *> edgeUses;
      for (OpOperand &use : value.getUses()) {
        if (!isChunkedUse(use)) {
          edgeUses.push_back(&use);
        } else {
          users.insert(use.getOwner());
        }
      }
      value.setType(chunkedType(wide));
      wideTypes.push_back({value, wide});
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        retyped.insert(cast<func::FuncOp>(arg.getOwner()->getParentOp()));
      }
      if (!edgeUses.empty()) {
        setAfterDef(value);
        Value back = createReshape(builder, value.getLoc(), value, wide);
        for (OpOperand *use : edgeUses) {
          use->set(back);
        }
      }
    }
    for (auto [value, wide] : wideTypes) {
      if (Operation *def = value.getDefiningOp()) {
        fixProducer(def, wide);
      }
    }
    for (Operation *user : users) {
      fixConsumer(user, retyped);
    }
  }

  // A member's defining op, once the member has the chunked type.
  void fixProducer(Operation *def, RankedTensorType wide) {
    OpBuilder builder(def);
    if (auto broadcast = dyn_cast<BroadcastOp>(def)) {
      SmallVector<int64_t> dims(broadcast.getBroadcastDimensions());
      const int64_t repeat = dims.pop_back_val();
      auto input = cast<RankedTensorType>(broadcast.getInput().getType());
      if (input.getRank() == wide.getRank() + 1) {
        // A chunked input: nothing to repeat along the chunks.
        dims.append({1, 1});
      } else {
        broadcast->setOperand(0, createReshape(builder, def->getLoc(),
                                               broadcast.getInput(),
                                               rowOperandType(input)));
        dims.append({repeat / kChunk, kChunk});
      }
      broadcast.setBroadcastDimensions(dims);
      return;
    }
    if (auto constant = dyn_cast<ConstantOp>(def)) {
      constant.setValueAttr(cast<DenseElementsAttr>(constant.getValue())
                                .reshape(cast<ShapedType>(constant.getType())));
      return;
    }
    if (auto partition = dyn_cast<MeshPartitionOp>(def)) {
      // Along the chunks, at the same (positive) index.
      partition.setDim(wide.getRank() - 1);
      return;
    }
    if (isa<FullOp, ZerosOp, OnesOp>(def)) {
      auto type = cast<RankedTensorType>(def->getResult(0).getType());
      SmallVector<int32_t> shape(type.getShape().begin(), type.getShape().end());
      def->setAttr("shape", builder.getDenseI32ArrayAttr(shape));
      return;
    }
    if (def->getNumResults() == 1 && !isa<ttcore::CaseOp>(def) &&
        isElementwiseOp(def)) {
      // Elementwise: per-row operands gain the chunk dimension.
      for (OpOperand &operand : def->getOpOperands()) {
        auto type = cast<RankedTensorType>(operand.get().getType());
        if (elementwiseOperand(type, wide) == Operand::Row) {
          operand.set(createReshape(builder, def->getLoc(), operand.get(),
                                    rowOperandType(type)));
        }
      }
    }
  }

  // An op with a chunked operand, once the members have the chunked type.
  void fixConsumer(Operation *user, llvm::SetVector<func::FuncOp> &retyped) {
    if (auto ret = dyn_cast<func::ReturnOp>(user)) {
      retyped.insert(ret->getParentOfType<func::FuncOp>());
      return;
    }
    if (isa<ArgMaxOp>(user)) {
      rewriteArgMax(cast<ArgMaxOp>(user));
      return;
    }
    if (!isa<SumOp, MaxOp, MinOp, MeanOp>(user)) {
      return;
    }
    auto input = cast<RankedTensorType>(user->getOperand(0).getType());
    if (isa<SumOp, MaxOp>(user) && input.getElementType().isF32()) {
      // masked_reduce reduces each row's [N, 32] matrix in one pass on all
      // cores; TTNN's two-dimensional reduction does not parallelize these.
      rewriteAsMaskedReduce(user, isa<SumOp>(user) ? ttcore::MaskedMode::Total
                                                   : ttcore::MaskedMode::Max);
      return;
    }
    // A reduction over the last dimension now reduces the last two.
    OpBuilder builder(user->getContext());
    const int32_t rank = input.getRank();
    user->setAttr("dim_arg", builder.getI32ArrayAttr({rank - 2, rank - 1}));
    if (user->getAttrOfType<BoolAttr>("keep_dim").getValue()) {
      Value result = user->getResult(0);
      auto kept = cast<RankedTensorType>(result.getType());
      result.setType(rowOperandType(kept));
      builder.setInsertionPointAfter(user);
      Value back = createReshape(builder, user->getLoc(), result, kept);
      result.replaceAllUsesExcept(back, back.getDefiningOp());
    }
  }

  // masked_reduce of the chunked `input` [..., H, N, 32] in `mode`, against
  // `threshold` ([..., H], or zeros for the modes that read none), as
  // [..., H] values of `elementType` (FP32 or I32).
  Value maskedReduce(OpBuilder &builder, Location loc, Value input,
                     Value threshold, ttcore::MaskedCompare compare,
                     ttcore::MaskedMode mode, Type elementType) {
    auto type = cast<RankedTensorType>(input.getType());
    ArrayRef<int64_t> rows = type.getShape().drop_back(2);
    if (!threshold) {
      threshold = builder.create<FullOp>(
          loc, RankedTensorType::get(rows, type.getElementType()),
          builder.getF32FloatAttr(0.0f));
    }
    return builder.create<MaskedReduceOp>(
        loc, RankedTensorType::get(rows, elementType), input, threshold,
        ttcore::MaskedCompareAttr::get(builder.getContext(), compare),
        ttcore::MaskedModeAttr::get(builder.getContext(), mode));
  }

  // `value` [..., H] in `op`'s result type (keep_dim: [..., H, 1]).
  Value asResult(OpBuilder &builder, Operation *op, Value value) {
    auto type = cast<RankedTensorType>(op->getResult(0).getType());
    auto valueType = cast<RankedTensorType>(value.getType());
    if (valueType.getElementType() != type.getElementType()) {
      value = builder.create<TypecastOp>(
          op->getLoc(),
          RankedTensorType::get(valueType.getShape(), type.getElementType()),
          value);
    }
    if (valueType.getShape() != type.getShape()) {
      value = createReshape(builder, op->getLoc(), value, type);
    }
    return value;
  }

  void rewriteAsMaskedReduce(Operation *op, ttcore::MaskedMode mode) {
    OpBuilder builder(op);
    Value result =
        maskedReduce(builder, op->getLoc(), op->getOperand(0), nullptr,
                     ttcore::MaskedCompare::GE, mode, builder.getF32Type());
    op->getResult(0).replaceAllUsesWith(asResult(builder, op, result));
    op->erase();
  }

  // argmax = the first index whose element is at least the maximum.
  void rewriteArgMax(ArgMaxOp argmax) {
    OpBuilder builder(argmax);
    Location loc = argmax.getLoc();
    Value input = argmax.getInput();
    Value max = maskedReduce(builder, loc, input, nullptr,
                             ttcore::MaskedCompare::GE, ttcore::MaskedMode::Max,
                             builder.getF32Type());
    Value index = maskedReduce(builder, loc, input, max,
                               ttcore::MaskedCompare::GE,
                               ttcore::MaskedMode::ArgFirst,
                               builder.getIntegerType(32));
    argmax.getResult().replaceAllUsesWith(asResult(builder, argmax, index));
    argmax->erase();
  }

  ModuleOp module;
  llvm::MapVector<Value, Value> parent;
  llvm::DenseMap<Value, Role> producers;
  llvm::DenseMap<OpOperand *, Role> consumers;
  SmallVector<Operation *> computeOps;
  llvm::DenseSet<Value> linkedArgs;
  llvm::DenseMap<Operation *, SmallVector<func::FuncOp>> caseBranches;
  SmallVector<Operation *> cases;
  llvm::DenseSet<Value> chunkedRoots;
  llvm::DenseMap<Operation *, func::FuncOp> loopBody;
  // A callee's caller op and how often the callee runs per caller run.
  llvm::DenseMap<func::FuncOp, std::pair<Operation *, double>> callers;
};

class TTIRDenseRowsPass
    : public PassWrapper<TTIRDenseRowsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRDenseRowsPass)

  StringRef getArgument() const final { return "ttir-dense-rows"; }
  StringRef getDescription() const final {
    return "Compute wide, short tensors in chunked, dense tiles";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<TTIRDialect>();
  }
  void runOnOperation() final {
    // LIBTT_DENSE_ROWS=0 turns the pass off, for comparisons.
    if (const char *env = std::getenv("LIBTT_DENSE_ROWS");
        env && std::string_view(env) == "0") {
      return;
    }
    DenseRows(getOperation()).run();
  }
};

} // namespace

std::unique_ptr<Pass> createTTIRDenseRowsPass() {
  return std::make_unique<TTIRDenseRowsPass>();
}

} // namespace mlir::tt::ttir
