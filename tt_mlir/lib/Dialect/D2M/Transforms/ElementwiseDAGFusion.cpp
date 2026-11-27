// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// libtt's elementwise DAG fusion for D2MGenericFusion (see ttmlir/D2MFusion.h):
// a producer generic fuses into its consumer even when its results also
// escape (they become additional outputs) or several consumer operands read
// it (one operand and CB for a shared input).

#include "ttmlir/D2MFusion.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"
#include "ttmlir/Dialect/D2M/Utils/DstRegisterAnalysis.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <tuple>

namespace mlir::tt::d2m {
namespace {

// Elementwise DAG fusion keeps escaping producer results as additional outputs.
// Unlike tree fusion, shared inputs have one operand/CB and one remote load.
struct ElementwiseFusionPlan {
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  SmallVector<AffineMap> maps;
  SmallVector<OpOperand *> sources;
  SmallVector<unsigned> keptProducerResults;
  // The consumer reads a one-tile producer through this broadcast
  // map; every core recomputes the tile (its loads use tile 0 along the
  // broadcast dims).
  AffineMap broadcastProducerMap;
};

// One tile from one-tile operands (a per-token column made by
// ttir.row_to_column), on one core.
static bool isSingleTileProducer(GenericOp producer) {
  return producer.getGrid().getGridVolume() == 1 &&
         llvm::all_of(producer->getOperands(), [](Value v) {
           auto t = mlir::dyn_cast<ShapedType>(v.getType());
           return t && t.hasStaticShape() && t.getNumElements() == 1;
         });
}

static bool isSimpleElementwiseGeneric(GenericOp op) {
  if (!op.isComputeOnlyForm() || !op.hasPureTensorSemantics() ||
      !op.isAllParallel() || op.getNumRegions() != 1 ||
      !op.getRegion(0).hasOneBlock() || !op.getAdditionalArgs().empty()) {
    return false;
  }
  bool skip = false;
  op.walk([&](Operation *nested) {
    skip |= nested->hasTrait<D2MSkipOpEltwiseFusionTrait>() &&
            !isa<TileTypecastOp>(nested);
  });
  if (skip) {
    return false;
  }
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    if (!GenericOp::getOperandAlloc(op.getRegion(0),
                                    op.getOperandIndex(op->getOperand(i)))) {
      return false;
    }
  }
  // Scalar constants too (cloned like any op by fuseElementwiseDAG).
  return llvm::all_of(op.getRegion(0).front(), [](Operation &nested) {
    return isa<tensor::EmptyOp, RemoteLoadOp, RemoteStoreOp, BlockIndexOp,
               linalg::GenericOp, YieldOp, arith::ConstantOp>(nested);
  });
}

static FailureOr<ElementwiseFusionPlan>
planElementwiseFusion(GenericOp producer, GenericOp consumer) {
  if (!isSimpleElementwiseGeneric(producer) ||
      !isSimpleElementwiseGeneric(consumer) ||
      producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer) ||
      !(producer.hasCompatibleBlocking(consumer) ||
        isSingleTileProducer(producer)) ||
      producer.getIteratorTypes() != consumer.getIteratorTypes()) {
    return failure();
  }
  ElementwiseFusionPlan plan;

  // Use a common iteration space. Reductions/permutations retain the existing
  // specialized fusion path below.
  AffineMap outputMap = consumer.getIndexingMap(
      consumer.getDpsInitOperand(0)->getOperandNumber());
  for (OpOperand *input : consumer.getDpsInputOperands()) {
    auto result = dyn_cast<OpResult>(input->get());
    if (!result || result.getOwner() != producer.getOperation()) {
      continue;
    }
    AffineMap consumerMap = consumer.getIndexingMap(input->getOperandNumber());
    AffineMap producerMap = producer.getIndexingMap(
        producer.getDpsInitOperand(result.getResultNumber())
            ->getOperandNumber());
    if (producerMap != outputMap) {
      return failure();
    }
    if (consumerMap != outputMap) {
      if (!isSingleTileProducer(producer) ||
          (plan.broadcastProducerMap &&
           plan.broadcastProducerMap != consumerMap)) {
        return failure();
      }
      plan.broadcastProducerMap = consumerMap;
    }
  }
  if (!plan.broadcastProducerMap && !producer.hasCompatibleBlocking(consumer)) {
    return failure();
  }

  for (auto result : producer.getResults()) {
    bool keep = false;
    for (Operation *user : result.getUsers()) {
      if (user == consumer || producer->isProperAncestor(user) ||
          consumer->isProperAncestor(user)) {
        continue;
      }
      Operation *ancestor = consumer->getBlock()->findAncestorOpInBlock(*user);
      // Inserting at the consumer must still dominate every escaping use.
      // This rejects paths through an outside operation back into the group.
      if (!ancestor || !consumer->isBeforeInBlock(ancestor)) {
        return failure();
      }
      keep = true;
    }
    if (keep) {
      if (plan.broadcastProducerMap) {
        return failure();
      }
      plan.keptProducerResults.push_back(result.getResultNumber());
    }
  }

  auto addInput = [&](OpOperand *input, GenericOp owner) {
    Value value = input->get();
    AffineMap map = owner.getIndexingMap(input->getOperandNumber());
    auto it = llvm::find(plan.inputs, value);
    if (it != plan.inputs.end()) {
      return success(plan.maps[it - plan.inputs.begin()] == map);
    }
    plan.inputs.push_back(value);
    plan.maps.push_back(map);
    plan.sources.push_back(input);
    return success();
  };
  for (OpOperand *input : producer.getDpsInputOperands()) {
    if (plan.broadcastProducerMap) {
      // The producer's identity maps compose to the consumer's broadcast.
      if (producer.getIndexingMap(input->getOperandNumber()) != outputMap ||
          llvm::is_contained(plan.inputs, input->get())) {
        return failure();
      }
      plan.inputs.push_back(input->get());
      plan.maps.push_back(plan.broadcastProducerMap);
      plan.sources.push_back(input);
      continue;
    }
    if (failed(addInput(input, producer))) {
      return failure();
    }
  }
  for (OpOperand *input : consumer.getDpsInputOperands()) {
    if (input->get().getDefiningOp() != producer &&
        failed(addInput(input, consumer))) {
      return failure();
    }
  }
  auto addOutput = [&](OpOperand *output, GenericOp owner) {
    if (owner.getIndexingMap(output->getOperandNumber()) != outputMap) {
      return failure();
    }
    plan.outputs.push_back(output->get());
    plan.maps.push_back(outputMap);
    plan.sources.push_back(output);
    return success();
  };
  for (OpOperand &output : consumer.getDpsInitsMutable()) {
    if (failed(addOutput(&output, consumer))) {
      return failure();
    }
  }
  for (unsigned i : plan.keptProducerResults) {
    if (failed(addOutput(producer.getDpsInitOperand(i), producer))) {
      return failure();
    }
  }
  size_t numOperands = plan.inputs.size() + plan.outputs.size();
  if (numOperands > 32) {
    return failure();
  }
  bool has32bit = utils::getRegionLargestDstElemType(producer.getRegion(0))
                          .getIntOrFloatBitWidth() > 16 ||
                  utils::getRegionLargestDstElemType(consumer.getRegion(0))
                          .getIntOrFloatBitWidth() > 16;
  if (has32bit && numOperands > 8) {
    return failure();
  }
  return plan;
}

static void fuseElementwiseDAG(GenericOp producer, GenericOp consumer,
                               const ElementwiseFusionPlan &plan,
                               PatternRewriter &rewriter) {
  auto fused = rewriter.create<GenericOp>(
      consumer.getLoc(), TypeRange(plan.outputs), plan.inputs, plan.outputs,
      ValueRange{}, consumer.getGrid(), consumer.getBlockFactors(),
      rewriter.getAffineMapArrayAttr(plan.maps), consumer.getIteratorTypes(),
      consumer.getThreads(), consumer.getFabricConnectionConfigAttr(),
      /*regions=*/1);
  Block &block = fused.getRegion(0).emplaceBlock();
  rewriter.setInsertionPointToStart(&block);
  IRMapping mapping;
  DenseMap<Value, Value> operandAllocs;
  auto makeEmpty = [&](Type type) -> Value {
    auto shaped = cast<ShapedType>(type);
    return rewriter.create<tensor::EmptyOp>(fused.getLoc(), shaped.getShape(),
                                            shaped.getElementType());
  };
  // Declare operand allocations first, in operand order. Intermediate
  // allocations follow and remain distinct from the output buffers.
  for (OpOperand *source : plan.sources) {
    auto owner = cast<GenericOp>(source->getOwner());
    Value alloc = GenericOp::getOperandAlloc(
        owner.getRegion(0), owner.getOperandIndex(source->get()));
    operandAllocs[source->get()] = makeEmpty(alloc.getType());
  }
  llvm::DenseSet<Value> associatedAllocs;
  for (GenericOp original : {producer, consumer}) {
    for (OpOperand &operand : original->getOpOperands()) {
      Value alloc = GenericOp::getOperandAlloc(
          original.getRegion(0), original.getOperandIndex(operand.get()));
      associatedAllocs.insert(alloc);
      auto it = operandAllocs.find(operand.get());
      if (it != operandAllocs.end()) {
        mapping.map(alloc, it->second);
      } else if (original == producer && original.isDpsInit(&operand)) {
        mapping.map(alloc, makeEmpty(alloc.getType()));
      }
    }
  }

  // A generic may read the same input more than once, with separate local
  // load buffers. Map every such buffer to the one deduplicated input CB.
  for (GenericOp original : {producer, consumer}) {
    for (auto load : original.getRegion(0).front().getOps<RemoteLoadOp>()) {
      Value alloc = load.getLocalBuffer();
      if (!alloc.getDefiningOp<tensor::EmptyOp>()) {
        continue;
      }
      associatedAllocs.insert(alloc);
      if (auto it = operandAllocs.find(load.getMemref());
          it != operandAllocs.end()) {
        mapping.map(alloc, it->second);
      }
    }
  }

  DenseMap<Value, Value> loadedInputs;
  DenseMap<Value, Value> producerLocalResults;
  SmallVector<Value> consumerYields;
  SmallVector<Value> producerYields(producer.getNumResults());
  // The producer's compute ops (their intermediates are one tile under a
  // broadcast map).
  llvm::SmallPtrSet<Operation *, 8> producerOps;
  auto cloneBody = [&](GenericOp original) {
    for (Operation &op : original.getRegion(0).front()) {
      if (auto empty = dyn_cast<tensor::EmptyOp>(op);
          empty && associatedAllocs.contains(empty.getResult())) {
        continue;
      }
      if (auto yield = dyn_cast<YieldOp>(op)) {
        if (original == consumer) {
          for (Value value : yield.getOperands()) {
            consumerYields.push_back(mapping.lookupOrDefault(value));
          }
        }
        continue;
      }
      if (auto load = dyn_cast<RemoteLoadOp>(op)) {
        Value source = load.getMemref();
        if (auto it = producerLocalResults.find(source);
            it != producerLocalResults.end()) {
          mapping.map(load.getResult(), it->second);
          continue;
        }
        if (auto it = loadedInputs.find(source); it != loadedInputs.end()) {
          mapping.map(load.getResult(), it->second);
          continue;
        }
        auto cloned = cast<RemoteLoadOp>(rewriter.clone(op, mapping));
        if (original == producer && plan.broadcastProducerMap) {
          for (auto [k, e] :
               llvm::enumerate(plan.broadcastProducerMap.getResults())) {
            if (mlir::isa<AffineConstantExpr>(e) &&
                k < cloned.getIndices().size()) {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPoint(cloned);
              cloned->setOperand(
                  cloned.getIndicesMutable()[k].getOperandNumber(),
                  rewriter.create<arith::ConstantIndexOp>(cloned.getLoc(), 0));
            }
          }
        }
        loadedInputs[source] = cloned.getResult();
        continue;
      }
      if (auto store = dyn_cast<RemoteStoreOp>(op);
          store && original == producer) {
        unsigned i = producer.getOperandIndex(store.getMemref()) -
                     producer.getInputs().size();
        Value local = mapping.lookupOrDefault(store.getLocalBuffer());
        producerLocalResults[producer.getResult(i)] = local;
        if (llvm::is_contained(plan.keptProducerResults, i)) {
          auto cloned = cast<RemoteStoreOp>(rewriter.clone(op, mapping));
          producerYields[i] = cloned.getResult();
        } else {
          mapping.map(store.getResult(), local);
        }
        continue;
      }
      Operation *cloned = rewriter.clone(op, mapping);
      if (original == producer) {
        producerOps.insert(cloned);
      }
    }
  };
  cloneBody(producer);
  cloneBody(consumer);
  for (unsigned i : plan.keptProducerResults) {
    consumerYields.push_back(producerYields[i]);
  }
  auto yield = rewriter.create<YieldOp>(fused.getLoc(), consumerYields);
  // All input transfers must precede output transfers. Compute may wait for
  // every input before producing an escaping intermediate; an earlier remote
  // store on the same data-movement thread would deadlock that wait.
  SmallVector<RemoteStoreOp> stores;
  for (auto store : block.getOps<RemoteStoreOp>()) {
    stores.push_back(store);
  }
  for (auto store : stores) {
    for (Value index : store.getIndices()) {
      if (auto blockIndex = index.getDefiningOp<BlockIndexOp>()) {
        blockIndex->moveBefore(yield);
      }
    }
    store->moveBefore(yield);
  }
  for (Operation &op : block) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      AffineMap map = plan.broadcastProducerMap && producerOps.contains(&op)
                          ? plan.broadcastProducerMap
                          : linalgOp.getIndexingMapsArray().back();
      op.setAttr("d2m.blocking_map", AffineMapAttr::get(map));
    }
  }
  for (auto [i, result] : llvm::enumerate(consumer.getResults())) {
    rewriter.replaceAllUsesWith(result, fused.getResult(i));
  }
  for (auto [i, resultNumber] : llvm::enumerate(plan.keptProducerResults)) {
    rewriter.replaceAllUsesWith(producer.getResult(resultNumber),
                                fused.getResult(consumer.getNumResults() + i));
  }
  rewriter.eraseOp(consumer);
  // A producer result the plan did not keep can still have users.
  if (producer->use_empty()) {
    rewriter.eraseOp(producer);
  }
}

} // namespace
} // namespace mlir::tt::d2m

namespace mlir::tt::d2m_fusion {

LogicalResult fuseElementwiseProducer(d2m::GenericOp consumer,
                                      PatternRewriter &rewriter) {
  if (!d2m::isSimpleElementwiseGeneric(consumer)) {
    return failure();
  }
  for (Value input : consumer.getInputs()) {
    auto producer = input.getDefiningOp<d2m::GenericOp>();
    if (!producer) {
      continue;
    }
    auto plan = d2m::planElementwiseFusion(producer, consumer);
    if (failed(plan)) {
      continue;
    }
    d2m::fuseElementwiseDAG(producer, consumer, *plan, rewriter);
    return success();
  }
  return failure();
}

} // namespace mlir::tt::d2m_fusion
