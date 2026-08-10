// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttmlir/Target/Utils/FlatbufferObjectCache.h"
#include "ttmlir/Target/Utils/FuncOpToProgram.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::tt {
namespace {

bool isScalarInteger(Value value) {
  auto type = value ? dyn_cast<RankedTensorType>(value.getType())
                    : RankedTensorType();
  auto integerType =
      type ? dyn_cast<IntegerType>(type.getElementType()) : IntegerType();
  return integerType && type.getNumElements() == 1 &&
         integerType.getWidth() <= 32;
}

FailureOr<int64_t> getScalarInteger(Value value) {
  if (!isScalarInteger(value)) {
    return failure();
  }
  auto constant = value.getDefiningOp<stablehlo::ConstantOp>();
  if (!constant || constant.getValue().getNumElements() != 1) {
    return failure();
  }
  return (*constant.getValue().value_begin<llvm::APInt>()).getSExtValue();
}

struct LoopBound {
  int64_t constant = 0;
  std::optional<int32_t> inputIndex;
};

struct CountedLoopInfo {
  LoopBound initial;
  LoopBound limit;
  LoopBound step;
};

struct LoopOutputs {
  SmallVector<int32_t> indices;
  SmallVector<Type> types;
};

FailureOr<LoopOutputs>
computeLoopOutputs(stablehlo::WhileOp whileOp,
                   stablehlo::ReturnOp bodyReturn,
                   const TypeConverter &typeConverter) {
  LoopOutputs outputs;
  Block &bodyBlock = whileOp.getBody().front();
  for (auto [index, result] : llvm::enumerate(bodyReturn.getOperands())) {
    if (result == bodyBlock.getArgument(index)) {
      continue;
    }
    Type resultType =
        typeConverter.convertType(whileOp.getResult(index).getType());
    if (!resultType) {
      return failure();
    }
    outputs.indices.push_back(static_cast<int32_t>(index));
    outputs.types.push_back(resultType);
  }
  return outputs;
}

FailureOr<LoopBound>
resolveBound(Value value, std::optional<int32_t> directInputIndex,
             Block *invariantRegion, stablehlo::ReturnOp bodyReturn,
             ArrayRef<Value> captures, unsigned stateCount) {
  FailureOr<int64_t> constant = getScalarInteger(value);
  if (succeeded(constant)) {
    return LoopBound{.constant = *constant};
  }
  if (!isScalarInteger(value)) {
    return failure();
  }
  if (directInputIndex) {
    return LoopBound{.inputIndex = *directInputIndex};
  }
  if (invariantRegion) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (argument && argument.getOwner() == invariantRegion) {
      int32_t index = static_cast<int32_t>(argument.getArgNumber());
      Block &bodyBlock = *bodyReturn->getBlock();
      if (bodyReturn.getOperand(index) == bodyBlock.getArgument(index)) {
        return LoopBound{.inputIndex = index};
      }
    }
  }
  auto capture = llvm::find(captures, value);
  if (capture == captures.end()) {
    return failure();
  }
  return LoopBound{.inputIndex = static_cast<int32_t>(
                       stateCount +
                       std::distance(captures.begin(), capture))};
}

ttcore::LoopBoundAttr getLoopBoundAttr(Builder &builder,
                                       const LoopBound &bound) {
  IntegerAttr inputIndex;
  if (bound.inputIndex) {
    inputIndex = builder.getI32IntegerAttr(*bound.inputIndex);
  }
  return ttcore::LoopBoundAttr::get(builder.getContext(), bound.constant,
                                    inputIndex);
}

// Recognize the canonical counted while emitted by JAX. Its scalar integer
// counter, limit, and positive step may be constants or runtime values.
FailureOr<CountedLoopInfo>
analyzeCountedLoop(stablehlo::WhileOp whileOp,
                   stablehlo::ReturnOp condReturn,
                   stablehlo::ReturnOp bodyReturn,
                   ArrayRef<Value> captures) {
  Region &cond = whileOp.getCond();
  Region &body = whileOp.getBody();
  auto compare =
      condReturn.getOperand(0).getDefiningOp<stablehlo::CompareOp>();
  auto counter =
      compare ? dyn_cast<BlockArgument>(compare.getLhs()) : BlockArgument();
  std::optional<stablehlo::ComparisonType> comparisonType =
      compare ? compare.getCompareType() : std::nullopt;
  if (!compare ||
      compare.getComparisonDirection() != stablehlo::ComparisonDirection::LT ||
      !comparisonType ||
      *comparisonType != stablehlo::ComparisonType::SIGNED || !counter ||
      counter.getOwner() != &cond.front()) {
    return failure();
  }

  unsigned counterIndex = counter.getArgNumber();
  FailureOr<LoopBound> initial = resolveBound(
      whileOp->getOperand(counterIndex), static_cast<int32_t>(counterIndex),
      /*invariantRegion=*/nullptr, bodyReturn, captures,
      whileOp.getNumOperands());
  FailureOr<LoopBound> limit =
      resolveBound(compare.getRhs(), /*directInputIndex=*/std::nullopt,
                   &cond.front(), bodyReturn, captures,
                   whileOp.getNumOperands());

  Value bodyCounter = body.front().getArgument(counterIndex);
  auto increment = bodyReturn.getOperand(counterIndex)
                       .getDefiningOp<stablehlo::AddOp>();
  Value step;
  if (increment && increment.getLhs() == bodyCounter) {
    step = increment.getRhs();
  } else if (increment && increment.getRhs() == bodyCounter) {
    step = increment.getLhs();
  }
  FailureOr<LoopBound> resolvedStep =
      resolveBound(step, /*directInputIndex=*/std::nullopt, &body.front(),
                   bodyReturn, captures, whileOp.getNumOperands());
  if (failed(initial) || failed(limit) || failed(resolvedStep) ||
      (!resolvedStep->inputIndex && resolvedStep->constant <= 0)) {
    return failure();
  }
  return CountedLoopInfo{.initial = *initial,
                         .limit = *limit,
                         .step = *resolvedStep};
}

func::FuncOp outlineRegion(ConversionPatternRewriter &rewriter,
                           Operation *controlFlowOp, Region &region,
                           llvm::StringRef suffix, ValueRange inputs,
                           ArrayRef<Value> captures, TypeRange resultTypes,
                           ArrayRef<int32_t> resultIndices) {
  OpBuilder::InsertionGuard guard(rewriter);
  ModuleOp module = controlFlowOp->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(module);
  auto parentFunc = controlFlowOp->getParentOfType<func::FuncOp>();
  std::string name = parentFunc.getSymName().str() + suffix.str();
  rewriter.setInsertionPointToEnd(module.getBody());
  auto function = rewriter.create<func::FuncOp>(
      controlFlowOp->getLoc(), name,
      rewriter.getFunctionType(inputs.getTypes(), resultTypes));
  symbolTable.insert(function);
  function.setPrivate();
  ttmlir::utils::setFunctionType(
      function, ttmlir::utils::FunctionType::ForwardDevice);

  Block *entry = function.addEntryBlock();
  IRMapping mapping;
  for (auto [source, target] :
       llvm::zip_equal(region.front().getArguments(),
                       entry->getArguments().take_front(
                           region.front().getNumArguments()))) {
    mapping.map(source, target);
  }
  for (auto [source, target] : llvm::zip_equal(
           captures, entry->getArguments().take_back(captures.size()))) {
    mapping.map(source, target);
  }
  rewriter.setInsertionPointToEnd(entry);
  for (Operation &op : region.front()) {
    rewriter.clone(op, mapping);
  }
  auto clonedReturn = cast<stablehlo::ReturnOp>(entry->getTerminator());
  rewriter.setInsertionPoint(clonedReturn);
  SmallVector<Value> results;
  for (int32_t index : resultIndices) {
    results.push_back(clonedReturn.getOperand(index));
  }
  rewriter.replaceOpWithNewOp<func::ReturnOp>(clonedReturn, results);
  return function;
}

// Outline a while loop as private body and optional condition programs. The
// runtime uses static bounds when counted-loop analysis succeeds; otherwise it
// evaluates the outlined scalar condition between body invocations.
class WhileOpConversionPattern
    : public OpConversionPattern<stablehlo::WhileOp> {
public:
  using OpConversionPattern<stablehlo::WhileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::WhileOp whileOp,
                  stablehlo::WhileOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Region &cond = whileOp.getCond();
    Region &body = whileOp.getBody();
    if (!cond.hasOneBlock() || !body.hasOneBlock()) {
      return failure();
    }

    auto condReturn =
        dyn_cast<stablehlo::ReturnOp>(cond.front().getTerminator());
    auto bodyReturn =
        dyn_cast<stablehlo::ReturnOp>(body.front().getTerminator());
    if (!condReturn || condReturn.getNumOperands() != 1 || !bodyReturn) {
      return failure();
    }
    auto conditionType =
        dyn_cast<RankedTensorType>(condReturn.getOperand(0).getType());
    if (!conditionType || conditionType.getNumElements() != 1 ||
        !conditionType.getElementType().isInteger(1)) {
      return failure();
    }

    FailureOr<LoopOutputs> outputs =
        computeLoopOutputs(whileOp, bodyReturn, *getTypeConverter());
    if (failed(outputs)) {
      return failure();
    }

    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(whileOp->getRegions(), captures);
    SmallVector<Value> convertedCaptures;
    if (failed(rewriter.getRemappedValues(captures.getArrayRef(),
                                          convertedCaptures))) {
      return failure();
    }
    SmallVector<Value> loopOperands(adaptor.getOperands());
    llvm::append_range(loopOperands, convertedCaptures);

    FailureOr<CountedLoopInfo> countedLoop = analyzeCountedLoop(
        whileOp, condReturn, bodyReturn, captures.getArrayRef());
    Type conditionResultType;
    if (failed(countedLoop)) {
      conditionResultType = getTypeConverter()->convertType(conditionType);
      if (!conditionResultType) {
        return failure();
      }
    }
    func::FuncOp bodyFunction = outlineRegion(
        rewriter, whileOp, body,
        succeeded(countedLoop) ? "_counted_loop" : "_while_body", loopOperands,
        captures.getArrayRef(), TypeRange(outputs->types), outputs->indices);

    FlatSymbolRefAttr conditionProgram;
    ttcore::LoopBoundAttr initial;
    ttcore::LoopBoundAttr limit;
    ttcore::LoopBoundAttr step;
    if (succeeded(countedLoop)) {
      initial = getLoopBoundAttr(rewriter, countedLoop->initial);
      limit = getLoopBoundAttr(rewriter, countedLoop->limit);
      step = getLoopBoundAttr(rewriter, countedLoop->step);
    } else {
      SmallVector<Type> conditionResultTypes{conditionResultType};
      SmallVector<int32_t> conditionResultIndices{0};
      func::FuncOp conditionFunction = outlineRegion(
          rewriter, whileOp, cond, "_while_condition", loopOperands,
          captures.getArrayRef(), conditionResultTypes,
          conditionResultIndices);
      conditionProgram = FlatSymbolRefAttr::get(rewriter.getContext(),
                                                conditionFunction.getSymName());
    }

    auto loop = rewriter.create<ttcore::WhileLoopOp>(
        whileOp.getLoc(), outputs->types,
        FlatSymbolRefAttr::get(rewriter.getContext(), bodyFunction.getSymName()),
        conditionProgram, initial, limit, step,
        rewriter.getI32IntegerAttr(whileOp.getNumOperands()),
        rewriter.getDenseI32ArrayAttr(outputs->indices), loopOperands);
    SmallVector<Value> replacements(adaptor.getOperands());
    for (auto [index, result] :
         llvm::zip_equal(outputs->indices, loop.getResults())) {
      replacements[index] = result;
    }
    rewriter.replaceOp(whileOp, replacements);
    return success();
  }
};

class CaseOpConversionPattern : public OpConversionPattern<stablehlo::CaseOp> {
public:
  using OpConversionPattern<stablehlo::CaseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stablehlo::CaseOp caseOp,
                  stablehlo::CaseOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(caseOp.getResultTypes(),
                                                resultTypes))) {
      return failure();
    }

    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(caseOp->getRegions(), captures);
    SmallVector<Value> convertedCaptures;
    if (failed(rewriter.getRemappedValues(captures.getArrayRef(),
                                          convertedCaptures))) {
      return failure();
    }

    for (Region &branch : caseOp.getBranches()) {
      if (!branch.hasOneBlock()) {
        return failure();
      }
      auto returnOp =
          dyn_cast<stablehlo::ReturnOp>(branch.front().getTerminator());
      if (!returnOp || returnOp.getNumOperands() != resultTypes.size()) {
        return failure();
      }
    }

    SmallVector<int32_t> resultIndices;
    resultIndices.reserve(resultTypes.size());
    for (int32_t index = 0, end = static_cast<int32_t>(resultTypes.size());
         index < end; ++index) {
      resultIndices.push_back(index);
    }

    SmallVector<Attribute> branchPrograms;
    branchPrograms.reserve(caseOp.getBranches().size());
    for (auto [index, branch] : llvm::enumerate(caseOp.getBranches())) {
      std::string suffix = "_case_branch_" + std::to_string(index);
      // StableHLO case branches have no block arguments: converted captures
      // are the function inputs, and original captures identify uses to remap.
      func::FuncOp branchFunction = outlineRegion(
          rewriter, caseOp, branch, suffix, convertedCaptures,
          captures.getArrayRef(), resultTypes, resultIndices);
      branchPrograms.push_back(FlatSymbolRefAttr::get(
          rewriter.getContext(), branchFunction.getSymName()));
    }

    rewriter.replaceOpWithNewOp<ttcore::CaseOp>(
        caseOp, resultTypes, rewriter.getArrayAttr(branchPrograms),
        adaptor.getIndex(), convertedCaptures);
    return success();
  }
};

} // namespace

void populateStableHLOControlFlowToTTIRPatterns(
    MLIRContext *context, RewritePatternSet &patterns,
    TypeConverter &typeConverter) {
  patterns.add<CaseOpConversionPattern, WhileOpConversionPattern>(
      typeConverter, context);
}

namespace ttnn {

static func::FuncOp getWhileLoopBody(ttcore::WhileLoopOp loop) {
  return SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      loop, loop.getBodyProgramAttr());
}

static SmallVector<func::FuncOp> getCaseBranches(ttcore::CaseOp caseOp) {
  SmallVector<func::FuncOp> branches;
  for (Attribute branch : caseOp.getBranchPrograms()) {
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        caseOp, cast<FlatSymbolRefAttr>(branch));
    if (!function) {
      return {};
    }
    branches.push_back(function);
  }
  return branches;
}

namespace {
class OptimizationBarrierLayoutPattern
    : public OpRewritePattern<ttcore::OptimizationBarrierOp> {
public:
  using OpRewritePattern<ttcore::OptimizationBarrierOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(ttcore::OptimizationBarrierOp barrier,
                  PatternRewriter &rewriter) const override {
    bool changed = false;
    for (auto [input, result] :
         llvm::zip_equal(barrier.getInputs(), barrier.getResults())) {
      if (input.getType() != result.getType()) {
        rewriter.modifyOpInPlace(
            barrier, [&] { result.setType(input.getType()); });
        changed = true;
      }
    }
    return success(changed);
  }
};

class WhileLoopLayoutPattern
    : public OpRewritePattern<ttcore::WhileLoopOp> {
public:
  using OpRewritePattern<ttcore::WhileLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttcore::WhileLoopOp loop,
                                PatternRewriter &rewriter) const override {
    func::FuncOp body = getWhileLoopBody(loop);
    if (!body) {
      return failure();
    }

    bool changed = false;
    for (auto [index, operand, targetType] : llvm::enumerate(
             loop.getInputs(), body.getArgumentTypes())) {
      if (operand.getType() == targetType) {
        continue;
      }
      auto tensorType = cast<RankedTensorType>(targetType);
      Value converted = ttir::utils::createDPSOp<ttir::ToLayoutOp>(
                            rewriter, loop.getLoc(), tensorType, operand,
                            nullptr)
                            ->getResult(0);
      rewriter.modifyOpInPlace(
          loop, [&] { loop->setOperand(index, converted); });
      changed = true;
    }

    for (auto [result, targetType] :
         llvm::zip_equal(loop.getResults(), body.getResultTypes())) {
      if (result.getType() != targetType) {
        rewriter.modifyOpInPlace(loop,
                                 [&] { result.setType(targetType); });
        changed = true;
      }
    }
    return success(changed);
  }
};

class CaseLayoutPattern : public OpRewritePattern<ttcore::CaseOp> {
public:
  using OpRewritePattern<ttcore::CaseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttcore::CaseOp caseOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<func::FuncOp> branches = getCaseBranches(caseOp);
    if (branches.empty()) {
      return failure();
    }

    FunctionType branchType = branches.front().getFunctionType();
    if (caseOp.getInputs().size() != branchType.getNumInputs() ||
        caseOp.getNumResults() != branchType.getNumResults() ||
        llvm::any_of(llvm::drop_begin(branches), [&](func::FuncOp branch) {
          return branch.getFunctionType() != branchType;
        })) {
      return failure();
    }

    bool changed = false;
    for (auto [index, operand, targetType] : llvm::enumerate(
             caseOp.getInputs(), branchType.getInputs())) {
      if (operand.getType() == targetType) {
        continue;
      }
      auto tensorType = cast<RankedTensorType>(targetType);
      Value converted = ttir::utils::createDPSOp<ttir::ToLayoutOp>(
                            rewriter, caseOp.getLoc(), tensorType, operand,
                            nullptr)
                            ->getResult(0);
      rewriter.modifyOpInPlace(
          caseOp, [&] { caseOp.getInputsMutable()[index].set(converted); });
      changed = true;
    }

    for (auto [index, result, targetType] : llvm::enumerate(
             caseOp.getResults(), branchType.getResults())) {
      if (result.getType() != targetType) {
        rewriter.modifyOpInPlace(caseOp,
                                 [&] { result.setType(targetType); });
        changed = true;
      }
    }
    return success(changed);
  }
};
} // namespace

void populateControlFlowLayoutPatterns(RewritePatternSet &patterns) {
  patterns.add<CaseLayoutPattern, OptimizationBarrierLayoutPattern,
               WhileLoopLayoutPattern>(patterns.getContext());
}

::flatbuffers::Offset<::tt::target::ttnn::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        ttcore::ShardStatus shardStatus,
                        std::optional<RankedTensorType> localShape);

static ::flatbuffers::Offset<::tt::target::ttnn::LoopBound>
loopBoundToFlatbuffer(FlatbufferObjectCache &cache,
                      ttcore::LoopBoundAttr bound) {
  ::flatbuffers::Optional<int32_t> inputIndex = ::flatbuffers::nullopt;
  if (IntegerAttr index = bound.getInputIndex()) {
    inputIndex = static_cast<int32_t>(index.getInt());
  }
  return ::tt::target::ttnn::CreateLoopBound(*cache.fbb, bound.getConstant(),
                                              inputIndex);
}

::flatbuffers::Offset<::tt::target::ttnn::Operation>
createWhileLoopOperation(
    FlatbufferObjectCache &cache, ttcore::WhileLoopOp loop,
    const llvm::StringMap<uint32_t> &programIndexMap,
    const std::string &debugString, const std::string &locInfo) {
  auto bodyIt = programIndexMap.find(loop.getBodyProgramAttr().getValue());
  if (bodyIt == programIndexMap.end()) {
    llvm::report_fatal_error("loop body program not found");
  }
  int32_t conditionProgram = -1;
  if (FlatSymbolRefAttr condition = loop.getConditionProgramAttr()) {
    auto conditionIt = programIndexMap.find(condition.getValue());
    if (conditionIt == programIndexMap.end()) {
      llvm::report_fatal_error("loop condition program not found");
    }
    conditionProgram = static_cast<int32_t>(conditionIt->second);
  }

  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> inputs;
  for (Value input : loop.getInputs()) {
    inputs.push_back(cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(input)));
  }

  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> outputs;
  for (Value output : loop.getResults()) {
    outputs.push_back(cache.getOrCreateNoSharding(
        output, tensorValueToFlatbuffer, std::nullopt));
  }

  ::flatbuffers::Offset<::tt::target::ttnn::LoopBound> initial;
  ::flatbuffers::Offset<::tt::target::ttnn::LoopBound> limit;
  ::flatbuffers::Offset<::tt::target::ttnn::LoopBound> step;
  if (conditionProgram < 0) {
    ttcore::LoopBoundAttr initialAttr = loop.getInitialAttr();
    ttcore::LoopBoundAttr limitAttr = loop.getLimitAttr();
    ttcore::LoopBoundAttr stepAttr = loop.getStepAttr();
    if (!initialAttr || !limitAttr || !stepAttr) {
      llvm::report_fatal_error("counted loop is missing its bounds");
    }
    initial = loopBoundToFlatbuffer(cache, initialAttr);
    limit = loopBoundToFlatbuffer(cache, limitAttr);
    step = loopBoundToFlatbuffer(cache, stepAttr);
  }
  uint32_t stateCount = static_cast<uint32_t>(loop.getStateCount());
  std::vector<uint32_t> outputIndices;
  for (int32_t index : loop.getOutputIndicesAttr().asArrayRef()) {
    outputIndices.push_back(static_cast<uint32_t>(index));
  }
  auto serializedLoop = ::tt::target::ttnn::CreateWhileLoopOpDirect(
      *cache.fbb, bodyIt->second, conditionProgram, initial, limit, step,
      stateCount, &outputIndices, &inputs, &outputs);
  return ::tt::target::ttnn::CreateOperationDirect(
      *cache.fbb,
      ::tt::target::ttnn::OpTypeTraits<
          ::tt::target::ttnn::WhileLoopOp>::enum_value,
      serializedLoop.Union(), debugString.c_str(), locInfo.c_str());
}

::flatbuffers::Offset<::tt::target::ttnn::Operation>
createCaseOperation(FlatbufferObjectCache &cache, ttcore::CaseOp caseOp,
                    const llvm::StringMap<uint32_t> &programIndexMap,
                    const std::string &debugString,
                    const std::string &locInfo) {
  std::vector<uint32_t> branchProgramIds;
  branchProgramIds.reserve(caseOp.getBranchPrograms().size());
  for (Attribute branch : caseOp.getBranchPrograms()) {
    auto symbol = cast<FlatSymbolRefAttr>(branch);
    auto programIt = programIndexMap.find(symbol.getValue());
    if (programIt == programIndexMap.end()) {
      llvm::report_fatal_error("case branch program not found");
    }
    branchProgramIds.push_back(programIt->second);
  }

  auto index = cache.at<::tt::target::ttnn::TensorRef>(
      getOperandThroughDPSOps(caseOp.getIndex()));
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> inputs;
  for (Value input : caseOp.getInputs()) {
    inputs.push_back(cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(input)));
  }
  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> outputs;
  for (Value output : caseOp.getResults()) {
    outputs.push_back(cache.getOrCreateNoSharding(
        output, tensorValueToFlatbuffer, std::nullopt));
  }

  auto serializedCase = ::tt::target::ttnn::CreateCaseOpDirect(
      *cache.fbb, &branchProgramIds, index, &inputs, &outputs);
  return ::tt::target::ttnn::CreateOperationDirect(
      *cache.fbb,
      ::tt::target::ttnn::OpTypeTraits<
          ::tt::target::ttnn::CaseOp>::enum_value,
      serializedCase.Union(), debugString.c_str(), locInfo.c_str());
}

} // namespace ttnn
} // namespace mlir::tt
