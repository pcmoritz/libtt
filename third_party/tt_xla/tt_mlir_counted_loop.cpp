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

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::tt {
namespace {

constexpr llvm::StringLiteral kCountedLoopAttr = "tt.counted_loop";
constexpr llvm::StringLiteral kCountedLoopInitialAttr =
    "tt.counted_loop_initial";
constexpr llvm::StringLiteral kCountedLoopLimitAttr = "tt.counted_loop_limit";
constexpr llvm::StringLiteral kCountedLoopStepAttr = "tt.counted_loop_step";
constexpr llvm::StringLiteral kCountedLoopInitialIndexAttr =
    "tt.counted_loop_initial_index";
constexpr llvm::StringLiteral kCountedLoopLimitIndexAttr =
    "tt.counted_loop_limit_index";
constexpr llvm::StringLiteral kCountedLoopStepIndexAttr =
    "tt.counted_loop_step_index";
constexpr llvm::StringLiteral kCountedLoopStateCountAttr =
    "tt.counted_loop_state_count";
constexpr llvm::StringLiteral kCountedLoopOutputIndicesAttr =
    "tt.counted_loop_output_indices";
constexpr llvm::StringLiteral kLoopConditionAttr = "tt.loop_condition";

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

struct Bound {
  int64_t constant = 0;
  int32_t inputIndex = -1;
};

struct CountedLoopInfo {
  Bound initial;
  Bound limit;
  Bound step;
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

FailureOr<Bound>
resolveBound(Value value, std::optional<int32_t> directInputIndex,
             Block *invariantRegion, stablehlo::ReturnOp bodyReturn,
             ArrayRef<Value> captures, unsigned stateCount) {
  FailureOr<int64_t> constant = getScalarInteger(value);
  if (succeeded(constant)) {
    return Bound{.constant = *constant};
  }
  if (!isScalarInteger(value)) {
    return failure();
  }
  if (directInputIndex) {
    return Bound{.inputIndex = *directInputIndex};
  }
  if (invariantRegion) {
    auto argument = dyn_cast<BlockArgument>(value);
    if (argument && argument.getOwner() == invariantRegion) {
      int32_t index = static_cast<int32_t>(argument.getArgNumber());
      Block &bodyBlock = *bodyReturn->getBlock();
      if (bodyReturn.getOperand(index) == bodyBlock.getArgument(index)) {
        return Bound{.inputIndex = index};
      }
    }
  }
  auto capture = llvm::find(captures, value);
  if (capture == captures.end()) {
    return failure();
  }
  return Bound{.inputIndex = static_cast<int32_t>(
                   stateCount + std::distance(captures.begin(), capture))};
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
  FailureOr<Bound> initial = resolveBound(
      whileOp->getOperand(counterIndex), static_cast<int32_t>(counterIndex),
      /*invariantRegion=*/nullptr, bodyReturn, captures,
      whileOp.getNumOperands());
  FailureOr<Bound> limit =
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
  FailureOr<Bound> resolvedStep =
      resolveBound(step, /*directInputIndex=*/std::nullopt, &body.front(),
                   bodyReturn, captures, whileOp.getNumOperands());
  if (failed(initial) || failed(limit) || failed(resolvedStep) ||
      (resolvedStep->inputIndex < 0 && resolvedStep->constant <= 0)) {
    return failure();
  }
  return CountedLoopInfo{.initial = *initial,
                         .limit = *limit,
                         .step = *resolvedStep};
}

std::string getUniqueLoopFunctionName(stablehlo::WhileOp whileOp,
                                      llvm::StringRef suffix) {
  ModuleOp module = whileOp->getParentOfType<ModuleOp>();
  auto parentFunc = whileOp->getParentOfType<func::FuncOp>();
  std::string name = parentFunc.getSymName().str() + suffix.str();
  for (unsigned index = 0; SymbolTable::lookupSymbolIn(module, name); ++index) {
    name = parentFunc.getSymName().str() + suffix.str() + "_" +
           std::to_string(index);
  }
  return name;
}

func::FuncOp outlineLoopRegion(ConversionPatternRewriter &rewriter,
                               stablehlo::WhileOp whileOp, Region &region,
                               llvm::StringRef name, ValueRange loopOperands,
                               ArrayRef<Value> captures, TypeRange resultTypes,
                               ArrayRef<int32_t> resultIndices) {
  OpBuilder::InsertionGuard guard(rewriter);
  ModuleOp module = whileOp->getParentOfType<ModuleOp>();
  rewriter.setInsertionPointToEnd(module.getBody());
  auto function = rewriter.create<func::FuncOp>(
      whileOp.getLoc(), name,
      rewriter.getFunctionType(loopOperands.getTypes(), resultTypes));
  function.setPrivate();
  function.setNoInline(true);
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

void emitLoopCall(ConversionPatternRewriter &rewriter,
                  stablehlo::WhileOp whileOp,
                  stablehlo::WhileOp::Adaptor adaptor, llvm::StringRef bodyName,
                  TypeRange bodyResultTypes, ValueRange loopOperands,
                  ArrayRef<int32_t> outputIndices) {
  auto call = rewriter.create<func::CallOp>(
      whileOp.getLoc(), bodyName, bodyResultTypes, loopOperands);
  call.setNoInline(true);
  SmallVector<Value> replacements(adaptor.getOperands());
  for (auto [index, result] :
       llvm::zip_equal(outputIndices, call.getResults())) {
    replacements[index] = result;
  }
  rewriter.replaceOp(whileOp, replacements);
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
    getUsedValuesDefinedAbove(MutableArrayRef<Region>(&cond, 1), captures);
    getUsedValuesDefinedAbove(MutableArrayRef<Region>(&body, 1), captures);
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
    std::string bodyName = getUniqueLoopFunctionName(
        whileOp, succeeded(countedLoop) ? "_counted_loop" : "_while_body");
    func::FuncOp bodyFunc = outlineLoopRegion(
        rewriter, whileOp, body, bodyName, loopOperands,
        captures.getArrayRef(), TypeRange(outputs->types), outputs->indices);
    bodyFunc->setAttr(kCountedLoopAttr, rewriter.getUnitAttr());
    bodyFunc->setAttr(kCountedLoopStateCountAttr,
                      rewriter.getI32IntegerAttr(whileOp.getNumOperands()));
    bodyFunc->setAttr(kCountedLoopOutputIndicesAttr,
                      rewriter.getDenseI32ArrayAttr(outputs->indices));

    if (succeeded(countedLoop)) {
      bodyFunc->setAttr(
          kCountedLoopInitialAttr,
          rewriter.getI64IntegerAttr(countedLoop->initial.constant));
      bodyFunc->setAttr(
          kCountedLoopLimitAttr,
          rewriter.getI64IntegerAttr(countedLoop->limit.constant));
      bodyFunc->setAttr(kCountedLoopStepAttr,
                        rewriter.getI64IntegerAttr(countedLoop->step.constant));
      bodyFunc->setAttr(
          kCountedLoopInitialIndexAttr,
          rewriter.getI32IntegerAttr(countedLoop->initial.inputIndex));
      bodyFunc->setAttr(
          kCountedLoopLimitIndexAttr,
          rewriter.getI32IntegerAttr(countedLoop->limit.inputIndex));
      bodyFunc->setAttr(kCountedLoopStepIndexAttr,
                        rewriter.getI32IntegerAttr(countedLoop->step.inputIndex));
    } else {
      std::string conditionName =
          getUniqueLoopFunctionName(whileOp, "_while_condition");
      SmallVector<Type> conditionResultTypes{conditionResultType};
      SmallVector<int32_t> conditionResultIndices{0};
      outlineLoopRegion(rewriter, whileOp, cond, conditionName, loopOperands,
                        captures.getArrayRef(), conditionResultTypes,
                        conditionResultIndices);
      bodyFunc->setAttr(kLoopConditionAttr,
                        FlatSymbolRefAttr::get(rewriter.getContext(),
                                               conditionName));
    }

    emitLoopCall(rewriter, whileOp, adaptor, bodyName, outputs->types,
                 loopOperands, outputs->indices);
    return success();
  }
};

} // namespace

void populateStableHLOCountedLoopToTTIRPatterns(
    MLIRContext *context, RewritePatternSet &patterns,
    TypeConverter &typeConverter) {
  patterns.add<WhileOpConversionPattern>(typeConverter, context);
}

namespace ttnn {

static func::FuncOp getCountedLoopBody(func::CallOp call) {
  auto body = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  return body && body->hasAttr(kCountedLoopAttr) ? body : func::FuncOp();
}

bool isCountedLoopCall(func::CallOp call) {
  return static_cast<bool>(getCountedLoopBody(call));
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

class CountedLoopLayoutPattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    func::FuncOp body = getCountedLoopBody(call);
    if (!body) {
      return failure();
    }

    bool changed = false;
    for (auto [index, operand, targetType] : llvm::enumerate(
             call.getOperands(), body.getArgumentTypes())) {
      if (operand.getType() == targetType) {
        continue;
      }
      auto tensorType = cast<RankedTensorType>(targetType);
      Value converted = ttir::utils::createDPSOp<ttir::ToLayoutOp>(
                            rewriter, call.getLoc(), tensorType, operand,
                            nullptr)
                            ->getResult(0);
      rewriter.modifyOpInPlace(
          call, [&] { call->setOperand(index, converted); });
      changed = true;
    }

    for (auto [result, targetType] :
         llvm::zip_equal(call.getResults(), body.getResultTypes())) {
      if (result.getType() != targetType) {
        rewriter.modifyOpInPlace(call,
                                 [&] { result.setType(targetType); });
        changed = true;
      }
    }
    return success(changed);
  }
};
} // namespace

void populateCountedLoopLayoutPatterns(RewritePatternSet &patterns) {
  patterns.add<OptimizationBarrierLayoutPattern, CountedLoopLayoutPattern>(
      patterns.getContext());
}

::flatbuffers::Offset<::tt::target::ttnn::TensorRef>
tensorValueToFlatbuffer(FlatbufferObjectCache &cache, Value value,
                        ttcore::ShardStatus shardStatus,
                        std::optional<RankedTensorType> localShape);

::flatbuffers::Offset<::tt::target::ttnn::Operation>
createCountedLoopOperation(
    FlatbufferObjectCache &cache, func::CallOp call,
    const llvm::StringMap<uint32_t> &programIndexMap,
    const std::string &debugString, const std::string &locInfo) {
  func::FuncOp body = getCountedLoopBody(call);
  assert(body && "counted loop body function not found");
  auto program = programIndexMap.find(call.getCallee());
  assert(program != programIndexMap.end() && "loop body function not found");
  int32_t conditionProgram = -1;
  if (auto condition =
          body->getAttrOfType<FlatSymbolRefAttr>(kLoopConditionAttr)) {
    auto program = programIndexMap.find(condition.getValue());
    assert(program != programIndexMap.end() &&
           "loop condition function not found");
    conditionProgram = static_cast<int32_t>(program->second);
  }

  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> inputs;
  for (Value input : call.getOperands()) {
    inputs.push_back(cache.at<::tt::target::ttnn::TensorRef>(
        getOperandThroughDPSOps(input)));
  }

  std::vector<::flatbuffers::Offset<::tt::target::ttnn::TensorRef>> outputs;
  for (Value output : call.getResults()) {
    outputs.push_back(cache.getOrCreateNoSharding(
        output, tensorValueToFlatbuffer, std::nullopt));
  }

  int64_t initial = 0;
  int64_t limit = 0;
  int64_t step = 0;
  int32_t initialIndex = -1;
  int32_t limitIndex = -1;
  int32_t stepIndex = -1;
  if (conditionProgram < 0) {
    initial = body->getAttrOfType<IntegerAttr>(kCountedLoopInitialAttr).getInt();
    limit = body->getAttrOfType<IntegerAttr>(kCountedLoopLimitAttr).getInt();
    step = body->getAttrOfType<IntegerAttr>(kCountedLoopStepAttr).getInt();
    initialIndex = static_cast<int32_t>(
        body->getAttrOfType<IntegerAttr>(kCountedLoopInitialIndexAttr).getInt());
    limitIndex = static_cast<int32_t>(
        body->getAttrOfType<IntegerAttr>(kCountedLoopLimitIndexAttr).getInt());
    stepIndex = static_cast<int32_t>(
        body->getAttrOfType<IntegerAttr>(kCountedLoopStepIndexAttr).getInt());
  }
  uint32_t stateCount = static_cast<uint32_t>(
      body->getAttrOfType<IntegerAttr>(kCountedLoopStateCountAttr).getInt());
  std::vector<uint32_t> outputIndices;
  for (int32_t index : body
                           ->getAttrOfType<DenseI32ArrayAttr>(
                               kCountedLoopOutputIndicesAttr)
                           .asArrayRef()) {
    outputIndices.push_back(static_cast<uint32_t>(index));
  }
  auto loop = ::tt::target::ttnn::CreateCountedLoopOpDirect(
      *cache.fbb, program->second, conditionProgram, initial, limit, step,
      initialIndex, limitIndex, stepIndex, stateCount, &outputIndices, &inputs,
      &outputs);
  return ::tt::target::ttnn::CreateOperationDirect(
      *cache.fbb,
      ::tt::target::ttnn::OpTypeTraits<
          ::tt::target::ttnn::CountedLoopOp>::enum_value,
      loop.Union(), debugString.c_str(), locInfo.c_str());
}

} // namespace ttnn
} // namespace mlir::tt
