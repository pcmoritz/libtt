#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/executable.pb.h"
#include "mlir/sdpa_fusing_pattern.h"
#include "mlir/stablehlo_utils.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

extern "C" {

using TT_MlirAllocOutput = char* (*)(size_t size, void* user_data);

bool TT_MlirAnalyzeProgram(
    const char* format,
    size_t format_size,
    const char* code,
    size_t code_size,
    TT_MlirAllocOutput alloc_output,
    void* user_data);
}

namespace {

using mlir::func::FuncOp;
using libtt::mlir_frontend::definingOpSkippingIdentityCustomCalls;
using libtt::mlir_frontend::isIdentityCustomCall;

std::optional<uint32_t> packedConstantValue(mlir::Value value, std::string& error);

void registerDialects(mlir::MLIRContext& context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    mlir::func::registerAllExtensions(registry);
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
}

mlir::OwningOpRef<mlir::ModuleOp> parseModule(
    mlir::MLIRContext& context,
    llvm::StringRef format,
    llvm::StringRef code) {
    if (format != "mlir" && format != "stablehlo") {
        return nullptr;
    }

    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        code,
        "mlir_program",
        false);
    if (auto module = mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), &context)) {
        return module;
    }

    return mlir::parseSourceString<mlir::ModuleOp>(
        code,
        &context);
}

struct CaseIndexInfo {
    mlir::RankedTensorType tensor_type;
    mlir::IntegerType integer_type;
};

std::optional<CaseIndexInfo> validateCaseIndex(mlir::stablehlo::CaseOp case_op) {
    auto index_type = mlir::dyn_cast<mlir::RankedTensorType>(
        case_op.getIndex().getType());
    if (!index_type) {
        case_op.emitError("stablehlo.case index must be a ranked tensor");
        return std::nullopt;
    }
    auto integer_type = mlir::dyn_cast<mlir::IntegerType>(index_type.getElementType());
    if (!integer_type) {
        case_op.emitError("stablehlo.case index must have integer element type");
        return std::nullopt;
    }
    if (index_type.getRank() != 0) {
        case_op.emitError("stablehlo.case index must be a scalar tensor");
        return std::nullopt;
    }
    return CaseIndexInfo{index_type, integer_type};
}

mlir::Value createCaseIndexConstant(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    CaseIndexInfo index_info,
    uint64_t value) {
    auto attr = mlir::DenseElementsAttr::get(
        index_info.tensor_type,
        llvm::APInt(
            index_info.integer_type.getWidth(),
            value,
            !index_info.integer_type.isUnsigned()));
    return builder.create<mlir::stablehlo::ConstantOp>(loc, attr).getResult();
}

mlir::Value createCaseBranchPredicate(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::Value index,
    CaseIndexInfo index_info,
    uint64_t branch_index) {
    auto constant = createCaseIndexConstant(
        builder, loc, index_info, branch_index);
    auto pred_type = mlir::RankedTensorType::get(
        index_info.tensor_type.getShape(),
        builder.getI1Type());
    auto compare_type = mlir::stablehlo::ComparisonTypeAttr::get(
        builder.getContext(),
        index_info.integer_type.isUnsigned()
            ? mlir::stablehlo::ComparisonType::UNSIGNED
            : mlir::stablehlo::ComparisonType::SIGNED);
    return builder.create<mlir::stablehlo::CompareOp>(
        loc,
        pred_type,
        index,
        constant,
        mlir::stablehlo::ComparisonDirection::EQ,
        compare_type).getResult();
}

mlir::LogicalResult validateCasePredicateBroadcast(
    mlir::stablehlo::CaseOp case_op,
    CaseIndexInfo index_info) {
    for (mlir::OpResult result : case_op->getResults()) {
        auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
        if (!tensor_type) {
            return case_op.emitError("stablehlo.case select values must be ranked tensors");
        }
        if (index_info.tensor_type.getShape() != tensor_type.getShape() &&
            index_info.tensor_type.getRank() != 0) {
            return case_op.emitError(
                "stablehlo.case non-scalar predicates cannot be broadcast to branch result shapes");
        }
    }
    return mlir::success();
}

mlir::Value broadcastCasePredicateToResult(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::Value predicate,
    mlir::Type result_type) {
    auto pred_type = mlir::cast<mlir::RankedTensorType>(predicate.getType());
    auto tensor_type = mlir::cast<mlir::RankedTensorType>(result_type);
    if (pred_type.getShape() == tensor_type.getShape()) {
        return predicate;
    }

    auto broadcast_type = mlir::RankedTensorType::get(
        tensor_type.getShape(),
        pred_type.getElementType());
    return builder.create<mlir::stablehlo::BroadcastInDimOp>(
        loc,
        broadcast_type,
        predicate,
        llvm::ArrayRef<int64_t>{}).getResult();
}

mlir::LogicalResult validateCaseBranch(
    mlir::stablehlo::CaseOp case_op,
    mlir::Region& branch) {
    if (branch.empty() || branch.getBlocks().size() != 1) {
        return case_op.emitError("stablehlo.case branches must contain exactly one block");
    }
    mlir::Block& block = branch.front();
    if (!block.getArguments().empty()) {
        return case_op.emitError("stablehlo.case branch block arguments are not supported");
    }

    auto return_op = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
    if (!return_op) {
        return case_op.emitError("stablehlo.case branches must terminate with stablehlo.return");
    }
    if (return_op.getResults().size() != case_op->getNumResults()) {
        return case_op.emitError("stablehlo.case branch result count does not match case result count");
    }
    for (auto [result_index, result] : llvm::enumerate(return_op.getResults())) {
        if (result.getType() != case_op->getResult(result_index).getType()) {
            return case_op.emitError("stablehlo.case branch result type does not match case result type");
        }
    }
    return mlir::success();
}

void cloneCaseBranch(
    mlir::OpBuilder& builder,
    mlir::Region& branch,
    llvm::SmallVectorImpl<mlir::Value>& results) {
    mlir::Block& block = branch.front();
    auto return_op = mlir::cast<mlir::stablehlo::ReturnOp>(block.getTerminator());
    mlir::IRMapping mapper;
    for (mlir::Operation& op : block) {
        if (&op == return_op.getOperation()) {
            break;
        }
        builder.clone(op, mapper);
    }

    results.clear();
    for (mlir::Value value : return_op.getResults()) {
        results.push_back(mapper.lookupOrDefault(value));
    }
}

std::optional<uint64_t> constantCaseIndex(mlir::Value index) {
    auto constant_op = index.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!constant_op) {
        return std::nullopt;
    }
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constant_op.getValue());
    if (!dense || !dense.isSplat()) {
        return std::nullopt;
    }
    auto integer_type = mlir::dyn_cast<mlir::IntegerType>(
        mlir::cast<mlir::RankedTensorType>(index.getType()).getElementType());
    if (!integer_type) {
        return std::nullopt;
    }
    return dense.getSplatValue<llvm::APInt>().getLimitedValue();
}

std::optional<int64_t> constantScalarIntegerValue(mlir::Value value) {
    auto constant_op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!constant_op) {
        return std::nullopt;
    }
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!tensor_type || tensor_type.getRank() != 0) {
        return std::nullopt;
    }
    auto integer_type = mlir::dyn_cast<mlir::IntegerType>(tensor_type.getElementType());
    if (!integer_type || integer_type.getWidth() > 63) {
        return std::nullopt;
    }
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constant_op.getValue());
    if (!dense || !dense.isSplat()) {
        return std::nullopt;
    }

    llvm::APInt value_bits = dense.getSplatValue<llvm::APInt>();
    if (integer_type.isUnsigned()) {
        if (value_bits.getActiveBits() > 63) {
            return std::nullopt;
        }
        return static_cast<int64_t>(value_bits.getZExtValue());
    }
    return value_bits.getSExtValue();
}

mlir::LogicalResult lowerSingleCaseOpToSelects(
    mlir::stablehlo::CaseOp case_op,
    mlir::PatternRewriter& rewriter) {
    auto branches = case_op.getBranches();
    if (branches.empty()) {
        return case_op.emitError("stablehlo.case must contain at least one branch");
    }

    auto index_info = validateCaseIndex(case_op);
    if (!index_info) {
        return mlir::failure();
    }
    for (mlir::Region& branch : branches) {
        if (mlir::failed(validateCaseBranch(case_op, branch))) {
            return mlir::failure();
        }
    }

    auto constant_index = constantCaseIndex(case_op.getIndex());
    if (!constant_index &&
        mlir::failed(validateCasePredicateBroadcast(case_op, *index_info))) {
        return mlir::failure();
    }

    rewriter.setInsertionPoint(case_op);
    if (constant_index) {
        uint64_t branch_index = *constant_index < branches.size()
            ? *constant_index
            : static_cast<uint64_t>(branches.size() - 1);
        llvm::SmallVector<mlir::Value> results;
        cloneCaseBranch(
            rewriter,
            branches[static_cast<size_t>(branch_index)],
            results);
        rewriter.replaceOp(case_op, results);
        return mlir::success();
    }

    llvm::SmallVector<llvm::SmallVector<mlir::Value>> branch_results;
    branch_results.reserve(branches.size());
    for (mlir::Region& branch : branches) {
        llvm::SmallVector<mlir::Value> results;
        cloneCaseBranch(rewriter, branch, results);
        branch_results.push_back(std::move(results));
    }

    llvm::SmallVector<mlir::Value> selected(branch_results.back().begin(), branch_results.back().end());
    for (int64_t branch_index = static_cast<int64_t>(branch_results.size()) - 2;
         branch_index >= 0;
         --branch_index) {
        auto predicate = createCaseBranchPredicate(
            rewriter,
            case_op.getLoc(),
            case_op.getIndex(),
            *index_info,
            static_cast<uint64_t>(branch_index));

        for (auto [result_index, selected_value] : llvm::enumerate(selected)) {
            mlir::Value pred_for_result = broadcastCasePredicateToResult(
                rewriter,
                case_op.getLoc(),
                predicate,
                case_op->getResult(result_index).getType());
            selected[result_index] = rewriter.create<mlir::stablehlo::SelectOp>(
                case_op.getLoc(),
                case_op->getResult(result_index).getType(),
                pred_for_result,
                branch_results[branch_index][result_index],
                selected_value).getResult();
        }
    }

    rewriter.replaceOp(case_op, selected);
    return mlir::success();
}

std::optional<mlir::stablehlo::ComparisonDirection> invertComparisonDirection(
    mlir::stablehlo::ComparisonDirection direction) {
    switch (direction) {
        case mlir::stablehlo::ComparisonDirection::LT:
            return mlir::stablehlo::ComparisonDirection::GT;
        case mlir::stablehlo::ComparisonDirection::LE:
            return mlir::stablehlo::ComparisonDirection::GE;
        case mlir::stablehlo::ComparisonDirection::GT:
            return mlir::stablehlo::ComparisonDirection::LT;
        case mlir::stablehlo::ComparisonDirection::GE:
            return mlir::stablehlo::ComparisonDirection::LE;
        default:
            return std::nullopt;
    }
}

std::optional<int64_t> computeStaticTripCount(
    int64_t init,
    int64_t bound,
    int64_t step,
    mlir::stablehlo::ComparisonDirection direction) {
    if (step == 0) {
        return std::nullopt;
    }

    auto ceilDivPositive = [](auto numerator, auto denominator) -> int64_t {
        return static_cast<int64_t>((numerator + denominator - 1) / denominator);
    };

    if (step > 0) {
        switch (direction) {
            case mlir::stablehlo::ComparisonDirection::LT:
                if (init >= bound) return 0;
                return ceilDivPositive(
                    static_cast<__int128>(bound) - init,
                    static_cast<__int128>(step));
            case mlir::stablehlo::ComparisonDirection::LE:
                if (init > bound) return 0;
                return static_cast<int64_t>(
                    (static_cast<__int128>(bound) - init) / step + 1);
            case mlir::stablehlo::ComparisonDirection::GT:
                if (init > bound) return std::nullopt;
                return 0;
            case mlir::stablehlo::ComparisonDirection::GE:
                if (init >= bound) return std::nullopt;
                return 0;
            default:
                return std::nullopt;
        }
    }

    int64_t step_abs = -step;
    switch (direction) {
        case mlir::stablehlo::ComparisonDirection::GT:
            if (init <= bound) return 0;
            return ceilDivPositive(
                static_cast<__int128>(init) - bound,
                static_cast<__int128>(step_abs));
        case mlir::stablehlo::ComparisonDirection::GE:
            if (init < bound) return 0;
            return static_cast<int64_t>(
                (static_cast<__int128>(init) - bound) / step_abs + 1);
        case mlir::stablehlo::ComparisonDirection::LT:
            if (init < bound) return std::nullopt;
            return 0;
        case mlir::stablehlo::ComparisonDirection::LE:
            if (init <= bound) return std::nullopt;
            return 0;
        default:
            return std::nullopt;
    }
}

struct StaticWhilePlan {
    int64_t trip_count;
};

std::optional<int64_t> loopCounterStep(
    mlir::Value updated_value,
    mlir::BlockArgument counter_argument) {
    if (auto add_op = updated_value.getDefiningOp<mlir::stablehlo::AddOp>()) {
        if (add_op.getLhs() == counter_argument) {
            return constantScalarIntegerValue(add_op.getRhs());
        }
        if (add_op.getRhs() == counter_argument) {
            return constantScalarIntegerValue(add_op.getLhs());
        }
    }

    if (auto subtract_op = updated_value.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
        if (subtract_op.getLhs() == counter_argument) {
            auto value = constantScalarIntegerValue(subtract_op.getRhs());
            if (!value || *value == std::numeric_limits<int64_t>::min()) {
                return std::nullopt;
            }
            return -*value;
        }
    }

    return std::nullopt;
}

std::optional<StaticWhilePlan> analyzeStaticWhile(mlir::stablehlo::WhileOp while_op) {
    if (while_op.getCond().empty() || while_op.getCond().getBlocks().size() != 1 ||
        while_op.getBody().empty() || while_op.getBody().getBlocks().size() != 1) {
        while_op.emitError("stablehlo.while regions must contain exactly one block");
        return std::nullopt;
    }

    mlir::Block& cond_block = while_op.getCond().front();
    mlir::Block& body_block = while_op.getBody().front();
    if (cond_block.getNumArguments() != while_op->getNumOperands() ||
        body_block.getNumArguments() != while_op->getNumOperands()) {
        while_op.emitError("stablehlo.while block argument count must match operand count");
        return std::nullopt;
    }

    auto cond_return = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(cond_block.getTerminator());
    auto body_return = mlir::dyn_cast<mlir::stablehlo::ReturnOp>(body_block.getTerminator());
    if (!cond_return || !body_return || cond_return.getResults().size() != 1 ||
        body_return.getResults().size() != while_op->getNumResults()) {
        while_op.emitError("stablehlo.while must have stablehlo.return terminators with matching result counts");
        return std::nullopt;
    }

    auto predicate_type = mlir::dyn_cast<mlir::RankedTensorType>(
        cond_return.getResults()[0].getType());
    if (!predicate_type || predicate_type.getRank() != 0 ||
        !predicate_type.getElementType().isInteger(1)) {
        while_op.emitError("stablehlo.while condition must return a scalar predicate");
        return std::nullopt;
    }

    auto compare_op = cond_return.getResults()[0].getDefiningOp<mlir::stablehlo::CompareOp>();
    if (!compare_op) {
        while_op.emitError("only statically bounded stablehlo.while compare conditions are supported");
        return std::nullopt;
    }

    auto direction = compare_op.getComparisonDirection();
    std::optional<size_t> counter_index;
    std::optional<int64_t> bound;
    if (auto lhs_argument = mlir::dyn_cast<mlir::BlockArgument>(compare_op.getLhs())) {
        if (lhs_argument.getOwner() == &cond_block) {
            counter_index = lhs_argument.getArgNumber();
            bound = constantScalarIntegerValue(compare_op.getRhs());
        }
    }
    if (!counter_index || !bound) {
        if (auto rhs_argument = mlir::dyn_cast<mlir::BlockArgument>(compare_op.getRhs())) {
            if (rhs_argument.getOwner() == &cond_block) {
                auto inverted = invertComparisonDirection(direction);
                if (!inverted) {
                    while_op.emitError("unsupported stablehlo.while comparison direction");
                    return std::nullopt;
                }
                counter_index = rhs_argument.getArgNumber();
                bound = constantScalarIntegerValue(compare_op.getLhs());
                direction = *inverted;
            }
        }
    }
    if (!counter_index || !bound) {
        while_op.emitError("stablehlo.while condition must compare one loop counter against a scalar constant");
        return std::nullopt;
    }
    if (*counter_index >= static_cast<size_t>(while_op->getNumOperands())) {
        while_op.emitError("stablehlo.while loop counter index is out of range");
        return std::nullopt;
    }

    mlir::Value init_value = while_op->getOperand(*counter_index);
    auto init = constantScalarIntegerValue(init_value);
    if (!init) {
        while_op.emitError("stablehlo.while loop counter initial value must be a scalar constant");
        return std::nullopt;
    }

    auto counter_arg = body_block.getArgument(*counter_index);
    auto step = loopCounterStep(body_return.getResults()[*counter_index], counter_arg);
    if (!step) {
        while_op.emitError("stablehlo.while loop counter update must be add/subtract by a scalar constant");
        return std::nullopt;
    }

    auto trip_count = computeStaticTripCount(*init, *bound, *step, direction);
    if (!trip_count) {
        while_op.emitError("stablehlo.while loop trip count is not statically bounded");
        return std::nullopt;
    }
    if (*trip_count > 4096) {
        while_op.emitError("stablehlo.while static trip count is too large to unroll");
        return std::nullopt;
    }

    return StaticWhilePlan{*trip_count};
}

mlir::LogicalResult lowerSingleStaticWhileOp(
    mlir::stablehlo::WhileOp while_op,
    mlir::PatternRewriter& rewriter) {
    auto plan = analyzeStaticWhile(while_op);
    if (!plan) {
        return mlir::failure();
    }

    mlir::Block& body_block = while_op.getBody().front();
    auto body_return = mlir::cast<mlir::stablehlo::ReturnOp>(body_block.getTerminator());

    rewriter.setInsertionPoint(while_op);
    llvm::SmallVector<mlir::Value> carried(
        while_op->getOperands().begin(),
        while_op->getOperands().end());
    for (int64_t iteration = 0; iteration < plan->trip_count; ++iteration) {
        mlir::IRMapping mapper;
        for (auto [index, argument] : llvm::enumerate(body_block.getArguments())) {
            mapper.map(argument, carried[index]);
        }

        for (mlir::Operation& op : body_block) {
            if (&op == body_return.getOperation()) {
                break;
            }
            mlir::Operation* cloned = rewriter.clone(op, mapper);
            rewriter.setInsertionPointAfter(cloned);
        }

        llvm::SmallVector<mlir::Value> next_carried;
        next_carried.reserve(body_return.getResults().size());
        for (mlir::Value result : body_return.getResults()) {
            next_carried.push_back(mapper.lookupOrDefault(result));
        }
        carried = std::move(next_carried);
    }

    rewriter.replaceOp(while_op, carried);
    return mlir::success();
}

void appendOverwriteUpdateComputation(
    mlir::stablehlo::ScatterOp scatter,
    mlir::Type element_type,
    mlir::Location loc) {
    auto scalar_type = mlir::RankedTensorType::get({}, element_type);
    mlir::Block& block = scatter.getUpdateComputation().emplaceBlock();
    block.addArgument(scalar_type, loc);
    block.addArgument(scalar_type, loc);
    mlir::OpBuilder::atBlockEnd(&block)
        .create<mlir::stablehlo::ReturnOp>(loc, block.getArgument(1));
}

std::optional<mlir::Value> createIntegerSplatConstant(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::RankedTensorType tensor_type,
    uint64_t value) {
    auto integer_type = mlir::dyn_cast<mlir::IntegerType>(tensor_type.getElementType());
    if (!integer_type) {
        mlir::emitError(loc, "expected integer tensor type for splat constant");
        return std::nullopt;
    }
    auto attr = mlir::DenseElementsAttr::get(
        tensor_type,
        llvm::APInt(integer_type.getWidth(), value, !integer_type.isUnsigned()));
    return builder.create<mlir::stablehlo::ConstantOp>(loc, attr).getResult();
}

mlir::LogicalResult lowerSingleDynamicUpdateSliceToScatter(
    mlir::stablehlo::DynamicUpdateSliceOp update_slice_op,
    mlir::PatternRewriter& rewriter) {
    auto operand_type = mlir::dyn_cast<mlir::RankedTensorType>(
        update_slice_op.getOperand().getType());
    auto update_type = mlir::dyn_cast<mlir::RankedTensorType>(
        update_slice_op.getUpdate().getType());
    if (!operand_type || !update_type || !operand_type.hasStaticShape() ||
        !update_type.hasStaticShape()) {
        return update_slice_op.emitError(
            "dynamic_update_slice-to-scatter requires ranked static tensors");
    }
    if (operand_type.getRank() != 1 || update_type.getRank() != 1) {
        return update_slice_op.emitError(
            "dynamic_update_slice-to-scatter currently supports only rank-1 updates");
    }
    if (update_slice_op.getStartIndices().size() != 1) {
        return update_slice_op.emitError(
            "rank-1 dynamic_update_slice requires one start index");
    }

    int64_t operand_size = operand_type.getDimSize(0);
    int64_t update_size = update_type.getDimSize(0);
    if (update_size > operand_size) {
        return update_slice_op.emitError("dynamic_update_slice update size exceeds operand size");
    }

    mlir::Value start = update_slice_op.getStartIndices().front();
    auto start_type = mlir::dyn_cast<mlir::RankedTensorType>(start.getType());
    if (!start_type || start_type.getRank() != 0 ||
        !mlir::isa<mlir::IntegerType>(start_type.getElementType())) {
        return update_slice_op.emitError(
            "dynamic_update_slice start index must be a scalar integer tensor");
    }

    rewriter.setInsertionPoint(update_slice_op);
    mlir::Location loc = update_slice_op.getLoc();
    auto zero = createIntegerSplatConstant(rewriter, loc, start_type, 0);
    if (!zero) {
        return mlir::failure();
    }
    auto max_start = createIntegerSplatConstant(
        rewriter, loc, start_type, static_cast<uint64_t>(operand_size - update_size));
    if (!max_start) {
        return mlir::failure();
    }

    // StableHLO dynamic_update_slice clamps starts into
    // [0, operand_size - update_size]. Express min(x, hi) using max/subtract
    // because the backend already supports integer maximum.
    mlir::Value nonnegative_start = rewriter.create<mlir::stablehlo::MaxOp>(
        loc, start_type, start, *zero).getResult();
    mlir::Value excess = rewriter.create<mlir::stablehlo::SubtractOp>(
        loc, start_type, nonnegative_start, *max_start).getResult();
    mlir::Value clamped_excess = rewriter.create<mlir::stablehlo::MaxOp>(
        loc, start_type, excess, *zero).getResult();
    mlir::Value clamped_start = rewriter.create<mlir::stablehlo::SubtractOp>(
        loc, start_type, nonnegative_start, clamped_excess).getResult();

    auto indices_vector_type = mlir::RankedTensorType::get(
        {update_size}, start_type.getElementType());
    mlir::Value iota = rewriter.create<mlir::stablehlo::IotaOp>(
        loc, indices_vector_type, rewriter.getI64IntegerAttr(0)).getResult();
    mlir::Value start_vector = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
        loc,
        indices_vector_type,
        clamped_start,
        rewriter.getDenseI64ArrayAttr({})).getResult();
    mlir::Value indices_vector = rewriter.create<mlir::stablehlo::AddOp>(
        loc, indices_vector_type, iota, start_vector).getResult();
    auto indices_matrix_type = mlir::RankedTensorType::get(
        {update_size, 1}, start_type.getElementType());
    mlir::Value indices = rewriter.create<mlir::stablehlo::ReshapeOp>(
        loc, indices_matrix_type, indices_vector).getResult();

    auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*updateWindowDims=*/{},
        /*insertedWindowDims=*/{0},
        /*inputBatchingDims=*/{},
        /*scatterIndicesBatchingDims=*/{},
        /*scatterDimsToOperandDims=*/{0},
        /*indexVectorDim=*/1);
    auto scatter = rewriter.create<mlir::stablehlo::ScatterOp>(
        loc,
        update_slice_op->getResultTypes(),
        mlir::ValueRange{update_slice_op.getOperand()},
        indices,
        mlir::ValueRange{update_slice_op.getUpdate()},
        scatter_dims,
        /*indicesAreSorted=*/true,
        /*uniqueIndices=*/true);

    appendOverwriteUpdateComputation(scatter, update_type.getElementType(), loc);

    rewriter.replaceOp(update_slice_op, mlir::ValueRange{scatter.getResult(0)});
    return mlir::success();
}

std::optional<mlir::Value> createI32ScalarConstant(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    int64_t value) {
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
        mlir::emitError(loc, "i32 scalar constant value is out of range");
        return std::nullopt;
    }
    auto type = mlir::RankedTensorType::get({}, builder.getI32Type());
    return createIntegerSplatConstant(
        builder, loc, type, static_cast<uint64_t>(value));
}

bool hasZeroDimension(llvm::ArrayRef<int64_t> shape) {
    return llvm::any_of(shape, [](int64_t dim) { return dim == 0; });
}

mlir::LogicalResult lowerSinglePadToScatter(
    mlir::stablehlo::PadOp pad_op,
    mlir::PatternRewriter& rewriter) {
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        pad_op.getOperand().getType());
    auto output_type = mlir::dyn_cast<mlir::RankedTensorType>(
        pad_op.getResult().getType());
    auto padding_type = mlir::dyn_cast<mlir::RankedTensorType>(
        pad_op.getPaddingValue().getType());
    if (!input_type || !output_type || !padding_type ||
        !input_type.hasStaticShape() || !output_type.hasStaticShape()) {
        return pad_op.emitError("pad-to-scatter requires ranked static tensors");
    }
    if (padding_type.getRank() != 0) {
        return pad_op.emitError("pad-to-scatter requires a scalar padding value");
    }
    int64_t rank = input_type.getRank();
    if (output_type.getRank() != rank ||
        pad_op.getEdgePaddingLow().size() != static_cast<size_t>(rank) ||
        pad_op.getEdgePaddingHigh().size() != static_cast<size_t>(rank) ||
        pad_op.getInteriorPadding().size() != static_cast<size_t>(rank)) {
        return pad_op.emitError("pad attributes must match input/output rank");
    }
    if (input_type.getShape() == output_type.getShape() &&
        llvm::all_of(pad_op.getEdgePaddingLow(), [](int64_t value) { return value == 0; }) &&
        llvm::all_of(pad_op.getEdgePaddingHigh(), [](int64_t value) { return value == 0; }) &&
        llvm::all_of(pad_op.getInteriorPadding(), [](int64_t value) { return value == 0; })) {
        rewriter.replaceOp(pad_op, mlir::ValueRange{pad_op.getOperand()});
        return mlir::success();
    }
    if (rank == 0) {
        return pad_op.emitError("non-noop scalar pad is unsupported");
    }

    auto low = pad_op.getEdgePaddingLow();
    auto high = pad_op.getEdgePaddingHigh();
    auto interior = pad_op.getInteriorPadding();
    for (int64_t dim = 0; dim < rank; ++dim) {
        if (low[dim] < 0 || high[dim] < 0 || interior[dim] < 0) {
            return pad_op.emitError("pad-to-scatter requires non-negative padding");
        }
        int64_t expected = low[dim] + input_type.getDimSize(dim) +
                           std::max<int64_t>(input_type.getDimSize(dim) - 1, 0) * interior[dim] +
                           high[dim];
        if (output_type.getDimSize(dim) != expected) {
            return pad_op.emitError("pad output shape does not match pad attributes");
        }
    }

    rewriter.setInsertionPoint(pad_op);
    mlir::Location loc = pad_op.getLoc();
    mlir::Type element_type = input_type.getElementType();
    mlir::Value current = pad_op.getOperand();
    llvm::SmallVector<int64_t> current_shape(
        input_type.getShape().begin(),
        input_type.getShape().end());

    for (int64_t dim = 0; dim < rank; ++dim) {
        if (low[dim] == 0 && high[dim] == 0 && interior[dim] == 0) {
            continue;
        }
        int64_t step = interior[dim] + 1;
        if (current_shape[dim] > std::numeric_limits<int32_t>::max() ||
            low[dim] > std::numeric_limits<int32_t>::max() ||
            step > std::numeric_limits<int32_t>::max()) {
            return pad_op.emitError("pad-to-scatter index values exceed i32 range");
        }

        llvm::SmallVector<int64_t> next_shape = current_shape;
        next_shape[dim] = low[dim] + current_shape[dim] +
                          std::max<int64_t>(current_shape[dim] - 1, 0) * interior[dim] +
                          high[dim];
        auto next_type = mlir::RankedTensorType::get(next_shape, element_type);
        mlir::Value base = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
            loc,
            next_type,
            pad_op.getPaddingValue(),
            rewriter.getDenseI64ArrayAttr({})).getResult();
        if (hasZeroDimension(current_shape)) {
            current = base;
            current_shape = next_shape;
            continue;
        }

        auto index_vector_type = mlir::RankedTensorType::get(
            {current_shape[dim]}, rewriter.getI32Type());
        mlir::Value indices_vector = rewriter.create<mlir::stablehlo::IotaOp>(
            loc, index_vector_type, rewriter.getI64IntegerAttr(0)).getResult();
        if (step != 1) {
            auto step_scalar = createI32ScalarConstant(rewriter, loc, step);
            if (!step_scalar) {
                return mlir::failure();
            }
            mlir::Value step_vector = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
                loc,
                index_vector_type,
                *step_scalar,
                rewriter.getDenseI64ArrayAttr({})).getResult();
            indices_vector = rewriter.create<mlir::stablehlo::MulOp>(
                loc, index_vector_type, indices_vector, step_vector).getResult();
        }
        if (low[dim] != 0) {
            auto low_scalar = createI32ScalarConstant(rewriter, loc, low[dim]);
            if (!low_scalar) {
                return mlir::failure();
            }
            mlir::Value low_vector = rewriter.create<mlir::stablehlo::BroadcastInDimOp>(
                loc,
                index_vector_type,
                *low_scalar,
                rewriter.getDenseI64ArrayAttr({})).getResult();
            indices_vector = rewriter.create<mlir::stablehlo::AddOp>(
                loc, index_vector_type, indices_vector, low_vector).getResult();
        }
        auto indices_matrix_type = mlir::RankedTensorType::get(
            {current_shape[dim], 1}, rewriter.getI32Type());
        mlir::Value indices = rewriter.create<mlir::stablehlo::ReshapeOp>(
            loc, indices_matrix_type, indices_vector).getResult();

        llvm::SmallVector<int64_t> update_window_dims;
        update_window_dims.reserve(rank > 0 ? rank - 1 : 0);
        for (int64_t update_dim = 0; update_dim < rank; ++update_dim) {
            if (update_dim != dim) {
                update_window_dims.push_back(update_dim);
            }
        }
        auto scatter_dims = mlir::stablehlo::ScatterDimensionNumbersAttr::get(
            rewriter.getContext(),
            update_window_dims,
            /*insertedWindowDims=*/{dim},
            /*inputBatchingDims=*/{},
            /*scatterIndicesBatchingDims=*/{},
            /*scatterDimsToOperandDims=*/{dim},
            /*indexVectorDim=*/1);
        llvm::SmallVector<mlir::Type> scatter_result_types{next_type};
        auto scatter = rewriter.create<mlir::stablehlo::ScatterOp>(
            loc,
            scatter_result_types,
            mlir::ValueRange{base},
            indices,
            mlir::ValueRange{current},
            scatter_dims,
            /*indicesAreSorted=*/true,
            /*uniqueIndices=*/true);

        appendOverwriteUpdateComputation(scatter, element_type, loc);

        current = scatter.getResult(0);
        current_shape = next_shape;
    }

    rewriter.replaceOp(pad_op, mlir::ValueRange{current});
    return mlir::success();
}

mlir::LogicalResult lowerSinglePredicateConvertToSelect(
    mlir::stablehlo::ConvertOp convert_op,
    mlir::PatternRewriter& rewriter) {
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        convert_op.getOperand().getType());
    auto output_type = mlir::dyn_cast<mlir::RankedTensorType>(
        convert_op.getResult().getType());
    if (!input_type || !output_type) {
        return convert_op.emitError("predicate convert requires ranked tensor operands");
    }
    if (input_type.getShape() != output_type.getShape()) {
        return convert_op.emitError("predicate convert requires matching input and output shapes");
    }

    rewriter.setInsertionPoint(convert_op);
    auto one = createIntegerSplatConstant(
        rewriter, convert_op.getLoc(), output_type, 1);
    if (!one) {
        return mlir::failure();
    }
    auto zero = createIntegerSplatConstant(
        rewriter, convert_op.getLoc(), output_type, 0);
    if (!zero) {
        return mlir::failure();
    }
    auto select = rewriter.create<mlir::stablehlo::SelectOp>(
        convert_op.getLoc(),
        output_type,
        convert_op.getOperand(),
        *one,
        *zero);
    rewriter.replaceOp(convert_op, mlir::ValueRange{select.getResult()});
    return mlir::success();
}

bool isPredicateConvert(mlir::stablehlo::ConvertOp convert_op) {
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        convert_op.getOperand().getType());
    return input_type && input_type.getElementType().isInteger(1);
}

bool isLastTwoDimSwap(mlir::stablehlo::TransposeOp transpose_op) {
    if (!transpose_op) {
        return false;
    }
    auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(
        transpose_op.getOperand().getType());
    auto output_type = mlir::dyn_cast<mlir::RankedTensorType>(
        transpose_op.getResult().getType());
    if (!input_type || !output_type ||
        !input_type.hasStaticShape() ||
        !output_type.hasStaticShape() ||
        input_type.getRank() < 2 ||
        output_type.getRank() != input_type.getRank()) {
        return false;
    }
    auto permutation = transpose_op.getPermutation();
    int64_t rank = input_type.getRank();
    if (static_cast<int64_t>(permutation.size()) != rank) {
        return false;
    }
    for (int64_t dim = 0; dim < rank - 2; ++dim) {
        if (permutation[dim] != dim) {
            return false;
        }
    }
    return permutation[rank - 2] == rank - 1 &&
           permutation[rank - 1] == rank - 2;
}

std::optional<mlir::stablehlo::TransposeOp> singleUseRhsTranspose(mlir::Value rhs) {
    mlir::Value current = rhs;
    while (auto custom_call_op =
               current.getDefiningOp<mlir::stablehlo::CustomCallOp>()) {
        if (!current.hasOneUse() || !isIdentityCustomCall(custom_call_op)) {
            return std::nullopt;
        }
        current = custom_call_op.getInputs().front();
    }

    auto transpose_op = current.getDefiningOp<mlir::stablehlo::TransposeOp>();
    if (!transpose_op || !current.hasOneUse() || !isLastTwoDimSwap(transpose_op)) {
        return std::nullopt;
    }
    return transpose_op;
}

bool isRhsMatmulTransposeFold(mlir::stablehlo::DotGeneralOp dot_op) {
    return singleUseRhsTranspose(dot_op.getRhs()).has_value();
}

llvm::SmallVector<int64_t> transposeOperandDims(
    mlir::stablehlo::TransposeOp transpose_op,
    llvm::ArrayRef<int64_t> result_dims) {
    auto permutation = transpose_op.getPermutation();
    llvm::SmallVector<int64_t> operand_dims;
    operand_dims.reserve(result_dims.size());
    for (int64_t dim : result_dims) {
        operand_dims.push_back(permutation[dim]);
    }
    return operand_dims;
}

mlir::LogicalResult lowerSingleRhsMatmulTranspose(
    mlir::stablehlo::DotGeneralOp dot_op,
    mlir::PatternRewriter& rewriter) {
    auto transpose_op = singleUseRhsTranspose(dot_op.getRhs());
    if (!transpose_op) {
        return mlir::failure();
    }

    auto dims = dot_op.getDotDimensionNumbers();
    auto rhs_batching_dimensions =
        transposeOperandDims(*transpose_op, dims.getRhsBatchingDimensions());
    auto rhs_contracting_dimensions =
        transposeOperandDims(*transpose_op, dims.getRhsContractingDimensions());
    auto remapped_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        dims.getLhsBatchingDimensions(),
        rhs_batching_dimensions,
        dims.getLhsContractingDimensions(),
        rhs_contracting_dimensions);

    auto replacement = rewriter.create<mlir::stablehlo::DotGeneralOp>(
        dot_op.getLoc(),
        dot_op.getResult().getType(),
        dot_op.getLhs(),
        transpose_op->getOperand(),
        remapped_dims,
        dot_op.getPrecisionConfigAttr(),
        dot_op.getAlgorithmAttr());
    rewriter.replaceOp(dot_op, replacement.getResult());
    return mlir::success();
}

bool isNestedInCaseRegion(mlir::Operation* op) {
    return op->getParentOfType<mlir::stablehlo::CaseOp>() != nullptr;
}

struct CleanupRewriteState {
    bool failed = false;
};

template <typename OpT>
struct CleanupPattern : public mlir::OpRewritePattern<OpT> {
    using Lower = mlir::LogicalResult (*)(OpT, mlir::PatternRewriter&);
    using Match = bool (*)(OpT);

    CleanupPattern(
        mlir::MLIRContext* context,
        CleanupRewriteState& state,
        Lower lower,
        Match match = nullptr)
        : mlir::OpRewritePattern<OpT>(context),
          state(state),
          lower(lower),
          match(match) {}

    mlir::LogicalResult matchAndRewrite(OpT op, mlir::PatternRewriter& rewriter) const override {
        if (state.failed) {
            return mlir::failure();
        }
        if (isNestedInCaseRegion(op.getOperation())) {
            return mlir::failure();
        }
        if (match && !match(op)) {
            return mlir::failure();
        }
        mlir::LogicalResult result = lower(op, rewriter);
        state.failed = mlir::failed(result);
        return result;
    }

    CleanupRewriteState& state;
    Lower lower;
    Match match;
};

mlir::LogicalResult runCleanupRewritePatterns(
    mlir::MLIRContext& context,
    mlir::ModuleOp module) {
    CleanupRewriteState state;
    mlir::RewritePatternSet patterns(&context);
    patterns.add(std::make_unique<CleanupPattern<mlir::stablehlo::ConvertOp>>(
        &context, state, lowerSinglePredicateConvertToSelect, isPredicateConvert));
    patterns.add(std::make_unique<CleanupPattern<mlir::stablehlo::CaseOp>>(
        &context, state, lowerSingleCaseOpToSelects));
    patterns.add(std::make_unique<CleanupPattern<mlir::stablehlo::WhileOp>>(
        &context, state, lowerSingleStaticWhileOp));
    patterns.add(std::make_unique<CleanupPattern<mlir::stablehlo::DynamicUpdateSliceOp>>(
        &context, state, lowerSingleDynamicUpdateSliceToScatter));
    patterns.add(std::make_unique<CleanupPattern<mlir::stablehlo::PadOp>>(
        &context, state, lowerSinglePadToScatter));
    patterns.add(std::make_unique<CleanupPattern<mlir::stablehlo::DotGeneralOp>>(
        &context, state, lowerSingleRhsMatmulTranspose, isRhsMatmulTransposeFold));
    patterns.add<libtt::mlir_frontend::SDPADecodeFusing>(&context);
    mlir::GreedyRewriteConfig config;
    config.enableFolding();
    if (mlir::failed(mlir::applyPatternsGreedily(module, std::move(patterns), config))) {
        return mlir::failure();
    }
    return state.failed ? mlir::failure() : mlir::success();
}

bool runCleanupPasses(mlir::MLIRContext& context, mlir::ModuleOp module, std::string& error) {
    auto addCleanup = [](mlir::PassManager& pm) {
        pm.addPass(mlir::createInlinerPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
    };

    mlir::PassManager pm(&context);
    addCleanup(pm);
    if (mlir::failed(pm.run(module))) {
        error = "failed to run MLIR cleanup passes";
        return false;
    }
    if (mlir::failed(runCleanupRewritePatterns(context, module))) {
        error = "failed to apply MLIR cleanup rewrite patterns; see MLIR diagnostics above";
        return false;
    }

    mlir::PassManager post_case_pm(&context);
    addCleanup(post_case_pm);
    if (mlir::failed(post_case_pm.run(module))) {
        error = "failed to run MLIR post-case cleanup passes";
        return false;
    }
    return true;
}

std::optional<FuncOp> findEntryFunction(mlir::ModuleOp module) {
    std::optional<FuncOp> entry;
    module.walk([&](FuncOp func) {
        if (!entry.has_value() || func.getName() == "main") {
            entry = func;
        }
    });
    return entry;
}

tt::TensorDesc::ElementType mapProtoElementType(mlir::Type element_type) {
    if (element_type.isBF16()) return tt::TensorDesc::ELEMENT_TYPE_BF16;
    if (element_type.isF16()) return tt::TensorDesc::ELEMENT_TYPE_F16;
    if (element_type.isF32()) return tt::TensorDesc::ELEMENT_TYPE_F32;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
        switch (integer.getWidth()) {
            case 1:
                return tt::TensorDesc::ELEMENT_TYPE_PRED;
            case 8:
                return integer.isUnsigned() ? tt::TensorDesc::ELEMENT_TYPE_U8
                                            : tt::TensorDesc::ELEMENT_TYPE_S8;
            case 16:
                return integer.isUnsigned() ? tt::TensorDesc::ELEMENT_TYPE_U16
                                            : tt::TensorDesc::ELEMENT_TYPE_UNKNOWN;
            case 32:
                return integer.isUnsigned() ? tt::TensorDesc::ELEMENT_TYPE_U32
                                            : tt::TensorDesc::ELEMENT_TYPE_S32;
            case 64:
                return integer.isUnsigned() ? tt::TensorDesc::ELEMENT_TYPE_U32
                                            : tt::TensorDesc::ELEMENT_TYPE_S32;
            default:
                return tt::TensorDesc::ELEMENT_TYPE_UNKNOWN;
        }
    }
    return tt::TensorDesc::ELEMENT_TYPE_UNKNOWN;
}

std::optional<uint32_t> elementBitWidth(mlir::Type element_type) {
    if (element_type.isBF16() || element_type.isF16()) return 16;
    if (element_type.isF32()) return 32;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
        return integer.getWidth();
    }
    return std::nullopt;
}

bool fillTensorDesc(mlir::Type type, tt::TensorDesc& tensor_desc, std::string& error) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensor) {
        error = "only ranked tensor values are currently supported";
        return false;
    }
    if (!tensor.hasStaticShape()) {
        error = "dynamic tensor shapes are not currently supported";
        return false;
    }

    auto element_type = mapProtoElementType(tensor.getElementType());
    if (element_type == tt::TensorDesc::ELEMENT_TYPE_UNKNOWN) {
        std::string type_string;
        llvm::raw_string_ostream os(type_string);
        tensor.getElementType().print(os);
        error = "unsupported tensor element type: " + os.str();
        return false;
    }

    tensor_desc.clear_dims();
    for (auto dim : tensor.getShape()) {
        tensor_desc.add_dims(dim);
    }
    tensor_desc.set_element_type(element_type);
    return true;
}

bool addValueDesc(
    mlir::Value value,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error,
    uint32_t& id_out) {
    auto existing = value_ids.find(value);
    if (existing != value_ids.end()) {
        id_out = existing->second;
        return true;
    }

    auto* value_desc = executable.add_values();
    if (!fillTensorDesc(value.getType(), *value_desc->mutable_tensor(), error)) {
        return false;
    }

    uint32_t value_id = executable.values_size() - 1;
    value_ids.try_emplace(value, value_id);
    id_out = value_id;
    return true;
}

bool addSyntheticValueDescLike(
    mlir::Value like,
    tt::TensorDesc::ElementType element_type,
    tt::Executable& executable,
    std::string& error,
    uint32_t& id_out) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(like.getType());
    if (!tensor || !tensor.hasStaticShape()) {
        error = "synthetic values require ranked static tensor shapes";
        return false;
    }

    auto* value_desc = executable.add_values();
    auto* tensor_desc = value_desc->mutable_tensor();
    for (auto dim : tensor.getShape()) {
        tensor_desc->add_dims(dim);
    }
    tensor_desc->set_element_type(element_type);
    id_out = executable.values_size() - 1;
    return true;
}

bool fillProgramSignature(FuncOp func, tt::AnalysisResult& result, std::string& error) {
    auto type = func.getFunctionType();

    result.clear_inputs();
    for (auto input_type : type.getInputs()) {
        auto* input = result.add_inputs();
        if (!fillTensorDesc(input_type, *input, error)) {
            error = "unsupported entry input: " + error;
            return false;
        }
    }

    result.clear_outputs();
    for (auto output_type : type.getResults()) {
        auto* output = result.add_outputs();
        if (!fillTensorDesc(output_type, *output, error)) {
            error = "unsupported entry output: " + error;
            return false;
        }
    }
    return true;
}

std::optional<uint32_t> packIntegerConstant(
    mlir::IntegerType integer_type,
    const llvm::APInt& bits,
    std::string& error) {
    if (mapProtoElementType(integer_type) == tt::TensorDesc::ELEMENT_TYPE_UNKNOWN) {
        error = "unsupported integer constant type";
        return std::nullopt;
    }
    if (integer_type.getWidth() <= 32) {
        return static_cast<uint32_t>(bits.getZExtValue());
    }

    // The executable type system maps 64-bit StableHLO integers onto 32-bit
    // runtime integer tensors, so only accept values that round-trip exactly.
    if (integer_type.isUnsigned()) {
        uint64_t value = bits.getZExtValue();
        if (value <= std::numeric_limits<uint32_t>::max()) {
            return static_cast<uint32_t>(value);
        }
        error = "unsigned 64-bit integer constant does not fit in the 32-bit executable type";
        return std::nullopt;
    }

    int64_t value = bits.getSExtValue();
    if (value >= std::numeric_limits<int32_t>::min() &&
        value <= std::numeric_limits<int32_t>::max()) {
        return static_cast<uint32_t>(static_cast<int32_t>(value));
    }
    error = "signed 64-bit integer constant does not fit in the 32-bit executable type";
    return std::nullopt;
}

std::optional<uint32_t> packedConstantValue(mlir::Value value, std::string& error) {
    while (auto broadcast_op = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        value = broadcast_op.getOperand();
    }

    auto constant_op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!constant_op) {
        error = "only broadcast_in_dim of constants is currently supported";
        return std::nullopt;
    }

    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constant_op.getValue());
    if (!dense || !dense.isSplat()) {
        error = "only splat constants are currently supported";
        return std::nullopt;
    }

    auto element_type = dense.getElementType();
    if (element_type.isBF16() || element_type.isF16()) {
        auto bits = dense.getSplatValue<llvm::APFloat>().bitcastToAPInt();
        uint32_t value16 = bits.extractBitsAsZExtValue(16, 0);
        return value16 | (value16 << 16);
    }
    if (element_type.isF32()) {
        auto bits = dense.getSplatValue<llvm::APFloat>().bitcastToAPInt();
        return bits.extractBitsAsZExtValue(32, 0);
    }
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
        return packIntegerConstant(integer, dense.getSplatValue<llvm::APInt>(), error);
    }
    error = "only bf16/f16/f32 and <=64-bit integer splat constants are currently supported";
    return std::nullopt;
}

void appendLittleEndian(std::vector<uint8_t>& data, uint64_t value, unsigned byte_count) {
    for (unsigned index = 0; index < byte_count; ++index) {
        data.push_back(static_cast<uint8_t>((value >> (index * 8)) & 0xff));
    }
}

std::optional<std::vector<uint8_t>> denseConstantData(
    mlir::Value value,
    std::string& error) {
    auto constant_op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!constant_op) {
        error = "dense constants require a stablehlo.constant";
        return std::nullopt;
    }
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constant_op.getValue());
    if (!dense) {
        error = "only dense constants are currently supported";
        return std::nullopt;
    }
    auto element_type = dense.getElementType();
    std::vector<uint8_t> data;
    data.reserve(dense.getNumElements() * 4);
    if (element_type.isBF16() || element_type.isF16()) {
        for (const llvm::APFloat& value : dense.getValues<llvm::APFloat>()) {
            auto bits = value.bitcastToAPInt();
            appendLittleEndian(data, bits.extractBitsAsZExtValue(16, 0), 2);
        }
        return data;
    }
    if (element_type.isF32()) {
        for (const llvm::APFloat& value : dense.getValues<llvm::APFloat>()) {
            auto bits = value.bitcastToAPInt();
            appendLittleEndian(data, bits.extractBitsAsZExtValue(32, 0), 4);
        }
        return data;
    }
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
        if (mapProtoElementType(integer) == tt::TensorDesc::ELEMENT_TYPE_UNKNOWN) {
            error = "unsupported integer dense constant type";
            return std::nullopt;
        }
        unsigned byte_count = std::min<unsigned>(
            (integer.getWidth() + 7) / 8,
            sizeof(uint32_t));
        for (const llvm::APInt& value : dense.getValues<llvm::APInt>()) {
            auto packed = packIntegerConstant(integer, value, error);
            if (!packed) return std::nullopt;
            appendLittleEndian(data, *packed, byte_count);
        }
        return data;
    }

    error = "only bf16/f16/f32 and <=64-bit integer dense constants are currently supported";
    return std::nullopt;
}

void addConstantOp(tt::Executable& executable, uint32_t output_id, uint32_t packed_value) {
    auto* constant = executable.add_ops();
    constant->set_output_id(output_id);
    constant->mutable_constant()->set_packed_value(packed_value);
}

void addConstantDataOp(
    tt::Executable& executable,
    uint32_t output_id,
    const std::vector<uint8_t>& data) {
    auto* constant = executable.add_ops();
    constant->set_output_id(output_id);
    constant->mutable_constant()->set_data(data.data(), data.size());
}

bool addConstantValueOp(
    mlir::Value value,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error) {
    uint32_t output_id = 0;
    if (!addValueDesc(value, executable, value_ids, error, output_id)) {
        return false;
    }

    std::string packed_error;
    if (auto packed_value = packedConstantValue(value, packed_error)) {
        addConstantOp(executable, output_id, *packed_value);
        return true;
    }

    if (auto data = denseConstantData(value, error)) {
        addConstantDataOp(executable, output_id, *data);
        return true;
    }
    if (error.empty()) {
        error = packed_error;
    }
    return false;
}

tt::FusedElementwiseOp::Node::CompareDirection mapCompareDirection(
    mlir::stablehlo::ComparisonDirection direction) {
    switch (direction) {
        case mlir::stablehlo::ComparisonDirection::EQ:
            return tt::FusedElementwiseOp::Node::DIRECTION_EQ;
        case mlir::stablehlo::ComparisonDirection::NE:
            return tt::FusedElementwiseOp::Node::DIRECTION_NE;
        case mlir::stablehlo::ComparisonDirection::GE:
            return tt::FusedElementwiseOp::Node::DIRECTION_GE;
        case mlir::stablehlo::ComparisonDirection::GT:
            return tt::FusedElementwiseOp::Node::DIRECTION_GT;
        case mlir::stablehlo::ComparisonDirection::LE:
            return tt::FusedElementwiseOp::Node::DIRECTION_LE;
        case mlir::stablehlo::ComparisonDirection::LT:
            return tt::FusedElementwiseOp::Node::DIRECTION_LT;
    }
    return tt::FusedElementwiseOp::Node::DIRECTION_EQ;
}

std::optional<tt::ReduceOp::Reducer> mapReduceReducer(
    mlir::stablehlo::ReduceOp reduce_op,
    std::string& error) {
    if (reduce_op.getInputs().size() != 1 ||
        reduce_op.getInitValues().size() != 1 ||
        reduce_op->getNumResults() != 1) {
        std::string op_text;
        llvm::raw_string_ostream os(op_text);
        reduce_op.print(os);
        error = "only single-input single-result reduce ops are currently supported; got: " +
                os.str();
        return std::nullopt;
    }

    auto& body = reduce_op.getBody();
    if (body.empty() || body.getBlocks().size() != 1) {
        error = "reduce bodies must contain exactly one block";
        return std::nullopt;
    }

    mlir::Operation* reducer_op = nullptr;
    mlir::Operation* return_operation = nullptr;
    for (mlir::Operation& body_op : body.front()) {
        if (mlir::isa<mlir::stablehlo::ReturnOp>(body_op)) {
            return_operation = &body_op;
            continue;
        }
        if (reducer_op) {
            error = "only single-op reduce bodies are currently supported";
            return std::nullopt;
        }
        reducer_op = &body_op;
    }

    if (!reducer_op || !return_operation) {
        error = "reduce body must contain a reducer op and stablehlo.return";
        return std::nullopt;
    }

    auto return_op = mlir::cast<mlir::stablehlo::ReturnOp>(return_operation);
    if (return_op.getNumOperands() != 1 ||
        return_op.getOperand(0).getDefiningOp() != reducer_op) {
        error = "reduce body must return the reducer op result";
        return std::nullopt;
    }

    if (mlir::isa<mlir::stablehlo::AddOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_ADD;
    }
    if (mlir::isa<mlir::stablehlo::MaxOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_MAX;
    }
    if (mlir::isa<mlir::stablehlo::MinOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_MIN;
    }
    if (mlir::isa<mlir::stablehlo::MulOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_MUL;
    }
    if (mlir::isa<mlir::stablehlo::AndOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_AND;
    }
    if (mlir::isa<mlir::stablehlo::OrOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_OR;
    }

    error = "unsupported reduce reducer: " + reducer_op->getName().getStringRef().str();
    return std::nullopt;
}

std::optional<tt::ReduceOp::Reducer> mapReduceWindowReducer(
    mlir::stablehlo::ReduceWindowOp reduce_window_op,
    std::string& error) {
    if (reduce_window_op.getInputs().size() != 1 ||
        reduce_window_op.getInitValues().size() != 1 ||
        reduce_window_op->getNumResults() != 1) {
        error = "only single-input single-result reduce_window ops are currently supported";
        return std::nullopt;
    }

    auto& body = reduce_window_op.getBody();
    if (body.empty() || body.getBlocks().size() != 1) {
        error = "reduce_window bodies must contain exactly one block";
        return std::nullopt;
    }

    mlir::Operation* reducer_op = nullptr;
    mlir::Operation* return_operation = nullptr;
    for (mlir::Operation& body_op : body.front()) {
        if (mlir::isa<mlir::stablehlo::ReturnOp>(body_op)) {
            return_operation = &body_op;
            continue;
        }
        if (reducer_op) {
            error = "only single-op reduce_window bodies are currently supported";
            return std::nullopt;
        }
        reducer_op = &body_op;
    }

    if (!reducer_op || !return_operation) {
        error = "reduce_window body must contain a reducer op and stablehlo.return";
        return std::nullopt;
    }

    auto return_op = mlir::cast<mlir::stablehlo::ReturnOp>(return_operation);
    if (return_op.getNumOperands() != 1 ||
        return_op.getOperand(0).getDefiningOp() != reducer_op) {
        error = "reduce_window body must return the reducer op result";
        return std::nullopt;
    }

    if (mlir::isa<mlir::stablehlo::AddOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_ADD;
    }
    if (mlir::isa<mlir::stablehlo::MaxOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_MAX;
    }
    if (mlir::isa<mlir::stablehlo::MinOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_MIN;
    }
    if (mlir::isa<mlir::stablehlo::MulOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_MUL;
    }
    if (mlir::isa<mlir::stablehlo::AndOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_AND;
    }
    if (mlir::isa<mlir::stablehlo::OrOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_OR;
    }

    error = "unsupported reduce_window reducer: " + reducer_op->getName().getStringRef().str();
    return std::nullopt;
}

std::vector<int64_t> optionalArrayOrOnes(
    std::optional<llvm::ArrayRef<int64_t>> values,
    size_t rank) {
    if (values) {
        return values->vec();
    }
    return std::vector<int64_t>(rank, 1);
}

bool reduceWindowPaddingVectors(
    std::optional<mlir::DenseIntElementsAttr> padding,
    size_t rank,
    std::vector<int64_t>& low,
    std::vector<int64_t>& high,
    std::string& error) {
    low.assign(rank, 0);
    high.assign(rank, 0);
    if (!padding) {
        return true;
    }
    if (padding->getNumElements() != static_cast<int64_t>(rank * 2)) {
        error = "reduce_window padding must have shape rank x 2";
        return false;
    }

    size_t index = 0;
    for (const llvm::APInt& value : padding->getValues<llvm::APInt>()) {
        if (index % 2 == 0) {
            low[index / 2] = value.getSExtValue();
        } else {
            high[index / 2] = value.getSExtValue();
        }
        ++index;
    }
    return true;
}

bool isSetScatter(mlir::stablehlo::ScatterOp scatter_op, std::string& error) {
    if (scatter_op.getInputs().size() != 1 || scatter_op.getUpdates().size() != 1 ||
        scatter_op->getNumResults() != 1) {
        error = "scatter currently requires one operand, one update, and one result";
        return false;
    }

    mlir::Region& body = scatter_op.getUpdateComputation();
    if (!body.hasOneBlock()) {
        error = "scatter update body must contain one block";
        return false;
    }
    mlir::Block& block = body.front();
    if (block.getNumArguments() != 2) {
        error = "scatter set update body must take old and new scalar arguments";
        return false;
    }

    mlir::Operation* return_operation = nullptr;
    for (mlir::Operation& body_op : block) {
        if (mlir::isa<mlir::stablehlo::ReturnOp>(body_op)) {
            return_operation = &body_op;
            continue;
        }
        error = "scatter currently only supports set update bodies";
        return false;
    }
    if (!return_operation) {
        error = "scatter update body must contain stablehlo.return";
        return false;
    }

    auto return_op = mlir::cast<mlir::stablehlo::ReturnOp>(return_operation);
    if (return_op.getNumOperands() != 1 || return_op.getOperand(0) != block.getArgument(1)) {
        error = "scatter currently only supports update bodies that return the new value";
        return false;
    }
    return true;
}

bool addBitwiseBinaryOp(
    mlir::Value lhs,
    mlir::Value rhs,
    mlir::Value result,
    tt::BitwiseBinaryOp::Kind kind,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error) {
    uint32_t lhs_id = 0;
    uint32_t rhs_id = 0;
    uint32_t output_id = 0;
    if (!addValueDesc(lhs, executable, value_ids, error, lhs_id) ||
        !addValueDesc(rhs, executable, value_ids, error, rhs_id) ||
        !addValueDesc(result, executable, value_ids, error, output_id)) {
        return false;
    }

    auto* bitwise = executable.add_ops();
    bitwise->set_output_id(output_id);
    bitwise->mutable_bitwise_binary()->set_lhs_id(lhs_id);
    bitwise->mutable_bitwise_binary()->set_rhs_id(rhs_id);
    bitwise->mutable_bitwise_binary()->set_kind(kind);
    return true;
}

std::optional<tt::BitwiseBinaryOp::Kind> bitwiseBinaryKind(mlir::Operation* op) {
    using Kind = tt::BitwiseBinaryOp::Kind;
    return llvm::TypeSwitch<mlir::Operation*, std::optional<Kind>>(op)
        .Case<mlir::stablehlo::AndOp>([](auto) { return tt::BitwiseBinaryOp::KIND_AND; })
        .Case<mlir::stablehlo::OrOp>([](auto) { return tt::BitwiseBinaryOp::KIND_OR; })
        .Case<mlir::stablehlo::XorOp>([](auto) { return tt::BitwiseBinaryOp::KIND_XOR; })
        .Case<mlir::stablehlo::ShiftLeftOp>(
            [](auto) { return tt::BitwiseBinaryOp::KIND_SHIFT_LEFT; })
        .Case<mlir::stablehlo::ShiftRightLogicalOp>(
            [](auto) { return tt::BitwiseBinaryOp::KIND_SHIFT_RIGHT_LOGICAL; })
        .Case<mlir::stablehlo::ShiftRightArithmeticOp>(
            [](auto) { return tt::BitwiseBinaryOp::KIND_SHIFT_RIGHT_ARITHMETIC; })
        .Default([](auto) { return std::nullopt; });
}

std::optional<uint32_t> sdpaDecodeScaleBf16Packed(
    mlir::stablehlo::CustomCallOp custom_call_op,
    std::string& error) {
    auto backend_config = custom_call_op.getBackendConfig();
    if (!backend_config) {
        error = "tt.sdpa_decode custom_call requires backend_config";
        return std::nullopt;
    }
    if (auto config = mlir::dyn_cast<mlir::StringAttr>(*backend_config)) {
        uint64_t value = 0;
        if (config.getValue().getAsInteger(10, value)) {
            error = "tt.sdpa_decode string backend_config must be a scale_bf16_packed integer";
            return std::nullopt;
        }
        if (value > std::numeric_limits<uint32_t>::max()) {
            error = "tt.sdpa_decode scale_bf16_packed is out of range";
            return std::nullopt;
        }
        return static_cast<uint32_t>(value);
    }
    auto config = mlir::dyn_cast<mlir::DictionaryAttr>(*backend_config);
    if (!config) {
        error = "tt.sdpa_decode custom_call requires string or dictionary backend_config";
        return std::nullopt;
    }
    auto scale = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
        config.get("scale_bf16_packed"));
    if (!scale) {
        error = "tt.sdpa_decode custom_call requires integer scale_bf16_packed";
        return std::nullopt;
    }
    uint64_t value = scale.getValue().getZExtValue();
    if (value > std::numeric_limits<uint32_t>::max()) {
        error = "tt.sdpa_decode scale_bf16_packed is out of range";
        return std::nullopt;
    }
    return static_cast<uint32_t>(value);
}

bool addSdpaDecodeOp(
    mlir::stablehlo::CustomCallOp custom_call_op,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error) {
    if (custom_call_op->getNumResults() != 1 || custom_call_op.getInputs().size() != 5) {
        error = "tt.sdpa_decode custom_call must have five inputs and one result";
        return false;
    }
    auto scale_bf16_packed = sdpaDecodeScaleBf16Packed(custom_call_op, error);
    if (!scale_bf16_packed) {
        return false;
    }

    uint32_t input_ids[5] = {};
    for (auto [index, input] : llvm::enumerate(custom_call_op.getInputs())) {
        if (!addValueDesc(input, executable, value_ids, error, input_ids[index])) {
            return false;
        }
    }
    uint32_t output_id = 0;
    if (!addValueDesc(custom_call_op->getResult(0), executable, value_ids, error, output_id)) {
        return false;
    }

    auto* op = executable.add_ops();
    op->set_output_id(output_id);
    auto* sdpa = op->mutable_sdpa_decode();
    sdpa->set_q_id(input_ids[0]);
    sdpa->set_k_id(input_ids[1]);
    sdpa->set_v_id(input_ids[2]);
    sdpa->set_seq_lens_id(input_ids[3]);
    sdpa->set_loc_id(input_ids[4]);
    sdpa->set_scale_bf16_packed(*scale_bf16_packed);
    return true;
}

bool addTopKOp(
    mlir::Value operand,
    mlir::Value values,
    mlir::Value indices,
    int64_t k,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error) {
    if (k < 0 || k > std::numeric_limits<uint32_t>::max()) {
        error = "top_k k is out of range";
        return false;
    }

    uint32_t input_id = 0;
    uint32_t values_id = 0;
    uint32_t indices_id = 0;
    if (!addValueDesc(operand, executable, value_ids, error, input_id) ||
        !addValueDesc(values, executable, value_ids, error, values_id) ||
        !addValueDesc(indices, executable, value_ids, error, indices_id)) {
        return false;
    }

    auto* top_k = executable.add_ops();
    top_k->set_output_id(values_id);
    top_k->mutable_top_k()->set_operand_id(input_id);
    top_k->mutable_top_k()->set_indices_id(indices_id);
    top_k->mutable_top_k()->set_k(static_cast<uint32_t>(k));
    return true;
}

bool isArgmaxReduceOp(mlir::stablehlo::ReduceOp reduce_op) {
    if (reduce_op.getInputs().size() != 2 ||
        reduce_op.getInitValues().size() != 2 ||
        reduce_op->getNumResults() != 2 ||
        reduce_op.getDimensions().size() != 1) {
        return false;
    }

    int64_t reduce_dim = reduce_op.getDimensions()[0];
    auto inputs = reduce_op.getInputs();
    auto input_it = inputs.begin();
    mlir::Value values_input = *input_it++;
    mlir::Value indices_input = *input_it;
    auto static_tensor_type = [](mlir::Value value) -> std::optional<mlir::RankedTensorType> {
        auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
        if (!tensor || !tensor.hasStaticShape()) {
            return std::nullopt;
        }
        return tensor;
    };
    auto input_type = static_tensor_type(values_input);
    auto index_type = static_tensor_type(indices_input);
    auto values_type = static_tensor_type(reduce_op->getResult(0));
    auto indices_type = static_tensor_type(reduce_op->getResult(1));
    auto input_element_type = input_type
        ? mapProtoElementType(input_type->getElementType())
        : tt::TensorDesc::ELEMENT_TYPE_UNKNOWN;
    bool is_float_input =
        input_element_type == tt::TensorDesc::ELEMENT_TYPE_BF16 ||
        input_element_type == tt::TensorDesc::ELEMENT_TYPE_F16 ||
        input_element_type == tt::TensorDesc::ELEMENT_TYPE_F32;
    if (!input_type || !index_type || !values_type || !indices_type ||
        input_type->getRank() == 0 ||
        reduce_dim != input_type->getRank() - 1 ||
        index_type->getShape() != input_type->getShape() ||
        values_type->getShape() != indices_type->getShape() ||
        !is_float_input ||
        mapProtoElementType(index_type->getElementType()) != tt::TensorDesc::ELEMENT_TYPE_S32 ||
        mapProtoElementType(indices_type->getElementType()) != tt::TensorDesc::ELEMENT_TYPE_S32 ||
        mapProtoElementType(values_type->getElementType()) != input_element_type) {
        return false;
    }
    if (input_type->getRank() == 2 && input_type->getShape()[0] != 1) {
        return false;
    }
    if (input_type->getRank() > 2) {
        return false;
    }

    return true;
}

constexpr unsigned kMaxFusedElementwiseNodes = 16;
constexpr unsigned kMaxFusedElementwiseInputs = 8;

// Fused elementwise lowering is intentionally conservative: it only folds
// single-use StableHLO elementwise producers into an elementwise root, and it
// keeps reductions, layout-changing ops, matmuls, and custom calls as
// boundaries. Non-constant broadcasts are only folded for logical scalar
// tensors; the runtime reader expands the first element to a full tile.
struct FusedElementwiseNodeDesc {
    tt::FusedElementwiseOp::Node::Kind kind;
    llvm::SmallVector<uint32_t> input_nodes;
    uint32_t input_index = 0;
    uint32_t packed_value = 0;
    tt::TensorDesc::ElementType element_type =
        tt::TensorDesc::ELEMENT_TYPE_UNKNOWN;
    bool single_tile_broadcast = false;
    tt::FusedElementwiseOp::Node::CompareDirection compare_direction =
        tt::FusedElementwiseOp::Node::DIRECTION_EQ;
};

// Temporary collector state for one fused subgraph. `inputs` are external MLIR
// values read by the runtime kernel, `nodes` are the internal fused DAG in
// dependency order, and `covered_ops` are skipped by the main lowering loop.
struct FusedElementwiseRegion {
    llvm::SmallVector<mlir::Value> inputs;
    llvm::SmallVector<FusedElementwiseNodeDesc> nodes;
    llvm::SmallVector<mlir::Operation*> covered_ops;
};

std::optional<mlir::RankedTensorType> getStaticTensorType(mlir::Value value) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!tensor || !tensor.hasStaticShape()) {
        return std::nullopt;
    }
    return tensor;
}

std::optional<tt::TensorDesc::ElementType> staticValueElementType(mlir::Value value) {
    if (auto tensor = getStaticTensorType(value)) {
        auto element_type = mapProtoElementType(tensor->getElementType());
        if (element_type != tt::TensorDesc::ELEMENT_TYPE_UNKNOWN) {
            return element_type;
        }
    }
    return std::nullopt;
}

bool isFusedFloatElementType(tt::TensorDesc::ElementType element_type) {
    return element_type == tt::TensorDesc::ELEMENT_TYPE_BF16 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_F16 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_F32;
}

bool supportsFusedValueElementType(tt::TensorDesc::ElementType element_type) {
    return isFusedFloatElementType(element_type) ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_U32 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_S32 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_U16;
}

bool supportsFusedConvertElementType(tt::TensorDesc::ElementType element_type) {
    return supportsFusedValueElementType(element_type) ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_PRED ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_U8;
}

bool supportsCompareElementType(tt::TensorDesc::ElementType element_type) {
    return isFusedFloatElementType(element_type) ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_S32;
}

bool supportsFusedElementwiseDTypes(
    tt::FusedElementwiseOp::Node::Kind kind,
    tt::TensorDesc::ElementType input_type,
    tt::TensorDesc::ElementType output_type) {
    using Node = tt::FusedElementwiseOp::Node;
    if (kind == Node::KIND_CONVERT) {
        return supportsFusedConvertElementType(input_type) &&
               supportsFusedConvertElementType(output_type);
    }
    if (kind == Node::KIND_COMPARE) {
        return supportsCompareElementType(input_type) &&
               output_type == tt::TensorDesc::ELEMENT_TYPE_PRED;
    }
    if (input_type != output_type) {
        return false;
    }
    switch (kind) {
        case Node::KIND_ADD:
        case Node::KIND_MULTIPLY:
            return supportsFusedValueElementType(input_type);
        case Node::KIND_SUBTRACT:
            return isFusedFloatElementType(input_type) ||
                   input_type == tt::TensorDesc::ELEMENT_TYPE_S32;
        case Node::KIND_DIVIDE:
        case Node::KIND_POWER:
        case Node::KIND_MAX:
        case Node::KIND_COSINE:
        case Node::KIND_SINE:
        case Node::KIND_NEGATE:
        case Node::KIND_EXPONENTIAL:
        case Node::KIND_RSQRT:
        case Node::KIND_LOG:
            return isFusedFloatElementType(input_type);
        default:
            return false;
    }
}

bool supportsFusedValueElement(mlir::Value value) {
    auto element_type = staticValueElementType(value);
    return element_type && supportsFusedConvertElementType(*element_type);
}

bool sameTensorShape(mlir::Value lhs, mlir::Value rhs) {
    auto lhs_tensor = getStaticTensorType(lhs);
    auto rhs_tensor = getStaticTensorType(rhs);
    return lhs_tensor && rhs_tensor && lhs_tensor->getShape() == rhs_tensor->getShape();
}

tt::TensorDesc::ElementType valueElementType(mlir::Value value) {
    auto tensor = mlir::cast<mlir::RankedTensorType>(value.getType());
    return mapProtoElementType(tensor.getElementType());
}

bool isLogicalScalarTile(mlir::Value value) {
    auto tensor = getStaticTensorType(value);
    if (!tensor) {
        return false;
    }
    return llvm::all_of(tensor->getShape(), [](int64_t dim) { return dim == 1; });
}

std::optional<tt::FusedElementwiseOp::Node::Kind> fusedElementwiseKind(
    mlir::Operation* op) {
    using Kind = tt::FusedElementwiseOp::Node::Kind;
    using Node = tt::FusedElementwiseOp::Node;
    return llvm::TypeSwitch<mlir::Operation*, std::optional<Kind>>(op)
        .Case<mlir::stablehlo::AddOp>([](auto) { return Node::KIND_ADD; })
        .Case<mlir::stablehlo::SubtractOp>([](auto) { return Node::KIND_SUBTRACT; })
        .Case<mlir::stablehlo::MulOp>([](auto) { return Node::KIND_MULTIPLY; })
        .Case<mlir::stablehlo::DivOp>([](auto) { return Node::KIND_DIVIDE; })
        .Case<mlir::stablehlo::PowOp>([](auto) { return Node::KIND_POWER; })
        .Case<mlir::stablehlo::MaxOp>([](auto) { return Node::KIND_MAX; })
        .Case<mlir::stablehlo::CompareOp>([](auto) { return Node::KIND_COMPARE; })
        .Case<mlir::stablehlo::CosineOp>([](auto) { return Node::KIND_COSINE; })
        .Case<mlir::stablehlo::SineOp>([](auto) { return Node::KIND_SINE; })
        .Case<mlir::stablehlo::NegOp>([](auto) { return Node::KIND_NEGATE; })
        .Case<mlir::stablehlo::ExpOp>([](auto) { return Node::KIND_EXPONENTIAL; })
        .Case<mlir::stablehlo::RsqrtOp>([](auto) { return Node::KIND_RSQRT; })
        .Case<mlir::stablehlo::LogOp>([](auto) { return Node::KIND_LOG; })
        .Case<mlir::stablehlo::ConvertOp>([](auto) { return Node::KIND_CONVERT; })
        .Default([](auto) { return std::nullopt; });
}

std::optional<tt::FusedElementwiseOp::Node::Kind> supportedFusedElementwiseKind(
    mlir::Operation* op) {
    if (!op || op->getNumResults() != 1) {
        return std::nullopt;
    }
    auto kind = fusedElementwiseKind(op);
    if (!kind) {
        return std::nullopt;
    }
    mlir::Value result = op->getResult(0);
    auto result_element = staticValueElementType(result);
    if (!result_element) {
        return std::nullopt;
    }

    std::optional<tt::TensorDesc::ElementType> input_element;
    for (mlir::Value operand : op->getOperands()) {
        auto operand_element = staticValueElementType(operand);
        if (!operand_element) {
            return std::nullopt;
        }
        if (input_element && *input_element != *operand_element) {
            return std::nullopt;
        }
        input_element = *operand_element;
        std::string ignored;
        if (packedConstantValue(operand, ignored).has_value()) {
            continue;
        }
        if (!sameTensorShape(operand, result)) {
            return std::nullopt;
        }
    }
    if (!input_element ||
        !supportsFusedElementwiseDTypes(*kind, *input_element, *result_element)) {
        return std::nullopt;
    }
    return kind;
}

uint32_t addFusedNode(
    FusedElementwiseRegion& region,
    FusedElementwiseNodeDesc node) {
    uint32_t node_id = static_cast<uint32_t>(region.nodes.size());
    region.nodes.push_back(std::move(node));
    return node_id;
}

uint32_t fusedElementwiseInputIndex(
    FusedElementwiseRegion& region,
    mlir::Value input) {
    auto input_it = std::find(region.inputs.begin(), region.inputs.end(), input);
    if (input_it != region.inputs.end()) {
        return static_cast<uint32_t>(input_it - region.inputs.begin());
    }
    uint32_t input_index = static_cast<uint32_t>(region.inputs.size());
    region.inputs.push_back(input);
    return input_index;
}

void addCoveredConstantProducers(
    FusedElementwiseRegion& region,
    mlir::Value value) {
    while (auto broadcast_op = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        if (!broadcast_op->getResult(0).hasOneUse()) {
            return;
        }
        region.covered_ops.push_back(broadcast_op);
        value = broadcast_op.getOperand();
    }
    if (auto constant_op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
        constant_op && constant_op->getResult(0).hasOneUse()) {
        region.covered_ops.push_back(constant_op);
    }
}

std::optional<uint32_t> collectFusedElementwiseValue(
    mlir::Value value,
    mlir::Value root_value,
    FusedElementwiseRegion& region,
    llvm::DenseMap<mlir::Value, uint32_t>& node_ids);

std::optional<uint32_t> collectFusedElementwiseOp(
    mlir::Operation* op,
    mlir::Value root_value,
    bool is_root,
    FusedElementwiseRegion& region,
    llvm::DenseMap<mlir::Value, uint32_t>& node_ids) {
    if (!op || op->getNumResults() != 1) {
        return std::nullopt;
    }
    auto kind = supportedFusedElementwiseKind(op);
    if (!kind) {
        return std::nullopt;
    }
    mlir::Value result = op->getResult(0);
    if (!is_root && !result.hasOneUse()) {
        return std::nullopt;
    }
    if (!sameTensorShape(result, root_value)) {
        return std::nullopt;
    }

    FusedElementwiseNodeDesc node;
    node.kind = *kind;
    node.element_type = valueElementType(result);
    if (auto compare_op = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
        node.compare_direction = mapCompareDirection(compare_op.getComparisonDirection());
    }

    FusedElementwiseRegion candidate_region = region;
    llvm::DenseMap<mlir::Value, uint32_t> candidate_node_ids = node_ids;
    for (mlir::Value operand : op->getOperands()) {
        auto node_id = collectFusedElementwiseValue(
            operand,
            root_value,
            candidate_region,
            candidate_node_ids);
        if (!node_id.has_value()) {
            return std::nullopt;
        }
        node.input_nodes.push_back(*node_id);
    }

    uint32_t node_id = addFusedNode(candidate_region, std::move(node));
    candidate_node_ids.try_emplace(result, node_id);
    candidate_region.covered_ops.push_back(op);
    region = std::move(candidate_region);
    node_ids = std::move(candidate_node_ids);
    return node_id;
}

std::optional<uint32_t> collectFusedElementwiseValue(
    mlir::Value value,
    mlir::Value root_value,
    FusedElementwiseRegion& region,
    llvm::DenseMap<mlir::Value, uint32_t>& node_ids) {
    auto existing = node_ids.find(value);
    if (existing != node_ids.end()) {
        return existing->second;
    }

    std::string ignored;
    if (auto packed = packedConstantValue(value, ignored)) {
        if (!supportsFusedValueElement(value)) {
            return std::nullopt;
        }
        FusedElementwiseNodeDesc node;
        node.kind = tt::FusedElementwiseOp::Node::KIND_CONSTANT;
        node.packed_value = *packed;
        node.element_type = valueElementType(value);
        uint32_t node_id = addFusedNode(region, std::move(node));
        node_ids.try_emplace(value, node_id);
        addCoveredConstantProducers(region, value);
        return node_id;
    }

    if (auto* defining_op = value.getDefiningOp()) {
        if (auto node_id = collectFusedElementwiseOp(
                defining_op, root_value, false, region, node_ids)) {
            return node_id;
        }
    }

    if (auto broadcast_op = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        mlir::Value operand = broadcast_op.getOperand();
        auto operand_element = staticValueElementType(operand);
        if (broadcast_op->getResult(0).hasOneUse() &&
            operand_element && isFusedFloatElementType(*operand_element) &&
            mlir::isa<mlir::BlockArgument>(operand) &&
            sameTensorShape(value, root_value) &&
            valueElementType(operand) == valueElementType(value) &&
            isLogicalScalarTile(operand)) {
            FusedElementwiseNodeDesc node;
            node.kind = tt::FusedElementwiseOp::Node::KIND_INPUT;
            node.input_index = fusedElementwiseInputIndex(region, operand);
            node.element_type = valueElementType(value);
            node.single_tile_broadcast = true;
            uint32_t node_id = addFusedNode(region, std::move(node));
            node_ids.try_emplace(value, node_id);
            region.covered_ops.push_back(broadcast_op);
            return node_id;
        }
    }

    if (!supportsFusedValueElement(value) || !sameTensorShape(value, root_value)) {
        return std::nullopt;
    }

    FusedElementwiseNodeDesc node;
    node.kind = tt::FusedElementwiseOp::Node::KIND_INPUT;
    node.input_index = fusedElementwiseInputIndex(region, value);
    node.element_type = valueElementType(value);
    uint32_t node_id = addFusedNode(region, std::move(node));
    node_ids.try_emplace(value, node_id);
    return node_id;
}

std::optional<FusedElementwiseRegion> collectFusedElementwiseRegion(
    mlir::Operation* root) {
    FusedElementwiseRegion region;
    llvm::DenseMap<mlir::Value, uint32_t> node_ids;
    auto collected_root = collectFusedElementwiseOp(
        root, root->getResult(0), true, region, node_ids);
    if (!collected_root.has_value() ||
        region.nodes.size() > kMaxFusedElementwiseNodes ||
        region.inputs.size() > kMaxFusedElementwiseInputs) {
        return std::nullopt;
    }
    return region;
}

struct FusedElementwisePlan {
    llvm::DenseMap<mlir::Operation*, FusedElementwiseRegion> roots;
    llvm::DenseSet<mlir::Operation*> covered_ops;
};

FusedElementwisePlan buildFusedElementwisePlan(FuncOp func) {
    FusedElementwisePlan plan;

    for (mlir::Operation& op : llvm::reverse(func.front().without_terminator())) {
        if (plan.covered_ops.contains(&op)) {
            continue;
        }
        auto region = collectFusedElementwiseRegion(&op);
        if (!region.has_value()) {
            continue;
        }
        for (mlir::Operation* covered : region->covered_ops) {
            plan.covered_ops.insert(covered);
        }
        plan.roots.try_emplace(&op, std::move(*region));
    }
    return plan;
}

bool addFusedElementwiseOp(
    mlir::Operation* root,
    const FusedElementwiseRegion& region,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error) {
    uint32_t output_id = 0;
    if (!addValueDesc(root->getResult(0), executable, value_ids, error, output_id)) {
        return false;
    }

    auto* fused = executable.add_ops();
    fused->set_output_id(output_id);
    auto* fused_op = fused->mutable_fused_elementwise();

    for (mlir::Value input : region.inputs) {
        uint32_t input_id = 0;
        if (!addValueDesc(input, executable, value_ids, error, input_id)) {
            return false;
        }
        fused_op->add_input_ids(input_id);
    }

    for (const FusedElementwiseNodeDesc& node : region.nodes) {
        auto* proto_node = fused_op->add_nodes();
        proto_node->set_kind(node.kind);
        for (uint32_t input_node : node.input_nodes) {
            proto_node->add_input_nodes(input_node);
        }
        proto_node->set_input_index(node.input_index);
        proto_node->set_packed_value(node.packed_value);
        proto_node->set_element_type(node.element_type);
        proto_node->set_single_tile_broadcast(node.single_tile_broadcast);
        proto_node->set_compare_direction(node.compare_direction);
    }
    return true;
}

mlir::Operation* singleUser(mlir::Value value) {
    if (!value.hasOneUse()) {
        return nullptr;
    }
    return *value.user_begin();
}

bool isBf16Tensor(mlir::Value value) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    return tensor && tensor.hasStaticShape() && tensor.getElementType().isBF16();
}

bool isS32Tensor(mlir::Value value) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!tensor || !tensor.hasStaticShape()) {
        return false;
    }
    auto integer = mlir::dyn_cast<mlir::IntegerType>(tensor.getElementType());
    return integer && integer.getWidth() == 32 && !integer.isUnsigned();
}

bool isTensorShape(mlir::Value value, llvm::ArrayRef<int64_t> shape) {
    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    return tensor && tensor.hasStaticShape() && tensor.getShape() == shape;
}

bool isSingleRowFlatten(mlir::stablehlo::ReshapeOp reshape_op) {
    auto input = mlir::dyn_cast<mlir::RankedTensorType>(
        reshape_op.getOperand().getType());
    auto output = mlir::dyn_cast<mlir::RankedTensorType>(
        reshape_op.getResult().getType());
    if (!input || !output || !input.hasStaticShape() ||
        !output.hasStaticShape() ||
        input.getElementType() != output.getElementType() ||
        input.getRank() != 2 || output.getRank() != 1) {
        return false;
    }
    return input.getShape()[0] == 1 &&
           output.getShape()[0] == input.getShape()[1];
}

std::optional<int64_t> topKCompositeK(mlir::stablehlo::CompositeOp composite_op) {
    if (composite_op.getName() != "chlo.top_k") {
        return std::nullopt;
    }
    auto attrs = composite_op.getCompositeAttributes();
    auto k = attrs ? attrs.getAs<mlir::IntegerAttr>("k") : nullptr;
    if (!k) {
        return std::nullopt;
    }
    return k.getInt();
}

struct MatmulTopKEpilogueRegion {
    mlir::stablehlo::CompositeOp top_k;
    mlir::stablehlo::ReshapeOp flatten;
};

// Lower the decode-logits idiom to a matmul with a top-k epilogue:
//   dot_general : tensor<1xNxbf16>
//   reshape     : tensor<1xNxbf16> -> tensor<Nxbf16>
//   chlo.top_k  : k = 1
// The full logits and reshape are single-use and are not materialized.
std::optional<MatmulTopKEpilogueRegion> collectMatmulTopKEpilogue(
    mlir::stablehlo::DotGeneralOp dot_op) {
    if (!isBf16Tensor(dot_op.getLhs()) ||
        !isBf16Tensor(dot_op.getRhs()) ||
        !isBf16Tensor(dot_op.getResult())) {
        return std::nullopt;
    }
    auto result_type = mlir::cast<mlir::RankedTensorType>(
        dot_op.getResult().getType());
    if (!result_type.hasStaticShape() || result_type.getRank() != 2 ||
        result_type.getShape()[0] != 1) {
        return std::nullopt;
    }

    auto* reshape_user = singleUser(dot_op.getResult());
    auto flatten = reshape_user
        ? mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(reshape_user)
        : nullptr;
    if (!flatten || !isSingleRowFlatten(flatten)) {
        return std::nullopt;
    }

    auto* top_k_user = singleUser(flatten.getResult());
    auto top_k = top_k_user
        ? mlir::dyn_cast<mlir::stablehlo::CompositeOp>(top_k_user)
        : nullptr;
    if (!top_k || top_k->getNumOperands() != 1 ||
        top_k->getOperand(0) != flatten.getResult() ||
        top_k->getNumResults() != 2) {
        return std::nullopt;
    }

    auto k = topKCompositeK(top_k);
    if (!k.has_value() || *k != 1) {
        return std::nullopt;
    }
    if (!isBf16Tensor(top_k->getResult(0)) ||
        !isS32Tensor(top_k->getResult(1)) ||
        !isTensorShape(top_k->getResult(0), {1}) ||
        !isTensorShape(top_k->getResult(1), {1})) {
        return std::nullopt;
    }

    return MatmulTopKEpilogueRegion{
        .top_k = top_k,
        .flatten = flatten,
    };
}

void addMatmulDimensions(
    mlir::stablehlo::DotGeneralOp dot_op,
    tt::MatmulOp& matmul) {
    auto dims = dot_op.getDotDimensionNumbers();
    for (int64_t dim : dims.getLhsBatchingDimensions()) {
        matmul.add_lhs_batching_dimensions(dim);
    }
    for (int64_t dim : dims.getRhsBatchingDimensions()) {
        matmul.add_rhs_batching_dimensions(dim);
    }
    for (int64_t dim : dims.getLhsContractingDimensions()) {
        matmul.add_lhs_contracting_dimensions(dim);
    }
    for (int64_t dim : dims.getRhsContractingDimensions()) {
        matmul.add_rhs_contracting_dimensions(dim);
    }
}

bool addMatmulOp(
    mlir::stablehlo::DotGeneralOp dot_op,
    const MatmulTopKEpilogueRegion* top_k_epilogue,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error) {
    uint32_t lhs_id = 0;
    uint32_t rhs_id = 0;
    uint32_t matmul_output_id = 0;
    if (!addValueDesc(dot_op.getLhs(), executable, value_ids, error, lhs_id) ||
        !addValueDesc(dot_op.getRhs(), executable, value_ids, error, rhs_id) ||
        !addValueDesc(dot_op.getResult(), executable, value_ids, error, matmul_output_id)) {
        return false;
    }

    auto* op = executable.add_ops();
    op->set_output_id(matmul_output_id);
    auto* matmul = op->mutable_matmul();
    matmul->set_lhs_id(lhs_id);
    matmul->set_rhs_id(rhs_id);
    addMatmulDimensions(dot_op, *matmul);

    if (top_k_epilogue) {
        uint32_t values_id = 0;
        uint32_t indices_id = 0;
        if (!addValueDesc(top_k_epilogue->top_k->getResult(0), executable, value_ids, error, values_id) ||
            !addValueDesc(top_k_epilogue->top_k->getResult(1), executable, value_ids, error, indices_id)) {
            return false;
        }
        op->set_output_id(values_id);
        auto* epilogue = matmul->mutable_top_k_epilogue();
        epilogue->set_matmul_output_id(matmul_output_id);
        epilogue->set_indices_id(indices_id);
        epilogue->set_k(1);
    }
    return true;
}

bool addIntegerMaximumOp(
    mlir::stablehlo::MaxOp max_op,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error) {
    if (valueElementType(max_op.getResult()) != tt::TensorDesc::ELEMENT_TYPE_S32) {
        return false;
    }

    uint32_t lhs_id = 0;
    uint32_t rhs_id = 0;
    uint32_t output_id = 0;
    uint32_t pred_id = 0;
    if (!addValueDesc(max_op.getLhs(), executable, value_ids, error, lhs_id) ||
        !addValueDesc(max_op.getRhs(), executable, value_ids, error, rhs_id) ||
        !addValueDesc(max_op.getResult(), executable, value_ids, error, output_id) ||
        !addSyntheticValueDescLike(
            max_op.getResult(),
            tt::TensorDesc::ELEMENT_TYPE_PRED,
            executable,
            error,
            pred_id)) {
        return false;
    }

    auto* compare = executable.add_ops();
    compare->set_output_id(pred_id);
    auto* compare_op = compare->mutable_fused_elementwise();
    compare_op->add_input_ids(lhs_id);
    compare_op->add_input_ids(rhs_id);

    auto* lhs = compare_op->add_nodes();
    lhs->set_kind(tt::FusedElementwiseOp::Node::KIND_INPUT);
    lhs->set_input_index(0);
    lhs->set_element_type(tt::TensorDesc::ELEMENT_TYPE_S32);

    auto* rhs = compare_op->add_nodes();
    rhs->set_kind(tt::FusedElementwiseOp::Node::KIND_INPUT);
    rhs->set_input_index(1);
    rhs->set_element_type(tt::TensorDesc::ELEMENT_TYPE_S32);

    auto* gt = compare_op->add_nodes();
    gt->set_kind(tt::FusedElementwiseOp::Node::KIND_COMPARE);
    gt->add_input_nodes(0);
    gt->add_input_nodes(1);
    gt->set_element_type(tt::TensorDesc::ELEMENT_TYPE_PRED);
    gt->set_compare_direction(tt::FusedElementwiseOp::Node::DIRECTION_GT);

    auto* select = executable.add_ops();
    select->set_output_id(output_id);
    select->mutable_select()->set_pred_id(pred_id);
    select->mutable_select()->set_on_true_id(lhs_id);
    select->mutable_select()->set_on_false_id(rhs_id);
    return true;
}

bool addIntegerNegateOp(
    mlir::stablehlo::NegOp neg_op,
    tt::Executable& executable,
    llvm::DenseMap<mlir::Value, uint32_t>& value_ids,
    std::string& error) {
    if (valueElementType(neg_op.getResult()) != tt::TensorDesc::ELEMENT_TYPE_S32) {
        return false;
    }

    uint32_t operand_id = 0;
    uint32_t output_id = 0;
    uint32_t zero_id = 0;
    if (!addValueDesc(neg_op.getOperand(), executable, value_ids, error, operand_id) ||
        !addValueDesc(neg_op.getResult(), executable, value_ids, error, output_id) ||
        !addSyntheticValueDescLike(
            neg_op.getResult(),
            tt::TensorDesc::ELEMENT_TYPE_S32,
            executable,
            error,
            zero_id)) {
        return false;
    }

    addConstantOp(executable, zero_id, 0);

    auto* subtract = executable.add_ops();
    subtract->set_output_id(output_id);
    auto* subtract_op = subtract->mutable_fused_elementwise();
    subtract_op->add_input_ids(zero_id);
    subtract_op->add_input_ids(operand_id);

    auto* zero = subtract_op->add_nodes();
    zero->set_kind(tt::FusedElementwiseOp::Node::KIND_INPUT);
    zero->set_input_index(0);
    zero->set_element_type(tt::TensorDesc::ELEMENT_TYPE_S32);

    auto* operand = subtract_op->add_nodes();
    operand->set_kind(tt::FusedElementwiseOp::Node::KIND_INPUT);
    operand->set_input_index(1);
    operand->set_element_type(tt::TensorDesc::ELEMENT_TYPE_S32);

    auto* sub = subtract_op->add_nodes();
    sub->set_kind(tt::FusedElementwiseOp::Node::KIND_SUBTRACT);
    sub->add_input_nodes(0);
    sub->add_input_nodes(1);
    sub->set_element_type(tt::TensorDesc::ELEMENT_TYPE_S32);
    return true;
}

bool lowerToExecutable(FuncOp func, tt::Executable& executable, std::string& error) {
    if (func.empty()) {
        error = "entry function contains no executable operations";
        return false;
    }
    if (func.getBlocks().size() != 1) {
        error = "multi-block entry functions are not currently supported";
        return false;
    }

    llvm::DenseMap<mlir::Value, uint32_t> value_ids;
    auto fused_elementwise = buildFusedElementwisePlan(func);
    llvm::DenseSet<mlir::Operation*> matmul_top_k_covered_ops;

    for (auto [index, argument] : llvm::enumerate(func.getArguments())) {
        uint32_t output_id = 0;
        if (!addValueDesc(argument, executable, value_ids, error, output_id)) {
            return false;
        }
        auto* parameter = executable.add_ops();
        parameter->set_output_id(output_id);
        parameter->mutable_parameter()->set_parameter_index(index);
    }

    for (mlir::Operation& op : func.front()) {
        if (auto return_op = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
            for (mlir::Value operand : return_op.getOperands()) {
                uint32_t output_id = 0;
                if (!addValueDesc(operand, executable, value_ids, error, output_id)) {
                    return false;
                }
                executable.add_output_ids(output_id);
            }
            continue;
        }

        auto fused_root = fused_elementwise.roots.find(&op);
        if (fused_root != fused_elementwise.roots.end()) {
            if (!addFusedElementwiseOp(
                    &op,
                    fused_root->second,
                    executable,
                    value_ids,
                    error)) {
                return false;
            }
            continue;
        }
        if (fused_elementwise.covered_ops.contains(&op)) {
            continue;
        }
        if (matmul_top_k_covered_ops.contains(&op)) {
            continue;
        }

        if (auto constant_op = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
            if (!addConstantValueOp(constant_op.getResult(), executable, value_ids, error)) {
                return false;
            }
            continue;
        }

        if (auto broadcast_op = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
            uint32_t output_id = 0;
            if (!addValueDesc(broadcast_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            if (broadcast_op.getOperand().getDefiningOp<mlir::stablehlo::ConstantOp>()) {
                auto packed_value = packedConstantValue(broadcast_op.getOperand(), error);
                if (!packed_value) {
                    return false;
                }
                addConstantOp(executable, output_id, *packed_value);
                continue;
            }

            uint32_t operand_id = 0;
            if (!addValueDesc(broadcast_op.getOperand(), executable, value_ids, error, operand_id)) {
                return false;
            }
            auto* broadcast = executable.add_ops();
            broadcast->set_output_id(output_id);
            broadcast->mutable_broadcast_in_dim()->set_operand_id(operand_id);
            for (int64_t dim : broadcast_op.getBroadcastDimensions()) {
                broadcast->mutable_broadcast_in_dim()->add_broadcast_dimensions(dim);
            }
            continue;
        }

        if (auto select_op = mlir::dyn_cast<mlir::stablehlo::SelectOp>(op)) {
            uint32_t pred_id = 0;
            uint32_t on_true_id = 0;
            uint32_t on_false_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(select_op.getPred(), executable, value_ids, error, pred_id) ||
                !addValueDesc(select_op.getOnTrue(), executable, value_ids, error, on_true_id) ||
                !addValueDesc(select_op.getOnFalse(), executable, value_ids, error, on_false_id) ||
                !addValueDesc(select_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* select = executable.add_ops();
            select->set_output_id(output_id);
            select->mutable_select()->set_pred_id(pred_id);
            select->mutable_select()->set_on_true_id(on_true_id);
            select->mutable_select()->set_on_false_id(on_false_id);
            continue;
        }

        if (auto neg_op = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
            if (!addIntegerNegateOp(neg_op, executable, value_ids, error)) {
                if (error.empty()) {
                    error = "unsupported stablehlo.negate dtype";
                }
                return false;
            }
            continue;
        }

        if (auto max_op = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
            if (!addIntegerMaximumOp(max_op, executable, value_ids, error)) {
                if (error.empty()) {
                    error = "unsupported stablehlo.maximum dtype";
                }
                return false;
            }
            continue;
        }

        if (auto composite_op = mlir::dyn_cast<mlir::stablehlo::CompositeOp>(op)) {
            if (composite_op.getName() != "chlo.top_k") {
                error = "unsupported stablehlo composite: " + composite_op.getName().str();
                return false;
            }
            if (composite_op->getNumOperands() != 1 || composite_op->getNumResults() != 2) {
                error = "top_k composite must have one operand and two results";
                return false;
            }
            auto k = topKCompositeK(composite_op);
            if (!k) {
                error = "top_k composite is missing k";
                return false;
            }
            if (!addTopKOp(
                    composite_op->getOperand(0),
                    composite_op->getResult(0),
                    composite_op->getResult(1),
                    *k,
                    executable,
                    value_ids,
                    error)) {
                return false;
            }
            continue;
        }

        if (auto concatenate_op = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(op)) {
            uint32_t output_id = 0;
            if (!addValueDesc(concatenate_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* concatenate = executable.add_ops();
            concatenate->set_output_id(output_id);
            for (mlir::Value input : concatenate_op.getInputs()) {
                uint32_t input_id = 0;
                if (!addValueDesc(input, executable, value_ids, error, input_id)) {
                    return false;
                }
                concatenate->mutable_concatenate()->add_input_ids(input_id);
            }
            concatenate->mutable_concatenate()->set_dimension(concatenate_op.getDimension());
            continue;
        }

        if (auto reshape_op = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(reshape_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(reshape_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* reshape = executable.add_ops();
            reshape->set_output_id(output_id);
            reshape->mutable_reshape()->set_operand_id(operand_id);
            continue;
        }

        if (auto bitcast_op = mlir::dyn_cast<mlir::stablehlo::BitcastConvertOp>(op)) {
            auto operand_type = mlir::cast<mlir::RankedTensorType>(
                bitcast_op.getOperand().getType()).getElementType();
            auto result_type = mlir::cast<mlir::RankedTensorType>(
                bitcast_op->getResult(0).getType()).getElementType();
            auto operand_width = elementBitWidth(operand_type);
            auto result_width = elementBitWidth(result_type);
            if (!operand_width || !result_width || *operand_width != *result_width) {
                std::string input_type;
                std::string output_type;
                llvm::raw_string_ostream input_os(input_type);
                llvm::raw_string_ostream output_os(output_type);
                operand_type.print(input_os);
                result_type.print(output_os);
                error = "bitcast_convert requires equal-width element types: " +
                        input_os.str() + " -> " + output_os.str();
                return false;
            }

            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(bitcast_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(bitcast_op->getResult(0), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* reshape = executable.add_ops();
            reshape->set_output_id(output_id);
            reshape->mutable_reshape()->set_operand_id(operand_id);
            continue;
        }

        if (auto convert_op = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
            auto operand_element = staticValueElementType(convert_op.getOperand());
            auto result_element = staticValueElementType(convert_op.getResult());
            if (!operand_element || !result_element || *operand_element != *result_element) {
                std::string input_type;
                std::string output_type;
                llvm::raw_string_ostream input_os(input_type);
                llvm::raw_string_ostream output_os(output_type);
                mlir::cast<mlir::RankedTensorType>(
                    convert_op.getOperand().getType()).getElementType().print(input_os);
                mlir::cast<mlir::RankedTensorType>(
                    convert_op.getResult().getType()).getElementType().print(output_os);
                error = "standalone convert requires matching backend element types: " +
                        input_os.str() + " -> " + output_os.str();
                return false;
            }

            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(convert_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(convert_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* reshape = executable.add_ops();
            reshape->set_output_id(output_id);
            reshape->mutable_reshape()->set_operand_id(operand_id);
            continue;
        }

        if (auto slice_op = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(slice_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(slice_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* slice = executable.add_ops();
            slice->set_output_id(output_id);
            slice->mutable_slice()->set_operand_id(operand_id);
            for (int64_t start : slice_op.getStartIndices()) {
                slice->mutable_slice()->add_start_indices(start);
            }
            for (int64_t limit : slice_op.getLimitIndices()) {
                slice->mutable_slice()->add_limit_indices(limit);
            }
            for (int64_t stride : slice_op.getStrides()) {
                slice->mutable_slice()->add_strides(stride);
            }
            continue;
        }

        if (auto transpose_op = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(transpose_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(transpose_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* transpose = executable.add_ops();
            transpose->set_output_id(output_id);
            transpose->mutable_transpose()->set_operand_id(operand_id);
            for (int64_t dim : transpose_op.getPermutation()) {
                transpose->mutable_transpose()->add_permutation(dim);
            }
            continue;
        }

        if (auto scatter_op = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(op)) {
            if (!isSetScatter(scatter_op, error)) {
                return false;
            }

            mlir::Value operand = *scatter_op.getInputs().begin();
            mlir::Value updates = *scatter_op.getUpdates().begin();
            uint32_t operand_id = 0;
            uint32_t start_indices_id = 0;
            uint32_t updates_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(operand, executable, value_ids, error, operand_id) ||
                !addValueDesc(scatter_op.getScatterIndices(), executable, value_ids, error, start_indices_id) ||
                !addValueDesc(updates, executable, value_ids, error, updates_id) ||
                !addValueDesc(scatter_op->getResult(0), executable, value_ids, error, output_id)) {
                return false;
            }

            auto dims = scatter_op.getScatterDimensionNumbers();
            auto* scatter = executable.add_ops();
            scatter->set_output_id(output_id);
            scatter->mutable_scatter()->set_operand_id(operand_id);
            scatter->mutable_scatter()->set_start_indices_id(start_indices_id);
            scatter->mutable_scatter()->set_updates_id(updates_id);
            for (int64_t dim : dims.getUpdateWindowDims()) {
                scatter->mutable_scatter()->add_update_window_dims(dim);
            }
            for (int64_t dim : dims.getInsertedWindowDims()) {
                scatter->mutable_scatter()->add_inserted_window_dims(dim);
            }
            for (int64_t dim : dims.getInputBatchingDims()) {
                scatter->mutable_scatter()->add_input_batching_dims(dim);
            }
            for (int64_t dim : dims.getScatterIndicesBatchingDims()) {
                scatter->mutable_scatter()->add_scatter_indices_batching_dims(dim);
            }
            for (int64_t dim : dims.getScatterDimsToOperandDims()) {
                scatter->mutable_scatter()->add_scatter_dims_to_operand_dims(dim);
            }
            scatter->mutable_scatter()->set_index_vector_dim(dims.getIndexVectorDim());
            scatter->mutable_scatter()->set_indices_are_sorted(scatter_op.getIndicesAreSorted());
            scatter->mutable_scatter()->set_unique_indices(scatter_op.getUniqueIndices());
            continue;
        }

        if (auto bitwise_kind = bitwiseBinaryKind(&op)) {
            if (!addBitwiseBinaryOp(
                    op.getOperand(0),
                    op.getOperand(1),
                    op.getResult(0),
                    *bitwise_kind,
                    executable,
                    value_ids,
                    error)) {
                return false;
            }
            continue;
        }

        if (auto custom_call_op = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
            if (custom_call_op->getNumResults() != 1) {
                error = "only single-result custom_call ops are currently supported";
                return false;
            }

            auto call_target = custom_call_op.getCallTargetName();
            if (call_target == libtt::mlir_frontend::kSdpaDecodeTarget) {
                if (!addSdpaDecodeOp(custom_call_op, executable, value_ids, error)) {
                    return false;
                }
                continue;
            }
            if ((call_target == "annotate_device_placement" || call_target == "Sharding") &&
                !custom_call_op.getHasSideEffect()) {
                auto inputs = custom_call_op.getInputs();
                if (inputs.size() != 1) {
                    error = "identity custom_call op must have exactly one input";
                    return false;
                }
                mlir::Value input = inputs.front();
                if (input.getType() != custom_call_op.getResult(0).getType()) {
                    error = "identity custom_call input and result types must match";
                    return false;
                }
                uint32_t input_id = 0;
                if (!addValueDesc(input, executable, value_ids, error, input_id)) {
                    return false;
                }
                value_ids.try_emplace(custom_call_op.getResult(0), input_id);
                continue;
            }

            uint32_t output_id = 0;
            if (!addValueDesc(custom_call_op->getResult(0), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* custom_call = executable.add_ops();
            custom_call->set_output_id(output_id);
            custom_call->mutable_custom_call()->set_call_target_name(
                custom_call_op.getCallTargetName().str());
            custom_call->mutable_custom_call()->set_has_side_effect(
                custom_call_op.getHasSideEffect());
            for (mlir::Value input : custom_call_op.getInputs()) {
                uint32_t input_id = 0;
                if (!addValueDesc(input, executable, value_ids, error, input_id)) {
                    return false;
                }
                custom_call->mutable_custom_call()->add_input_ids(input_id);
            }
            continue;
        }

        if (auto reduce_op = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
            if (isArgmaxReduceOp(reduce_op)) {
                auto inputs = reduce_op.getInputs();
                mlir::Value values_input = *inputs.begin();
                if (!addTopKOp(
                        values_input,
                        reduce_op->getResult(0),
                        reduce_op->getResult(1),
                        1,
                        executable,
                        value_ids,
                        error)) {
                    return false;
                }
                continue;
            }

            auto reducer = mapReduceReducer(reduce_op, error);
            if (!reducer) {
                return false;
            }

            uint32_t input_id = 0;
            uint32_t init_value_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(*reduce_op.getInputs().begin(), executable, value_ids, error, input_id) ||
                !addValueDesc(*reduce_op.getInitValues().begin(), executable, value_ids, error, init_value_id) ||
                !addValueDesc(reduce_op->getResult(0), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* reduce = executable.add_ops();
            reduce->set_output_id(output_id);
            reduce->mutable_reduce()->add_input_ids(input_id);
            reduce->mutable_reduce()->add_init_value_ids(init_value_id);
            for (int64_t dim : reduce_op.getDimensions()) {
                reduce->mutable_reduce()->add_dimensions(dim);
            }
            reduce->mutable_reduce()->set_reducer(*reducer);
            continue;
        }

        if (auto reduce_window_op = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(op)) {
            auto reducer = mapReduceWindowReducer(reduce_window_op, error);
            if (!reducer) {
                return false;
            }

            auto window_dimensions = reduce_window_op.getWindowDimensions().vec();
            auto window_strides = optionalArrayOrOnes(
                reduce_window_op.getWindowStrides(),
                window_dimensions.size());
            auto base_dilations = optionalArrayOrOnes(
                reduce_window_op.getBaseDilations(),
                window_dimensions.size());
            auto window_dilations = optionalArrayOrOnes(
                reduce_window_op.getWindowDilations(),
                window_dimensions.size());
            std::vector<int64_t> padding_low;
            std::vector<int64_t> padding_high;
            if (!reduceWindowPaddingVectors(
                    reduce_window_op.getPadding(),
                    window_dimensions.size(),
                    padding_low,
                    padding_high,
                    error)) {
                return false;
            }

            uint32_t input_id = 0;
            uint32_t init_value_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(*reduce_window_op.getInputs().begin(), executable, value_ids, error, input_id) ||
                !addValueDesc(*reduce_window_op.getInitValues().begin(), executable, value_ids, error, init_value_id) ||
                !addValueDesc(reduce_window_op->getResult(0), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* reduce_window = executable.add_ops();
            reduce_window->set_output_id(output_id);
            reduce_window->mutable_reduce_window()->add_input_ids(input_id);
            reduce_window->mutable_reduce_window()->add_init_value_ids(init_value_id);
            for (int64_t value : window_dimensions) {
                reduce_window->mutable_reduce_window()->add_window_dimensions(value);
            }
            for (int64_t value : window_strides) {
                reduce_window->mutable_reduce_window()->add_window_strides(value);
            }
            for (int64_t value : base_dilations) {
                reduce_window->mutable_reduce_window()->add_base_dilations(value);
            }
            for (int64_t value : window_dilations) {
                reduce_window->mutable_reduce_window()->add_window_dilations(value);
            }
            for (int64_t value : padding_low) {
                reduce_window->mutable_reduce_window()->add_padding_low(value);
            }
            for (int64_t value : padding_high) {
                reduce_window->mutable_reduce_window()->add_padding_high(value);
            }
            reduce_window->mutable_reduce_window()->set_reducer(*reducer);
            continue;
        }

        if (auto iota_op = mlir::dyn_cast<mlir::stablehlo::IotaOp>(op)) {
            uint32_t output_id = 0;
            if (!addValueDesc(iota_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* iota = executable.add_ops();
            iota->set_output_id(output_id);
            iota->mutable_iota()->set_iota_dimension(iota_op.getIotaDimension());
            continue;
        }

        if (auto dot_op = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
            auto epilogue = collectMatmulTopKEpilogue(dot_op);
            if (!addMatmulOp(
                    dot_op,
                    epilogue.has_value() ? &*epilogue : nullptr,
                    executable,
                    value_ids,
                    error)) {
                return false;
            }
            if (epilogue.has_value()) {
                matmul_top_k_covered_ops.insert(epilogue->flatten.getOperation());
                matmul_top_k_covered_ops.insert(epilogue->top_k.getOperation());
            }
            continue;
        }

        if (auto gather_op = mlir::dyn_cast<mlir::stablehlo::GatherOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t start_indices_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(gather_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(gather_op.getStartIndices(), executable, value_ids, error, start_indices_id) ||
                !addValueDesc(gather_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto dims = gather_op.getDimensionNumbers();
            auto* gather = executable.add_ops();
            gather->set_output_id(output_id);
            gather->mutable_gather()->set_operand_id(operand_id);
            gather->mutable_gather()->set_start_indices_id(start_indices_id);
            for (int64_t dim : dims.getOffsetDims()) {
                gather->mutable_gather()->add_offset_dims(dim);
            }
            for (int64_t dim : dims.getCollapsedSliceDims()) {
                gather->mutable_gather()->add_collapsed_slice_dims(dim);
            }
            for (int64_t dim : dims.getOperandBatchingDims()) {
                gather->mutable_gather()->add_operand_batching_dims(dim);
            }
            for (int64_t dim : dims.getStartIndicesBatchingDims()) {
                gather->mutable_gather()->add_start_indices_batching_dims(dim);
            }
            for (int64_t dim : dims.getStartIndexMap()) {
                gather->mutable_gather()->add_start_index_map(dim);
            }
            gather->mutable_gather()->set_index_vector_dim(dims.getIndexVectorDim());
            for (int64_t size : gather_op.getSliceSizes()) {
                gather->mutable_gather()->add_slice_sizes(size);
            }
            gather->mutable_gather()->set_indices_are_sorted(gather_op.getIndicesAreSorted());
            continue;
        }

        error = "unsupported entry op: " + op.getName().getStringRef().str();
        return false;
    }

    if (executable.output_ids_size() == 0) {
        error = "entry function must return at least one value";
        return false;
    }

    return true;
}

bool eraseDeadOps(FuncOp func) {
    bool changed = false;
    bool local_changed = true;
    while (local_changed) {
        local_changed = false;
        llvm::SmallVector<mlir::Operation*> dead_ops;
        for (mlir::Operation& op : func.front()) {
            if (mlir::isa<mlir::func::ReturnOp>(op)) {
                continue;
            }
            if (auto custom_call = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
                custom_call && custom_call.getHasSideEffect()) {
                continue;
            }
            if (op.use_empty()) {
                dead_ops.push_back(&op);
            }
        }
        for (mlir::Operation* op : dead_ops) {
            op->erase();
            local_changed = true;
            changed = true;
        }
    }
    return changed;
}

tt::AnalysisResult makeResult(tt::AnalysisResult::Status status, const std::string& error = "") {
    tt::AnalysisResult result;
    result.set_status(status);
    result.set_error_message(error);
    return result;
}

bool emitResult(
    const tt::AnalysisResult& result,
    TT_MlirAllocOutput alloc_output,
    void* user_data) {
    if (!alloc_output) {
        return false;
    }

    size_t size = result.ByteSizeLong();
    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    char* data = alloc_output(size, user_data);
    if (!data && size != 0) {
        return false;
    }
    return result.SerializeToArray(data, static_cast<int>(size));
}

}  // namespace

extern "C" bool TT_MlirAnalyzeProgram(
    const char* format,
    size_t format_size,
    const char* code,
    size_t code_size,
    TT_MlirAllocOutput alloc_output,
    void* user_data) {
    if (!format || !code) {
        return emitResult(makeResult(
            tt::AnalysisResult::STATUS_PARSE_ERROR,
            "program format and code must not be null"), alloc_output, user_data);
    }

    mlir::MLIRContext context;
    registerDialects(context);
    context.allowUnregisteredDialects();

    auto module = parseModule(
        context,
        llvm::StringRef(format, format_size),
        llvm::StringRef(code, code_size));
    if (!module) {
        return emitResult(makeResult(
            tt::AnalysisResult::STATUS_PARSE_ERROR,
            "failed to parse StableHLO/MLIR program"), alloc_output, user_data);
    }

    std::string error;
    if (!runCleanupPasses(context, *module, error)) {
        return emitResult(makeResult(
            tt::AnalysisResult::STATUS_INTERNAL_ERROR,
            error.empty() ? "failed to run MLIR cleanup passes" : error), alloc_output, user_data);
    }

    auto entry = findEntryFunction(*module);
    if (!entry.has_value()) {
        return emitResult(makeResult(
            tt::AnalysisResult::STATUS_PARSE_ERROR,
            "module does not contain a function"), alloc_output, user_data);
    }
    eraseDeadOps(*entry);

    tt::AnalysisResult result;
    result.set_status(tt::AnalysisResult::STATUS_OK);

    if (!fillProgramSignature(*entry, result, error)) {
        return emitResult(
            makeResult(tt::AnalysisResult::STATUS_UNSUPPORTED, error),
            alloc_output,
            user_data);
    }

    if (!lowerToExecutable(*entry, *result.mutable_executable(), error)) {
        return emitResult(
            makeResult(tt::AnalysisResult::STATUS_UNSUPPORTED, error),
            alloc_output,
            user_data);
    }

    return emitResult(result, alloc_output, user_data);
}
