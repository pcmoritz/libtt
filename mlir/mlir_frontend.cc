#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/executable.pb.h"
#include "stablehlo/dialect/ChloOps.h"
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

void registerDialects(mlir::MLIRContext& context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    mlir::func::registerAllExtensions(registry);
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
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

bool runCleanupPasses(mlir::MLIRContext& context, mlir::ModuleOp module) {
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    return mlir::succeeded(pm.run(module));
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
            default:
                return tt::TensorDesc::ELEMENT_TYPE_UNKNOWN;
        }
    }
    return tt::TensorDesc::ELEMENT_TYPE_UNKNOWN;
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
        error = "unsupported tensor element type";
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

    auto element_type = mlir::cast<mlir::ShapedType>(dense.getType()).getElementType();
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
        if (integer.getWidth() <= 32) {
            auto bits = dense.getSplatValue<llvm::APInt>();
            return static_cast<uint32_t>(bits.getZExtValue());
        }
    }
    error = "only bf16/f16/f32 and <=32-bit integer splat constants are currently supported";
    return std::nullopt;
}

void addConstantOp(tt::Executable& executable, uint32_t output_id, uint32_t packed_value) {
    auto* constant = executable.add_ops();
    constant->set_output_id(output_id);
    constant->mutable_constant()->set_packed_value(packed_value);
}

tt::CompareOp::Direction mapCompareDirection(
    mlir::stablehlo::ComparisonDirection direction) {
    switch (direction) {
        case mlir::stablehlo::ComparisonDirection::EQ:
            return tt::CompareOp::DIRECTION_EQ;
        case mlir::stablehlo::ComparisonDirection::NE:
            return tt::CompareOp::DIRECTION_NE;
        case mlir::stablehlo::ComparisonDirection::GE:
            return tt::CompareOp::DIRECTION_GE;
        case mlir::stablehlo::ComparisonDirection::GT:
            return tt::CompareOp::DIRECTION_GT;
        case mlir::stablehlo::ComparisonDirection::LE:
            return tt::CompareOp::DIRECTION_LE;
        case mlir::stablehlo::ComparisonDirection::LT:
            return tt::CompareOp::DIRECTION_LT;
    }
    return tt::CompareOp::DIRECTION_EQ;
}

std::optional<tt::ReduceOp::Reducer> mapReduceReducer(
    mlir::stablehlo::ReduceOp reduce_op,
    std::string& error) {
    if (reduce_op.getInputs().size() != 1 ||
        reduce_op.getInitValues().size() != 1 ||
        reduce_op->getNumResults() != 1) {
        error = "only single-input single-result reduce ops are currently supported";
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
    if (mlir::isa<mlir::stablehlo::MulOp>(reducer_op)) {
        return tt::ReduceOp::REDUCER_MUL;
    }

    error = "unsupported reduce reducer: " + reducer_op->getName().getStringRef().str();
    return std::nullopt;
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
            if (return_op.getNumOperands() != 1) {
                error = "only single-result functions are currently supported";
                return false;
            }
            uint32_t output_id = 0;
            if (!addValueDesc(return_op.getOperand(0), executable, value_ids, error, output_id)) {
                return false;
            }
            executable.add_output_ids(output_id);
            continue;
        }

        if (auto constant_op = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op)) {
            uint32_t output_id = 0;
            if (!addValueDesc(constant_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }
            auto packed_value = packedConstantValue(constant_op.getResult(), error);
            if (!packed_value) {
                return false;
            }
            addConstantOp(executable, output_id, *packed_value);
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

        if (auto compare_op = mlir::dyn_cast<mlir::stablehlo::CompareOp>(op)) {
            uint32_t lhs_id = 0;
            uint32_t rhs_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(compare_op.getLhs(), executable, value_ids, error, lhs_id) ||
                !addValueDesc(compare_op.getRhs(), executable, value_ids, error, rhs_id) ||
                !addValueDesc(compare_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* compare = executable.add_ops();
            compare->set_output_id(output_id);
            compare->mutable_compare()->set_lhs_id(lhs_id);
            compare->mutable_compare()->set_rhs_id(rhs_id);
            compare->mutable_compare()->set_direction(
                mapCompareDirection(compare_op.getComparisonDirection()));
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

        if (auto add_op = mlir::dyn_cast<mlir::stablehlo::AddOp>(op)) {
            uint32_t lhs_id = 0;
            uint32_t rhs_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(add_op.getLhs(), executable, value_ids, error, lhs_id) ||
                !addValueDesc(add_op.getRhs(), executable, value_ids, error, rhs_id) ||
                !addValueDesc(add_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* add = executable.add_ops();
            add->set_output_id(output_id);
            add->mutable_add()->set_lhs_id(lhs_id);
            add->mutable_add()->set_rhs_id(rhs_id);
            continue;
        }

        if (auto subtract_op = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(op)) {
            uint32_t lhs_id = 0;
            uint32_t rhs_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(subtract_op.getLhs(), executable, value_ids, error, lhs_id) ||
                !addValueDesc(subtract_op.getRhs(), executable, value_ids, error, rhs_id) ||
                !addValueDesc(subtract_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* subtract = executable.add_ops();
            subtract->set_output_id(output_id);
            subtract->mutable_subtract()->set_lhs_id(lhs_id);
            subtract->mutable_subtract()->set_rhs_id(rhs_id);
            continue;
        }

        if (auto mul_op = mlir::dyn_cast<mlir::stablehlo::MulOp>(op)) {
            uint32_t lhs_id = 0;
            uint32_t rhs_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(mul_op.getLhs(), executable, value_ids, error, lhs_id) ||
                !addValueDesc(mul_op.getRhs(), executable, value_ids, error, rhs_id) ||
                !addValueDesc(mul_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* multiply = executable.add_ops();
            multiply->set_output_id(output_id);
            multiply->mutable_multiply()->set_lhs_id(lhs_id);
            multiply->mutable_multiply()->set_rhs_id(rhs_id);
            continue;
        }

        if (auto div_op = mlir::dyn_cast<mlir::stablehlo::DivOp>(op)) {
            uint32_t lhs_id = 0;
            uint32_t rhs_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(div_op.getLhs(), executable, value_ids, error, lhs_id) ||
                !addValueDesc(div_op.getRhs(), executable, value_ids, error, rhs_id) ||
                !addValueDesc(div_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* divide = executable.add_ops();
            divide->set_output_id(output_id);
            divide->mutable_divide()->set_lhs_id(lhs_id);
            divide->mutable_divide()->set_rhs_id(rhs_id);
            continue;
        }

        if (auto pow_op = mlir::dyn_cast<mlir::stablehlo::PowOp>(op)) {
            uint32_t lhs_id = 0;
            uint32_t rhs_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(pow_op.getLhs(), executable, value_ids, error, lhs_id) ||
                !addValueDesc(pow_op.getRhs(), executable, value_ids, error, rhs_id) ||
                !addValueDesc(pow_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* power = executable.add_ops();
            power->set_output_id(output_id);
            power->mutable_power()->set_lhs_id(lhs_id);
            power->mutable_power()->set_rhs_id(rhs_id);
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

        if (auto cosine_op = mlir::dyn_cast<mlir::stablehlo::CosineOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(cosine_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(cosine_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* cosine = executable.add_ops();
            cosine->set_output_id(output_id);
            cosine->mutable_cosine()->set_operand_id(operand_id);
            continue;
        }

        if (auto sine_op = mlir::dyn_cast<mlir::stablehlo::SineOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(sine_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(sine_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* sine = executable.add_ops();
            sine->set_output_id(output_id);
            sine->mutable_sine()->set_operand_id(operand_id);
            continue;
        }

        if (auto rsqrt_op = mlir::dyn_cast<mlir::stablehlo::RsqrtOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(rsqrt_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(rsqrt_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* rsqrt = executable.add_ops();
            rsqrt->set_output_id(output_id);
            rsqrt->mutable_rsqrt()->set_operand_id(operand_id);
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

        if (auto negate_op = mlir::dyn_cast<mlir::stablehlo::NegOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(negate_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(negate_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* negate = executable.add_ops();
            negate->set_output_id(output_id);
            negate->mutable_negate()->set_operand_id(operand_id);
            continue;
        }

        if (auto exponential_op = mlir::dyn_cast<mlir::stablehlo::ExpOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(exponential_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(exponential_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* exponential = executable.add_ops();
            exponential->set_output_id(output_id);
            exponential->mutable_exponential()->set_operand_id(operand_id);
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

        if (auto custom_call_op = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
            if (custom_call_op->getNumResults() != 1) {
                error = "only single-result custom_call ops are currently supported";
                return false;
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

        if (auto convert_op = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
            uint32_t operand_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(convert_op.getOperand(), executable, value_ids, error, operand_id) ||
                !addValueDesc(convert_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* convert = executable.add_ops();
            convert->set_output_id(output_id);
            convert->mutable_convert()->set_operand_id(operand_id);
            continue;
        }

        if (auto reduce_op = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(op)) {
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

        if (auto max_op = mlir::dyn_cast<mlir::stablehlo::MaxOp>(op)) {
            uint32_t lhs_id = 0;
            uint32_t rhs_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(max_op.getLhs(), executable, value_ids, error, lhs_id) ||
                !addValueDesc(max_op.getRhs(), executable, value_ids, error, rhs_id) ||
                !addValueDesc(max_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* max = executable.add_ops();
            max->set_output_id(output_id);
            max->mutable_max()->set_lhs_id(lhs_id);
            max->mutable_max()->set_rhs_id(rhs_id);
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
            auto dims = dot_op.getDotDimensionNumbers();
            uint32_t lhs_id = 0;
            uint32_t rhs_id = 0;
            uint32_t output_id = 0;
            if (!addValueDesc(dot_op.getLhs(), executable, value_ids, error, lhs_id) ||
                !addValueDesc(dot_op.getRhs(), executable, value_ids, error, rhs_id) ||
                !addValueDesc(dot_op.getResult(), executable, value_ids, error, output_id)) {
                return false;
            }

            auto* matmul = executable.add_ops();
            matmul->set_output_id(output_id);
            matmul->mutable_matmul()->set_lhs_id(lhs_id);
            matmul->mutable_matmul()->set_rhs_id(rhs_id);
            for (int64_t dim : dims.getLhsBatchingDimensions()) {
                matmul->mutable_matmul()->add_lhs_batching_dimensions(dim);
            }
            for (int64_t dim : dims.getRhsBatchingDimensions()) {
                matmul->mutable_matmul()->add_rhs_batching_dimensions(dim);
            }
            for (int64_t dim : dims.getLhsContractingDimensions()) {
                matmul->mutable_matmul()->add_lhs_contracting_dimensions(dim);
            }
            for (int64_t dim : dims.getRhsContractingDimensions()) {
                matmul->mutable_matmul()->add_rhs_contracting_dimensions(dim);
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

    if (executable.output_ids_size() != 1) {
        error = "entry function must return exactly one value";
        return false;
    }

    return true;
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

    if (!runCleanupPasses(context, *module)) {
        return emitResult(makeResult(
            tt::AnalysisResult::STATUS_INTERNAL_ERROR,
            "failed to run MLIR cleanup passes"), alloc_output, user_data);
    }

    auto entry = findEntryFunction(*module);
    if (!entry.has_value()) {
        return emitResult(makeResult(
            tt::AnalysisResult::STATUS_PARSE_ERROR,
            "module does not contain a function"), alloc_output, user_data);
    }

    tt::AnalysisResult result;
    result.set_status(tt::AnalysisResult::STATUS_OK);

    std::string error;
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
