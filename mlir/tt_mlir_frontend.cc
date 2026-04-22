#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
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
#include "mlir/tt_executable.pb.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

extern "C" {

enum TT_MlirAnalysisStatus : int32_t {
    TT_MLIR_ANALYSIS_STATUS_OK = 0,
    TT_MLIR_ANALYSIS_STATUS_PARSE_ERROR = 1,
    TT_MLIR_ANALYSIS_STATUS_UNSUPPORTED = 2,
    TT_MLIR_ANALYSIS_STATUS_INTERNAL_ERROR = 3,
};

enum TT_MlirElementType : int32_t {
    TT_MLIR_ELEMENT_TYPE_UNKNOWN = 0,
    TT_MLIR_ELEMENT_TYPE_BF16 = 1,
    TT_MLIR_ELEMENT_TYPE_F16 = 2,
    TT_MLIR_ELEMENT_TYPE_F32 = 3,
    TT_MLIR_ELEMENT_TYPE_U32 = 4,
    TT_MLIR_ELEMENT_TYPE_U16 = 5,
    TT_MLIR_ELEMENT_TYPE_U8 = 6,
    TT_MLIR_ELEMENT_TYPE_S32 = 7,
    TT_MLIR_ELEMENT_TYPE_S8 = 8,
};

struct TT_MlirAnalysis {
    int32_t status;
    int32_t output_type;
    size_t num_output_dims;
    int64_t* output_dims;
    char* optimized_program;
    size_t optimized_program_size;
    char* error_message;
};

TT_MlirAnalysis* TT_MlirAnalyzeProgram(
    const char* format,
    size_t format_size,
    const char* code,
    size_t code_size);
void TT_MlirAnalysisDestroy(TT_MlirAnalysis* analysis);
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
    const char* code,
    size_t code_size) {
    if (format != "mlir" && format != "stablehlo") {
        return nullptr;
    }

    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(code, code_size),
        "tt_mlir_program",
        false);
    if (auto module = mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), &context)) {
        return module;
    }

    return mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(code, code_size),
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

int32_t mapElementType(mlir::Type element_type) {
    switch (mapProtoElementType(element_type)) {
        case tt::TensorDesc::ELEMENT_TYPE_BF16:
            return TT_MLIR_ELEMENT_TYPE_BF16;
        case tt::TensorDesc::ELEMENT_TYPE_F16:
            return TT_MLIR_ELEMENT_TYPE_F16;
        case tt::TensorDesc::ELEMENT_TYPE_F32:
            return TT_MLIR_ELEMENT_TYPE_F32;
        case tt::TensorDesc::ELEMENT_TYPE_U32:
            return TT_MLIR_ELEMENT_TYPE_U32;
        case tt::TensorDesc::ELEMENT_TYPE_U16:
            return TT_MLIR_ELEMENT_TYPE_U16;
        case tt::TensorDesc::ELEMENT_TYPE_U8:
            return TT_MLIR_ELEMENT_TYPE_U8;
        case tt::TensorDesc::ELEMENT_TYPE_S32:
            return TT_MLIR_ELEMENT_TYPE_S32;
        case tt::TensorDesc::ELEMENT_TYPE_S8:
            return TT_MLIR_ELEMENT_TYPE_S8;
        case tt::TensorDesc::ELEMENT_TYPE_UNKNOWN:
            return TT_MLIR_ELEMENT_TYPE_UNKNOWN;
    }
    return TT_MLIR_ELEMENT_TYPE_UNKNOWN;
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
    tt::TTExecutableV1& executable,
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

bool fillOutputSignature(FuncOp func, TT_MlirAnalysis& analysis, std::string& error) {
    auto type = func.getFunctionType();
    if (type.getNumResults() != 1) {
        error = "only single-result functions are currently supported";
        return false;
    }

    auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type.getResult(0));
    if (!tensor) {
        error = "entry function result must be a ranked tensor";
        return false;
    }

    if (!tensor.hasStaticShape()) {
        error = "dynamic result shapes are not currently supported";
        return false;
    }

    analysis.output_type = mapElementType(tensor.getElementType());
    if (analysis.output_type == TT_MLIR_ELEMENT_TYPE_UNKNOWN) {
        error = "unsupported result element type";
        return false;
    }

    analysis.num_output_dims = tensor.getRank();
    if (analysis.num_output_dims == 0) {
        analysis.output_dims = nullptr;
        return true;
    }

    analysis.output_dims = static_cast<int64_t*>(
        std::malloc(sizeof(int64_t) * analysis.num_output_dims));
    if (!analysis.output_dims) {
        error = "failed to allocate output shape";
        return false;
    }

    for (size_t i = 0; i < analysis.num_output_dims; ++i) {
        analysis.output_dims[i] = tensor.getShape()[i];
    }
    return true;
}

void setCString(char*& out, const std::string& value) {
    out = static_cast<char*>(std::malloc(value.size() + 1));
    if (!out) {
        return;
    }
    std::memcpy(out, value.data(), value.size());
    out[value.size()] = '\0';
}

void setProgramBytes(TT_MlirAnalysis& analysis, const std::string& bytes) {
    analysis.optimized_program = static_cast<char*>(std::malloc(bytes.size()));
    if (!analysis.optimized_program) {
        analysis.optimized_program_size = 0;
        return;
    }
    analysis.optimized_program_size = bytes.size();
    std::memcpy(analysis.optimized_program, bytes.data(), bytes.size());
}

bool lowerToExecutable(FuncOp func, TT_MlirAnalysis& analysis, std::string& error) {
    if (func.empty()) {
        error = "entry function contains no executable operations";
        return false;
    }
    if (func.getBlocks().size() != 1) {
        error = "multi-block entry functions are not currently supported";
        return false;
    }

    tt::TTExecutableV1 executable;
    llvm::DenseMap<mlir::Value, uint32_t> value_ids;

    for (auto [index, argument] : llvm::enumerate(func.getArguments())) {
        uint32_t output_id = 0;
        if (!addValueDesc(argument, executable, value_ids, error, output_id)) {
            return false;
        }
        auto* parameter = executable.add_ops();
        parameter->set_opcode(tt::Op::OPCODE_PARAMETER);
        parameter->set_parameter_index(index);
        parameter->set_output_id(output_id);
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
            add->set_opcode(tt::Op::OPCODE_ADD);
            add->set_output_id(output_id);
            add->add_input_ids(lhs_id);
            add->add_input_ids(rhs_id);
            continue;
        }

        error = "unsupported entry op: " + op.getName().getStringRef().str();
        return false;
    }

    if (executable.output_ids_size() != 1) {
        error = "entry function must return exactly one value";
        return false;
    }

    std::string bytes;
    if (!executable.SerializeToString(&bytes)) {
        error = "failed to serialize TT executable";
        return false;
    }

    setProgramBytes(analysis, bytes);
    return analysis.optimized_program != nullptr || bytes.empty();
}

TT_MlirAnalysis* makeAnalysis() {
    auto* analysis = new TT_MlirAnalysis();
    analysis->status = TT_MLIR_ANALYSIS_STATUS_INTERNAL_ERROR;
    analysis->output_type = TT_MLIR_ELEMENT_TYPE_UNKNOWN;
    analysis->num_output_dims = 0;
    analysis->output_dims = nullptr;
    analysis->optimized_program = nullptr;
    analysis->optimized_program_size = 0;
    analysis->error_message = nullptr;
    return analysis;
}

}  // namespace

extern "C" TT_MlirAnalysis* TT_MlirAnalyzeProgram(
    const char* format,
    size_t format_size,
    const char* code,
    size_t code_size) {
    auto* analysis = makeAnalysis();
    if (!format || !code) {
        setCString(analysis->error_message, "program format and code must not be null");
        analysis->status = TT_MLIR_ANALYSIS_STATUS_PARSE_ERROR;
        return analysis;
    }

    mlir::MLIRContext context;
    registerDialects(context);

    auto module = parseModule(context, llvm::StringRef(format, format_size), code, code_size);
    if (!module) {
        setCString(analysis->error_message, "failed to parse StableHLO/MLIR program");
        analysis->status = TT_MLIR_ANALYSIS_STATUS_PARSE_ERROR;
        return analysis;
    }

    if (!runCleanupPasses(context, *module)) {
        setCString(analysis->error_message, "failed to run MLIR cleanup passes");
        analysis->status = TT_MLIR_ANALYSIS_STATUS_INTERNAL_ERROR;
        return analysis;
    }

    auto entry = findEntryFunction(*module);
    if (!entry.has_value()) {
        setCString(analysis->error_message, "module does not contain a function");
        analysis->status = TT_MLIR_ANALYSIS_STATUS_PARSE_ERROR;
        return analysis;
    }

    std::string error;
    if (!fillOutputSignature(*entry, *analysis, error)) {
        setCString(analysis->error_message, error);
        analysis->status = TT_MLIR_ANALYSIS_STATUS_UNSUPPORTED;
        return analysis;
    }

    if (!lowerToExecutable(*entry, *analysis, error)) {
        setCString(analysis->error_message, error);
        analysis->status = TT_MLIR_ANALYSIS_STATUS_UNSUPPORTED;
        return analysis;
    }

    analysis->status = TT_MLIR_ANALYSIS_STATUS_OK;
    return analysis;
}

extern "C" void TT_MlirAnalysisDestroy(TT_MlirAnalysis* analysis) {
    if (!analysis) {
        return;
    }
    std::free(analysis->output_dims);
    std::free(analysis->optimized_program);
    std::free(analysis->error_message);
    delete analysis;
}
