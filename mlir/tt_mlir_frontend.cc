#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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

enum TT_MlirExecutableKind : int32_t {
    TT_MLIR_EXECUTABLE_KIND_UNKNOWN = 0,
    TT_MLIR_EXECUTABLE_KIND_ELTWISE_ADD_BF16 = 1,
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
    int32_t executable_kind;
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

std::vector<std::string> collectEntryOps(FuncOp func) {
    std::vector<std::string> names;
    if (func.empty()) {
        return names;
    }
    for (auto& op : func.front()) {
        if (mlir::isa<mlir::func::ReturnOp>(op)) {
            continue;
        }
        names.push_back(op.getName().getStringRef().str());
    }
    return names;
}

int32_t classifyExecutableKind(const std::vector<std::string>& op_names, std::string& error) {
    if (op_names.empty()) {
        error = "entry function contains no executable operations";
        return TT_MLIR_EXECUTABLE_KIND_UNKNOWN;
    }
    if (op_names.size() != 1) {
        error = "only single-op StableHLO programs are currently supported";
        return TT_MLIR_EXECUTABLE_KIND_UNKNOWN;
    }

    const std::string& op_name = op_names.front();
    if (op_name == "stablehlo.add" || op_name == "mhlo.add") {
        return TT_MLIR_EXECUTABLE_KIND_ELTWISE_ADD_BF16;
    }

    error = "unsupported entry op: " + op_name;
    return TT_MLIR_EXECUTABLE_KIND_UNKNOWN;
}

int32_t mapElementType(mlir::Type element_type) {
    if (element_type.isBF16()) return TT_MLIR_ELEMENT_TYPE_BF16;
    if (element_type.isF16()) return TT_MLIR_ELEMENT_TYPE_F16;
    if (element_type.isF32()) return TT_MLIR_ELEMENT_TYPE_F32;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element_type)) {
        switch (integer.getWidth()) {
            case 8:
                return integer.isUnsigned() ? TT_MLIR_ELEMENT_TYPE_U8 : TT_MLIR_ELEMENT_TYPE_S8;
            case 16:
                return integer.isUnsigned() ? TT_MLIR_ELEMENT_TYPE_U16 : TT_MLIR_ELEMENT_TYPE_UNKNOWN;
            case 32:
                return integer.isUnsigned() ? TT_MLIR_ELEMENT_TYPE_U32 : TT_MLIR_ELEMENT_TYPE_S32;
            default:
                return TT_MLIR_ELEMENT_TYPE_UNKNOWN;
        }
    }
    return TT_MLIR_ELEMENT_TYPE_UNKNOWN;
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

void setProgramText(TT_MlirAnalysis& analysis, mlir::ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream os(text);
    module.print(os);
    os.flush();

    analysis.optimized_program = static_cast<char*>(std::malloc(text.size()));
    if (!analysis.optimized_program) {
        analysis.optimized_program_size = 0;
        return;
    }
    analysis.optimized_program_size = text.size();
    std::memcpy(analysis.optimized_program, text.data(), text.size());
}

TT_MlirAnalysis* makeAnalysis() {
    auto* analysis = new TT_MlirAnalysis();
    analysis->status = TT_MLIR_ANALYSIS_STATUS_INTERNAL_ERROR;
    analysis->executable_kind = TT_MLIR_EXECUTABLE_KIND_UNKNOWN;
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
    auto op_names = collectEntryOps(*entry);
    analysis->executable_kind = classifyExecutableKind(op_names, error);
    if (analysis->executable_kind == TT_MLIR_EXECUTABLE_KIND_UNKNOWN) {
        setCString(analysis->error_message, error);
        analysis->status = TT_MLIR_ANALYSIS_STATUS_UNSUPPORTED;
        return analysis;
    }

    if (!fillOutputSignature(*entry, *analysis, error)) {
        setCString(analysis->error_message, error);
        analysis->status = TT_MLIR_ANALYSIS_STATUS_UNSUPPORTED;
        return analysis;
    }

    setProgramText(*analysis, *module);
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
