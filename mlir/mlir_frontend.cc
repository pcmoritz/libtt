#include <cstdlib>
#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/executable.pb.h"
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

std::optional<uint32_t> packedConvertedConstantValue(
    mlir::stablehlo::ConvertOp convert_op,
    std::string& error) {
    mlir::Value value = convert_op.getOperand();
    while (auto broadcast_op = value.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>()) {
        value = broadcast_op.getOperand();
    }

    auto constant_op = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!constant_op) {
        return std::nullopt;
    }
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constant_op.getValue());
    if (!dense || !dense.isSplat()) {
        error = "only splat constant converts are currently supported";
        return std::nullopt;
    }

    auto input_element_type = mlir::cast<mlir::ShapedType>(dense.getType()).getElementType();
    auto output_element_type =
        mlir::cast<mlir::RankedTensorType>(convert_op.getResult().getType()).getElementType();
    if (!input_element_type.isBF16() && !input_element_type.isF16() &&
        !input_element_type.isF32()) {
        error = "constant convert currently supports only float inputs";
        return std::nullopt;
    }

    auto value_float = dense.getSplatValue<llvm::APFloat>();
    auto pack_float = [&](const llvm::fltSemantics& semantics, unsigned bit_width) {
        auto converted = value_float;
        bool loses_info = false;
        converted.convert(
            semantics,
            llvm::APFloat::rmNearestTiesToEven,
            &loses_info);
        auto bits = converted.bitcastToAPInt();
        uint32_t value = bits.extractBitsAsZExtValue(bit_width, 0);
        return bit_width == 16 ? value | (value << 16) : value;
    };

    if (output_element_type.isBF16()) {
        return pack_float(llvm::APFloat::BFloat(), 16);
    }
    if (output_element_type.isF16()) {
        return pack_float(llvm::APFloat::IEEEhalf(), 16);
    }
    if (output_element_type.isF32()) {
        return pack_float(llvm::APFloat::IEEEsingle(), 32);
    }

    error = "constant convert currently supports only float outputs";
    return std::nullopt;
}

void addConstantOp(tt::Executable& executable, uint32_t output_id, uint32_t packed_value) {
    auto* constant = executable.add_ops();
    constant->set_output_id(output_id);
    constant->mutable_constant()->set_packed_value(packed_value);
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

bool supportsFusedLeafElementType(tt::TensorDesc::ElementType element_type) {
    return isFusedFloatElementType(element_type) ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_U32 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_S32 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_U16;
}

bool supportsConvertElementType(tt::TensorDesc::ElementType element_type) {
    return element_type == tt::TensorDesc::ELEMENT_TYPE_BF16 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_F32 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_U32 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_S32 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_U16;
}

bool supportsCompareElementType(tt::TensorDesc::ElementType element_type) {
    return element_type == tt::TensorDesc::ELEMENT_TYPE_BF16 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_F32 ||
           element_type == tt::TensorDesc::ELEMENT_TYPE_S32;
}

bool supportsFusedElementwiseDTypes(
    tt::FusedElementwiseOp::Node::Kind kind,
    tt::TensorDesc::ElementType input_type,
    tt::TensorDesc::ElementType output_type) {
    using Node = tt::FusedElementwiseOp::Node;
    if (kind == Node::KIND_CONVERT) {
        return supportsConvertElementType(input_type) &&
               supportsConvertElementType(output_type);
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
            return supportsFusedLeafElementType(input_type);
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
            return isFusedFloatElementType(input_type);
        default:
            return false;
    }
}

bool supportsFusedLeafElement(mlir::Value value) {
    auto element_type = staticValueElementType(value);
    return element_type && supportsFusedLeafElementType(*element_type);
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
        .Case<mlir::stablehlo::ConvertOp>([](auto) { return Node::KIND_CONVERT; })
        .Default([](auto) { return std::nullopt; });
}

bool isFusableElementwiseOp(mlir::Operation* op) {
    return op && op->getNumResults() == 1 &&
           fusedElementwiseKind(op).has_value() &&
           staticValueElementType(op->getResult(0)).has_value();
}

bool hasSupportedFusedElementwiseTypes(mlir::Operation* op) {
    auto kind = fusedElementwiseKind(op);
    if (!isFusableElementwiseOp(op) || !kind) {
        return false;
    }
    mlir::Value result = op->getResult(0);
    auto result_element = staticValueElementType(result);
    if (!result_element) {
        return false;
    }

    std::optional<tt::TensorDesc::ElementType> input_element;
    for (mlir::Value operand : op->getOperands()) {
        auto operand_element = staticValueElementType(operand);
        if (!operand_element) {
            return false;
        }
        if (input_element && *input_element != *operand_element) {
            return false;
        }
        input_element = *operand_element;
        std::string ignored;
        if (packedConstantValue(operand, ignored).has_value()) {
            continue;
        }
        if (!sameTensorShape(operand, result)) {
            return false;
        }
    }
    return input_element &&
           supportsFusedElementwiseDTypes(*kind, *input_element, *result_element);
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
    if (!is_root && !op->getResult(0).hasOneUse()) {
        return std::nullopt;
    }
    if (!sameTensorShape(op->getResult(0), root_value) ||
        !hasSupportedFusedElementwiseTypes(op)) {
        return std::nullopt;
    }

    auto kind = *fusedElementwiseKind(op);
    FusedElementwiseNodeDesc node;
    node.kind = kind;
    node.element_type = valueElementType(op->getResult(0));
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
    candidate_node_ids.try_emplace(op->getResult(0), node_id);
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
        if (!supportsFusedLeafElement(value)) {
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

    if (auto* defining_op = value.getDefiningOp();
        defining_op && isFusableElementwiseOp(defining_op)) {
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

    if (!supportsFusedLeafElement(value) || !sameTensorShape(value, root_value)) {
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
    if (!hasSupportedFusedElementwiseTypes(root)) {
        return std::nullopt;
    }

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

        if (auto convert_op = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(op)) {
            if (auto packed_value = packedConvertedConstantValue(convert_op, error)) {
                uint32_t output_id = 0;
                if (!addValueDesc(convert_op.getResult(), executable, value_ids, error, output_id)) {
                    return false;
                }
                addConstantOp(executable, output_id, *packed_value);
                continue;
            }
            if (!error.empty()) {
                return false;
            }
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

        if (auto composite_op = mlir::dyn_cast<mlir::stablehlo::CompositeOp>(op)) {
            if (composite_op.getName() != "chlo.top_k") {
                error = "unsupported stablehlo composite: " + composite_op.getName().str();
                return false;
            }
            if (composite_op->getNumOperands() != 1 || composite_op->getNumResults() != 2) {
                error = "top_k composite must have one operand and two results";
                return false;
            }
            auto attrs = composite_op.getCompositeAttributes();
            auto k = attrs ? attrs.getAs<mlir::IntegerAttr>("k") : nullptr;
            if (!k) {
                error = "top_k composite is missing k";
                return false;
            }
            if (!addTopKOp(
                    composite_op->getOperand(0),
                    composite_op->getResult(0),
                    composite_op->getResult(1),
                    k.getInt(),
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
    eraseDeadOps(*entry);

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
