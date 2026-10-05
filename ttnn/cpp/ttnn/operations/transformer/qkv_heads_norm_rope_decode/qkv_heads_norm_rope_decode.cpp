// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/qkv_heads_norm_rope_decode/qkv_heads_norm_rope_decode.hpp"

#include <algorithm>
#include <bit>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {
namespace {

struct QKVHeadsNormRopeDecodeParams {
    uint32_t num_q_heads = 0;
    uint32_t num_kv_heads = 0;
    float epsilon = 1e-6f;
    // 0 rotates the whole head.
    uint32_t rotary_dim = 0;
    bool gated_query = false;
};

struct QKVHeadsNormRopeDecodeInputs {
    Tensor qkv;
    Tensor norm_weight;
    Tensor cos;
    Tensor sin;
};

struct QKVHeadsNormRopeDecodeOperation {
    using operation_attributes_t = QKVHeadsNormRopeDecodeParams;
    using tensor_args_t = QKVHeadsNormRopeDecodeInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Heads of qkv: the q heads (each with its gate head when gated), k and v.
uint32_t input_heads(const QKVHeadsNormRopeDecodeParams& params) {
    return (params.gated_query ? 2 : 1) * params.num_q_heads + 2 * params.num_kv_heads;
}

uint32_t head_dim_of(const QKVHeadsNormRopeDecodeParams& params, const Tensor& qkv) {
    return qkv.logical_shape()[3] / input_heads(params);
}

uint32_t rotary_dim_of(const QKVHeadsNormRopeDecodeParams& params, const Tensor& qkv) {
    return params.rotary_dim ? params.rotary_dim : head_dim_of(params, qkv);
}

void validate_interleaved(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(
        tensor.memory_config().buffer_type() == BufferType::DRAM &&
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{} must be interleaved in DRAM", name);
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::FLOAT32,
        "{} must be BFLOAT16 or FLOAT32", name);
    TT_FATAL(tensor.tensor_spec().tile() == Tile(), "{} must use the default tile", name);
}

void QKVHeadsNormRopeDecodeOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& in) {
    validate_interleaved(in.qkv, "qkv");
    const auto& shape = in.qkv.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[0] == 1 && shape[1] == 1, "qkv must be [1, 1, B, W], got {}", shape);
    TT_FATAL(shape[2] > 0 && shape[2] <= TILE_HEIGHT, "qkv must have at most 32 users, got {}", shape[2]);
    TT_FATAL(
        params.num_q_heads > 0 && params.num_kv_heads > 0 && params.num_kv_heads <= params.num_q_heads &&
            params.num_kv_heads <= TILE_HEIGHT,
        "unsupported head counts {} and {}", params.num_q_heads, params.num_kv_heads);
    const uint32_t heads = input_heads(params);
    TT_FATAL(shape[3] % heads == 0, "width {} must split into {} heads", shape[3], heads);
    const uint32_t head_dim = head_dim_of(params, in.qkv);
    const uint32_t rotary_dim = rotary_dim_of(params, in.qkv);
    TT_FATAL(head_dim % TILE_WIDTH == 0, "head_dim {} must be a multiple of 32", head_dim);
    // rotate_half swaps whole tiles.
    TT_FATAL(
        rotary_dim % (2 * TILE_WIDTH) == 0 && rotary_dim <= head_dim,
        "rotary_dim {} must be a multiple of 64 of at most head_dim {}", rotary_dim, head_dim);
    auto validate_row = [&](const Tensor& tensor, const char* name) {
        validate_interleaved(tensor, name);
        const auto& row_shape = tensor.logical_shape();
        TT_FATAL(
            row_shape[-1] == rotary_dim && row_shape.volume() == rotary_dim,
            "{} shape {} must be one row of rotary_dim {}", name, row_shape, rotary_dim);
    };
    validate_interleaved(in.norm_weight, "norm_weight");
    TT_FATAL(
        in.norm_weight.logical_shape() == Shape({params.num_q_heads + params.num_kv_heads, head_dim}),
        "norm_weight shape {} must be [{}, {}], a row per q and k head",
        in.norm_weight.logical_shape(),
        params.num_q_heads + params.num_kv_heads,
        head_dim);
    validate_row(in.cos, "cos");
    validate_row(in.sin, "sin");
}

QKVHeadsNormRopeDecodeOperation::spec_return_value_t QKVHeadsNormRopeDecodeOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    // The specs of nlp_create_qkv_heads_decode for an interleaved input: one
    // user per core, row-wise from (0, 0), q, k and v on the same cores.
    const auto& shape = in.qkv.logical_shape();
    const uint32_t batch = shape[2];
    const uint32_t head_dim = head_dim_of(params, in.qkv);
    const CoreCoord grid = in.qkv.device()->compute_with_storage_grid_size();
    const CoreRangeSet all_cores{CoreRange{CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1}}};
    const CoreRangeSet cores =
        num_cores_to_corerangeset_in_subcoregrids(CoreCoord{0, 0}, batch, all_cores, /*row_wise=*/true);
    auto spec = [&](uint32_t heads) {
        const ShardSpec shard{cores, {(heads + TILE_HEIGHT - 1) / TILE_HEIGHT * TILE_HEIGHT, head_dim}};
        return TensorSpec(
            Shape({shape[0], batch, heads, head_dim}),
            TensorLayout(
                in.qkv.dtype(),
                PageConfig(Layout::TILE),
                MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, shard)));
    };
    return {spec(params.num_q_heads), spec(params.num_kv_heads), spec(params.num_kv_heads)};
}

QKVHeadsNormRopeDecodeOperation::tensor_return_value_t QKVHeadsNormRopeDecodeOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(params, in)) {
        outputs.push_back(create_device_tensor(spec, in.qkv.device()));
    }
    return outputs;
}

ProgramDescriptor QKVHeadsNormRopeDecodeOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& outputs) {
    const auto& qkv = in.qkv.mesh_tensor();
    const auto& weight = in.norm_weight.mesh_tensor();
    const auto& cos = in.cos.mesh_tensor();
    const auto& sin = in.sin.mesh_tensor();

    const uint32_t head_dim = head_dim_of(params, in.qkv);
    const uint32_t head_tiles = head_dim / TILE_WIDTH;
    const uint32_t rotary_tiles = rotary_dim_of(params, in.qkv) / TILE_WIDTH;
    // The heads read: q, k and v, not the gates.
    const uint32_t num_heads = params.num_q_heads + 2 * params.num_kv_heads;
    // The k heads are stacked below the q heads, a row each.
    const uint32_t row_tiles = (params.num_q_heads + params.num_kv_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t q_row_tiles = (params.num_q_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const CoreRangeSet cores = outputs.at(0).shard_spec().value().grid;
    const auto core_list = tt::tt_metal::corerange_to_cores(cores, std::nullopt, /*row_wise=*/true);

    const auto row_format = datatype_to_dataformat_converter(in.qkv.dtype());
    const uint32_t element_size = in.qkv.element_size();
    // One face row of a head, and the chunk read around it: the DRAM read
    // alignment, at least a face row.
    const uint32_t row_bytes = 16 * element_size;
    const uint32_t alignment = std::max(hal::get_dram_alignment(), row_bytes);

    ProgramDescriptor program;
    auto add_cb = [&](uint32_t index, uint32_t total_size, uint32_t page_size, tt::DataFormat format,
                      Buffer* buffer = nullptr) {
        program.cbs.push_back(CBDescriptor{
            .total_size = total_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = page_size}}},
            .buffer = buffer});
    };
    auto make_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format, Buffer* buffer = nullptr) {
        add_cb(index, tiles * tt::tile_size(format), tt::tile_size(format), format, buffer);
    };
    const auto fp32 = tt::DataFormat::Float32;
    make_cb(0, row_tiles * head_tiles, row_format);                                            // q and k heads
    make_cb(1, row_tiles * head_tiles, datatype_to_dataformat_converter(in.norm_weight.dtype()));  // weights
    make_cb(2, rotary_tiles, datatype_to_dataformat_converter(in.cos.dtype()));                  // cos
    make_cb(3, rotary_tiles, datatype_to_dataformat_converter(in.sin.dtype()));                  // sin
    make_cb(4, 1, fp32);                                                                       // reduce scaler
    make_cb(5, head_tiles, fp32);                                                              // squares
    make_cb(6, 1, fp32);                                                                       // rstd, a column
    make_cb(7, head_tiles, fp32);                                                              // normalized
    make_cb(8, head_tiles, fp32);                                                              // * weight
    make_cb(9, rotary_tiles, fp32);                                                            // * cos
    make_cb(10, rotary_tiles, fp32);                                                           // rotated * sin
    if (row_tiles > q_row_tiles) {
        make_cb(11, (row_tiles - q_row_tiles) * head_tiles, row_format);  // rows past the q output
    }
    // Raw scratch for one aligned chunk per face row read, plus the rounding
    // of its base up to the alignment; and one page the writer pushes once it
    // has placed its heads.
    const uint32_t scratch_bytes = (2 * num_heads * head_tiles + 1) * alignment;
    add_cb(12, scratch_bytes, scratch_bytes, tt::DataFormat::Float16_b);
    add_cb(13, 16, 16, tt::DataFormat::Float16_b);
    make_cb(16, q_row_tiles * head_tiles, row_format, outputs.at(0).buffer());  // q out
    make_cb(17, head_tiles, row_format, outputs.at(1).buffer());                // k out
    make_cb(18, head_tiles, row_format, outputs.at(2).buffer());                // v out

    std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_heads", 0},
        {"cb_weight", 1},
        {"cb_cos", 2},
        {"cb_sin", 3},
        {"cb_scaler", 4},
        {"cb_squares", 5},
        {"cb_rstd", 6},
        {"cb_normed", 7},
        {"cb_weighted", 8},
        {"cb_cos_part", 9},
        {"cb_sin_part", 10},
        {"cb_k_stage", 11},
        {"cb_scratch", 12},
        {"cb_placed", 13},
        {"cb_q_out", 16},
        {"cb_k_out", 17},
        {"cb_v_out", 18},
        {"num_q_heads", params.num_q_heads},
        {"num_kv_heads", params.num_kv_heads},
        {"head_tiles", head_tiles},
        {"rotary_tiles", rotary_tiles},
        // Tiles from one q head of qkv to the next.
        {"q_head_stride", (params.gated_query ? 2 : 1) * head_tiles},
        {"row_tiles", row_tiles},
        {"q_row_tiles", q_row_tiles},
        {"row_bytes", row_bytes},
        {"alignment", alignment},
        {"eps", std::bit_cast<uint32_t>(params.epsilon)},
        {"inv_head_dim", std::bit_cast<uint32_t>(1.0f / static_cast<float>(head_dim))},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/transformer/qkv_heads_norm_rope_decode/device/kernels/";

    std::vector<uint32_t> dataflow_args;
    for (const MeshTensor* tensor : {&qkv, &weight, &cos, &sin}) {
        TensorAccessorArgs(*tensor).append_to(dataflow_args);
    }
    auto dataflow_kernel = [&](uint32_t role, DataMovementProcessor processor, NOC noc) {
        auto args = common_args;
        args.emplace_back("role", role);
        return KernelDescriptor{
            .kernel_source = kernel_dir + "dataflow/dataflow_qkv_heads_norm_rope_decode.cpp",
            .core_ranges = cores,
            .compile_time_args = dataflow_args,
            .named_compile_time_args = std::move(args),
            .config = DataMovementConfigDescriptor{.processor = processor, .noc = noc}};
    };
    KernelDescriptor reader = dataflow_kernel(0, DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);
    KernelDescriptor writer = dataflow_kernel(1, DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/qkv_heads_norm_rope_decode.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false}};

    // User i is row i of the QKV tiles: in face 0 (and 1) for rows 0-15, in
    // face 2 (and 3) for rows 16-31.
    const uint32_t face_bytes = 256 * element_size;
    for (uint32_t i = 0; i < core_list.size(); ++i) {
        const uint32_t row_offset = (i % 16) * row_bytes + (i / 16) * 2 * face_bytes;
        reader.emplace_runtime_args(core_list[i], {qkv, weight, cos, sin, row_offset});
        writer.emplace_runtime_args(core_list[i], {qkv, weight, cos, sin, row_offset});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::transformer {

std::vector<Tensor> nlp_create_qkv_heads_decode_norm_rope(
    const Tensor& qkv,
    const Tensor& norm_weight,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    float epsilon,
    uint32_t rotary_dim,
    bool gated_query) {
    using Op = prim::QKVHeadsNormRopeDecodeOperation;
    return device_operation::launch<Op>(
        {.num_q_heads = num_q_heads,
         .num_kv_heads = num_kv_heads,
         .epsilon = epsilon,
         .rotary_dim = rotary_dim,
         .gated_query = gated_query},
        {.qkv = qkv, .norm_weight = norm_weight, .cos = cos, .sin = sin});
}

}  // namespace ttnn::transformer
