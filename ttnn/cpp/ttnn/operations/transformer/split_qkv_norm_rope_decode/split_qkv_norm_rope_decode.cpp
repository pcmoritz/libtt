// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/split_qkv_norm_rope_decode/split_qkv_norm_rope_decode.hpp"

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

// Heads per unit of work: one core's rows of a tile, two 256-byte runs of
// each of its tiles (rows 0-7, 8-15, 16-23 or 24-31).
constexpr uint32_t kGroup = 8;

// What a unit computes: rms_norm and rotary_embedding for q or k heads; v
// heads are only moved.
enum UnitKind : uint32_t { kQ = 0, kK = 1, kV = 2 };

struct SplitQKVNormRopeDecodeParams {
    uint32_t num_q_heads = 0;
    uint32_t num_kv_heads = 0;
    float epsilon = 1e-6f;
    uint32_t q_width = 0;
    MemoryConfig q_memory_config;
    MemoryConfig kv_memory_config;
};

struct SplitQKVNormRopeDecodeInputs {
    Tensor qkv;
    Tensor q_weight;
    Tensor k_weight;
    Tensor cos;
    Tensor sin;
};

struct SplitQKVNormRopeDecodeOperation {
    using operation_attributes_t = SplitQKVNormRopeDecodeParams;
    using tensor_args_t = SplitQKVNormRopeDecodeInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

uint32_t head_dim_of(const SplitQKVNormRopeDecodeParams& params, const Tensor& qkv) {
    return qkv.logical_shape()[-1] / (params.num_q_heads + 2 * params.num_kv_heads);
}

void validate_tiled(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{} must be BFLOAT16", name);
    TT_FATAL(tensor.tensor_spec().tile() == Tile(), "{} must use the default tile", name);
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "{} must be interleaved", name);
}

void SplitQKVNormRopeDecodeOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& in) {
    validate_tiled(in.qkv, "qkv");
    const auto& shape = in.qkv.logical_shape();
    TT_FATAL(shape.rank() >= 2 && shape.volume() == shape[-2] * shape[-1], "qkv must be [B, W], got {}", shape);
    TT_FATAL(shape[-2] > 0 && shape[-2] <= TILE_HEIGHT, "qkv must have at most 32 rows, got {}", shape[-2]);
    TT_FATAL(
        params.num_q_heads > 0 && params.num_kv_heads > 0 && params.num_kv_heads <= TILE_HEIGHT,
        "unsupported head counts {} and {}", params.num_q_heads, params.num_kv_heads);
    const uint32_t heads = params.num_q_heads + 2 * params.num_kv_heads;
    TT_FATAL(shape[-1] % heads == 0, "width {} must split into {} heads", shape[-1], heads);
    const uint32_t head_dim = head_dim_of(params, in.qkv);
    // rotate_half swaps whole tiles.
    TT_FATAL(head_dim % (2 * TILE_WIDTH) == 0, "head_dim {} must be a multiple of 64", head_dim);
    TT_FATAL(
        params.q_width >= head_dim && params.q_width % TILE_WIDTH == 0,
        "q_width {} must be a multiple of 32 of at least head_dim {}", params.q_width, head_dim);
    for (const auto& [tensor, name] : {std::pair{&in.q_weight, "q_weight"}, std::pair{&in.k_weight, "k_weight"}}) {
        validate_tiled(*tensor, name);
        TT_FATAL(
            tensor->logical_shape()[-1] == head_dim && tensor->logical_shape().volume() == head_dim,
            "{} shape {} must be one row of head_dim {}", name, tensor->logical_shape(), head_dim);
    }
    for (const auto& [tensor, name] : {std::pair{&in.cos, "cos"}, std::pair{&in.sin, "sin"}}) {
        validate_tiled(*tensor, name);
        const auto& table = tensor->logical_shape();
        TT_FATAL(
            table.rank() == 4 && table[0] == 1 && table[2] == 1 && table[3] == head_dim &&
                (table[1] == 1 || table[1] == shape[-2]),
            "{} shape {} must be [1, 1 or {}, 1, {}]", name, table, shape[-2], head_dim);
    }
    TT_FATAL(in.cos.logical_shape() == in.sin.logical_shape(), "cos and sin must have one shape");
}

SplitQKVNormRopeDecodeOperation::spec_return_value_t SplitQKVNormRopeDecodeOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const uint32_t tokens = in.qkv.logical_shape()[-2];
    const uint32_t head_dim = head_dim_of(params, in.qkv);
    auto spec = [&](const Shape& shape, const MemoryConfig& memory_config) {
        return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config));
    };
    spec_return_value_t specs{spec(Shape({1, tokens, params.num_q_heads, params.q_width}), params.q_memory_config)};
    for (uint32_t b = 0; b < tokens; ++b) {
        specs.push_back(spec(Shape({1, 1, params.num_kv_heads, 2 * head_dim}), params.kv_memory_config));
    }
    return specs;
}

SplitQKVNormRopeDecodeOperation::tensor_return_value_t SplitQKVNormRopeDecodeOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(params, in)) {
        outputs.push_back(create_device_tensor(spec, in.qkv.device()));
    }
    return outputs;
}

// One core's work on up to kGroup heads of one token: its rows of the q
// output or of the token's kv output, and the zero rows and tiles of that
// tile row nobody else writes.
struct Unit {
    UnitKind kind;
    uint32_t token;
    uint32_t source_head;  // head index in the QKV row
    uint32_t heads;
    uint32_t first_tile;
    uint32_t first_row;
    uint32_t zero_rows_from;  // rows from here to 32 of its tiles are zero
    uint32_t zero_tiles;      // whole zero tiles after its tiles
};

ProgramDescriptor SplitQKVNormRopeDecodeOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& outputs) {
    const auto& qkv = in.qkv.mesh_tensor();
    const auto& q_weight = in.q_weight.mesh_tensor();
    const auto& k_weight = in.k_weight.mesh_tensor();
    const auto& cos = in.cos.mesh_tensor();
    const auto& sin = in.sin.mesh_tensor();

    const uint32_t tokens = in.qkv.logical_shape()[-2];
    const uint32_t num_q_heads = params.num_q_heads;
    const uint32_t num_kv_heads = params.num_kv_heads;
    const uint32_t head_dim = head_dim_of(params, in.qkv);
    const uint32_t head_tiles = head_dim / TILE_WIDTH;
    const uint32_t q_row_tiles = (num_q_heads + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t q_width_tiles = params.q_width / TILE_WIDTH;
    auto round_up = [](uint32_t value, uint32_t step) { return (value + step - 1) / step * step; };

    std::vector<Unit> units;
    for (uint32_t b = 0; b < tokens; ++b) {
        for (uint32_t h = 0; h < num_q_heads; h += kGroup) {
            const uint32_t row = h % TILE_HEIGHT;
            const uint32_t rows = std::min(TILE_HEIGHT, num_q_heads - h / TILE_HEIGHT * TILE_HEIGHT);
            units.push_back(Unit{
                .kind = kQ,
                .token = b,
                .source_head = h,
                .heads = std::min(kGroup, num_q_heads - h),
                .first_tile = (b * q_row_tiles + h / TILE_HEIGHT) * q_width_tiles,
                .first_row = row,
                .zero_rows_from = row == 0 ? round_up(rows, kGroup) : TILE_HEIGHT,
                .zero_tiles = row == 0 ? q_width_tiles - head_tiles : 0});
        }
        for (uint32_t h = 0; h < num_kv_heads; h += kGroup) {
            for (UnitKind kind : {kK, kV}) {
                units.push_back(Unit{
                    .kind = kind,
                    .token = b,
                    .source_head = num_q_heads + (kind == kV ? num_kv_heads : 0) + h,
                    .heads = std::min(kGroup, num_kv_heads - h),
                    .first_tile = kind == kV ? head_tiles : 0,
                    .first_row = h,
                    .zero_rows_from = h == 0 ? round_up(num_kv_heads, kGroup) : TILE_HEIGHT,
                    .zero_tiles = 0});
            }
        }
    }
    const CoreCoord grid = in.qkv.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint32_t>(units.size(), grid.x * grid.y);
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const auto core_list = corerange_to_cores(cores, std::nullopt, /*row_wise=*/true);

    const auto bf16 = tt::DataFormat::Float16_b;
    const auto fp32 = tt::DataFormat::Float32;
    const uint32_t tile_bytes = tt::tile_size(bf16);
    // A face row (16 BF16 values) of a QKV tile and the same row of the next
    // face, 512 bytes on, read as one aligned window.
    const uint32_t row_bytes = 16 * sizeof(uint16_t);
    const uint32_t alignment = std::max<uint32_t>(hal::get_dram_alignment(), row_bytes);
    const uint32_t window = round_up(tile_bytes / 4 + row_bytes + alignment - row_bytes, alignment);

    ProgramDescriptor program;
    auto add_cb = [&](uint32_t index, uint32_t total_size, uint32_t page_size, tt::DataFormat format) {
        program.cbs.push_back(CBDescriptor{
            .total_size = total_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = page_size}}}});
    };
    auto make_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format) {
        add_cb(index, tiles * tt::tile_size(format), tt::tile_size(format), format);
    };
    // The formats of rms_norm's and rotary_embedding's buffers, which set how
    // each step rounds.
    make_cb(0, head_tiles, bf16);   // heads, rms_norm's input
    make_cb(1, head_tiles, bf16);   // norm weight
    make_cb(2, head_tiles, bf16);   // cos
    make_cb(3, head_tiles, bf16);   // sin
    make_cb(4, 1, bf16);            // reduce scaler
    make_cb(5, 1, bf16);            // epsilon
    make_cb(6, head_tiles, bf16);   // v heads
    make_cb(7, head_tiles, fp32);   // x^2
    make_cb(8, 1, fp32);            // mean of x^2
    make_cb(9, 1, fp32);            // 1 / sqrt(mean + eps)
    make_cb(10, head_tiles, fp32);  // x / rms
    make_cb(11, head_tiles, bf16);  // rms_norm's output, rotary's input
    make_cb(12, 1, bf16);           // x of the other half * sin
    make_cb(13, 1, bf16);           // x * cos
    make_cb(14, head_tiles, bf16);  // rotary's output
    make_cb(15, 1, bf16);           // zeros
    const uint32_t scratch_bytes = kGroup * head_tiles * window + alignment;
    add_cb(16, scratch_bytes, scratch_bytes, bf16);

    std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_x", 0},
        {"cb_weight", 1},
        {"cb_cos", 2},
        {"cb_sin", 3},
        {"cb_scaler", 4},
        {"cb_eps", 5},
        {"cb_v", 6},
        {"cb_sq", 7},
        {"cb_mean", 8},
        {"cb_rstd", 9},
        {"cb_scaled", 10},
        {"cb_normed", 11},
        {"cb_sin_part", 12},
        {"cb_cos_part", 13},
        {"cb_out", 14},
        {"cb_zero", 15},
        {"cb_scratch", 16},
        {"head_tiles", head_tiles},
        {"group", kGroup},
        {"row_bytes", row_bytes},
        {"alignment", alignment},
        {"window", window},
        {"qkv_width_tiles", static_cast<uint32_t>(in.qkv.logical_shape()[-1]) / TILE_WIDTH},
        {"cos_per_token", in.cos.logical_shape()[1] > 1 ? 1u : 0u},
        // rms_norm's epsilon tile holds the upper half of its FP32 bits.
        {"eps_bf16", std::bit_cast<uint32_t>(params.epsilon) >> 16},
        {"inv_width", std::bit_cast<uint32_t>(1.0f / static_cast<float>(head_dim))},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/transformer/split_qkv_norm_rope_decode/device/kernels/";

    std::vector<uint32_t> dataflow_args;
    for (const MeshTensor* tensor : {&qkv, &q_weight, &k_weight, &cos, &sin}) {
        TensorAccessorArgs(*tensor).append_to(dataflow_args);
    }
    TensorAccessorArgs(outputs.at(0).mesh_tensor()).append_to(dataflow_args);
    TensorAccessorArgs(outputs.at(1).mesh_tensor()).append_to(dataflow_args);
    auto dataflow_kernel = [&](const char* name, DataMovementProcessor processor, NOC noc) {
        return KernelDescriptor{
            .kernel_source = kernel_dir + "dataflow/" + name,
            .core_ranges = cores,
            .compile_time_args = dataflow_args,
            .named_compile_time_args = common_args,
            .config = DataMovementConfigDescriptor{.processor = processor, .noc = noc}};
    };
    KernelDescriptor reader = dataflow_kernel(
        "reader_split_qkv_norm_rope_decode.cpp", DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);
    KernelDescriptor writer = dataflow_kernel(
        "writer_split_qkv_norm_rope_decode.cpp", DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/split_qkv_norm_rope_decode.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false}};

    for (uint32_t c = 0; c < num_cores; ++c) {
        KernelDescriptor::RTArgList args;
        for (const MeshTensor* tensor : {&qkv, &q_weight, &k_weight, &cos, &sin}) {
            args.push_back(*tensor);
        }
        uint32_t norm_units = 0;  // the compute kernel's: q and k units
        for (uint32_t u = c; u < units.size(); u += num_cores) {
            const Unit& unit = units[u];
            norm_units += unit.kind != kV;
            args.push_back(static_cast<uint32_t>(unit.kind));
            args.push_back(unit.token);
            args.push_back(unit.source_head);
            args.push_back(unit.heads);
            args.push_back(outputs.at(unit.kind == kQ ? 0 : 1 + unit.token).mesh_tensor());
            args.push_back(unit.first_tile);
            args.push_back(unit.first_row);
            args.push_back(unit.zero_rows_from);
            args.push_back(unit.zero_tiles);
        }
        args.push_back(0xFFFFFFFFu);  // end of the units
        reader.emplace_runtime_args(core_list[c], args);
        writer.emplace_runtime_args(core_list[c], args);
        compute.runtime_args.emplace_back(core_list[c], std::vector<uint32_t>{norm_units});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::transformer {

std::vector<Tensor> split_qkv_norm_rope_decode(
    const Tensor& qkv,
    const Tensor& q_weight,
    const Tensor& k_weight,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    float epsilon,
    uint32_t q_width,
    const tt::tt_metal::MemoryConfig& q_memory_config,
    const tt::tt_metal::MemoryConfig& kv_memory_config) {
    using Op = prim::SplitQKVNormRopeDecodeOperation;
    return device_operation::launch<Op>(
        {.num_q_heads = num_q_heads,
         .num_kv_heads = num_kv_heads,
         .epsilon = epsilon,
         .q_width = q_width,
         .q_memory_config = q_memory_config,
         .kv_memory_config = kv_memory_config},
        {.qkv = qkv, .q_weight = q_weight, .k_weight = k_weight, .cos = cos, .sin = sin});
}

}  // namespace ttnn::transformer
