// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/attention_output_gate/attention_output_gate.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {
namespace {

struct AttentionOutputGateParams {
    uint32_t head_dim = 0;
    uint32_t value_offset = 0;
    uint32_t gate_offset = 0;
    uint32_t gate_stride = 0;
    MemoryConfig memory_config;
};

struct AttentionOutputGateInputs {
    Tensor attention;
    Tensor gate;
};

struct AttentionOutputGateOperation {
    using operation_attributes_t = AttentionOutputGateParams;
    using tensor_args_t = AttentionOutputGateInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

void validate_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr, "{} must be on device", name);
    TT_FATAL(tensor.layout() == Layout::TILE && tensor.tensor_spec().tile() == Tile(), "{} must use 32x32 tiles", name);
    TT_FATAL(
        tensor.memory_config().buffer_type() == BufferType::DRAM &&
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{} must be interleaved in DRAM",
        name);
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{} must be BFLOAT16", name);
}

void AttentionOutputGateOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& in) {
    validate_tensor(in.attention, "attention");
    validate_tensor(in.gate, "gate");
    const auto& shape = in.attention.logical_shape();
    const auto& gate_shape = in.gate.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[0] == 1, "attention must be [1, T, H, W], got {}", shape);
    const uint32_t tokens = shape[1];
    const uint32_t heads = shape[2];
    TT_FATAL(tokens >= 1 && tokens <= TILE_HEIGHT, "attention must have 1 to 32 tokens, got {}", tokens);
    TT_FATAL(
        gate_shape.rank() == 2 && gate_shape[0] == tokens, "gate must be [T, N] for T = {}, got {}", tokens, gate_shape);
    for (auto [value, name] : {std::pair{params.head_dim, "head_dim"},
                               {params.value_offset, "value_offset"},
                               {params.gate_offset, "gate_offset"},
                               {params.gate_stride, "gate_stride"}}) {
        TT_FATAL(value % TILE_WIDTH == 0, "{} must be a multiple of 32, got {}", name, value);
    }
    TT_FATAL(params.head_dim > 0, "head_dim must be positive");
    TT_FATAL(
        params.value_offset + params.head_dim <= shape[3],
        "values {} .. {} must lie in the attention's {} columns",
        params.value_offset,
        params.value_offset + params.head_dim,
        shape[3]);
    TT_FATAL(
        (heads - 1) * params.gate_stride + params.gate_offset + params.head_dim <= gate_shape[1],
        "the gate heads must lie in the gate's {} columns",
        gate_shape[1]);
    TT_FATAL(
        params.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "attention_output_gate writes an interleaved output, got {}",
        params.memory_config);
}

AttentionOutputGateOperation::spec_return_value_t AttentionOutputGateOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const auto& shape = in.attention.logical_shape();
    return TensorSpec(
        Shape({shape[1], shape[2] * params.head_dim}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), params.memory_config));
}

AttentionOutputGateOperation::tensor_return_value_t AttentionOutputGateOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(params, in), in.attention.device());
}

ProgramDescriptor AttentionOutputGateOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& output_tensor) {
    const auto& attention = in.attention.mesh_tensor();
    const auto& gate = in.gate.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    auto* device = &attention.mutable_device();

    // Work units are output tiles: head u / Dt, columns (u % Dt) * 32 of it.
    const uint32_t tokens = attention.logical_shape()[1];
    const uint32_t heads = attention.logical_shape()[2];
    const uint32_t Dt = params.head_dim / TILE_WIDTH;
    const uint32_t units = heads * Dt;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min(units, static_cast<uint32_t>(grid.x * grid.y));
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const uint32_t base = units / num_cores;
    const uint32_t extra = units % num_cores;

    ProgramDescriptor program;
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    auto make_cb = [&](uint32_t index, uint32_t tiles) {
        program.cbs.push_back(CBDescriptor{
            .total_size = tiles * tile_bytes,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = tt::DataFormat::Float16_b,
                .page_size = tile_bytes}}}});
    };
    make_cb(0, 2);       // the unit's gate tile
    make_cb(1, 2);       // its attention rows, row t from token t's tile
    make_cb(2, tokens);  // the tokens' attention tiles, as read
    make_cb(3, 2);       // its output tile

    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_gate", 0},
        {"cb_values", 1},
        {"cb_attention", 2},
        {"cb_output", 3},
        {"tokens", tokens},
        {"Dt", Dt},
        // Tile rows per token of the attention, and tile columns of a row.
        {"attention_Ht", tt::div_up(heads, TILE_HEIGHT)},
        {"attention_Wt", attention.padded_shape()[-1] / TILE_WIDTH},
        {"value_offset_tiles", params.value_offset / TILE_WIDTH},
        {"gate_Wt", gate.padded_shape()[-1] / TILE_WIDTH},
        {"gate_offset_tiles", params.gate_offset / TILE_WIDTH},
        {"gate_stride_tiles", params.gate_stride / TILE_WIDTH},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/transformer/attention_output_gate/device/kernels/";

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(attention).append_to(reader_args);
    TensorAccessorArgs(gate).append_to(reader_args);
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_attention_output_gate.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = ReaderConfigDescriptor{}};
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(output).append_to(writer_args);
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_attention_output_gate.cpp",
        .core_ranges = cores,
        .compile_time_args = writer_args,
        .named_compile_time_args = common_args,
        .config = WriterConfigDescriptor{}};
    // As ttnn.sigmoid and ttnn.multiply run with FP32 accumulation.
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/attention_output_gate.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false}};

    const auto core_list = corerange_to_cores(cores, num_cores, /*row_wise=*/true);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t start = i * base + std::min(i, extra);
        const uint32_t count = base + (i < extra ? 1 : 0);
        reader.emplace_runtime_args(core_list[i], {start, count, attention, gate});
        writer.emplace_runtime_args(core_list[i], {start, count, output});
        compute.emplace_runtime_args(core_list[i], {count});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::transformer {

Tensor attention_output_gate(
    const Tensor& attention,
    const Tensor& gate,
    uint32_t head_dim,
    uint32_t value_offset,
    uint32_t gate_offset,
    uint32_t gate_stride,
    const tt::tt_metal::MemoryConfig& memory_config) {
    using Op = prim::AttentionOutputGateOperation;
    return device_operation::launch<Op>(
        {.head_dim = head_dim,
         .value_offset = value_offset,
         .gate_offset = gate_offset,
         .gate_stride = gate_stride,
         .memory_config = memory_config},
        {.attention = attention, .gate = gate});
}

}  // namespace ttnn::transformer
