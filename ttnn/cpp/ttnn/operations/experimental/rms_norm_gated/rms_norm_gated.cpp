// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/rms_norm_gated/rms_norm_gated.hpp"

#include <algorithm>
#include <bit>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {
namespace {

struct RMSNormGatedParams {
    float epsilon = 1e-6f;
    uint32_t gate_offset = 0;
    DataType dtype = DataType::BFLOAT16;
};

struct RMSNormGatedInputs {
    Tensor input;
    Tensor gate;
    Tensor weight;
};

struct RMSNormGatedOperation {
    using operation_attributes_t = RMSNormGatedParams;
    using tensor_args_t = RMSNormGatedInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// A group's tiles all sit in DST at once, four FP32 tiles.
constexpr uint32_t max_group = 4 * TILE_WIDTH;

bool is_float(DataType dtype) { return dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32; }

void validate_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr, "{} must be on device", name);
    TT_FATAL(tensor.layout() == Layout::TILE && tensor.tensor_spec().tile() == Tile(), "{} must use 32x32 tiles", name);
    TT_FATAL(
        tensor.memory_config().buffer_type() == BufferType::DRAM &&
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{} must be interleaved in DRAM",
        name);
    TT_FATAL(is_float(tensor.dtype()), "{} must be BFLOAT16 or FLOAT32", name);
}

void RMSNormGatedOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    validate_tensor(in.input, "input");
    validate_tensor(in.gate, "gate");
    validate_tensor(in.weight, "weight");
    const auto& shape = in.input.logical_shape();
    const auto& gate_shape = in.gate.logical_shape();
    const uint32_t width = shape[-1];
    const uint32_t group = in.weight.logical_shape().volume();
    TT_FATAL(
        group % TILE_WIDTH == 0 && group <= max_group && in.weight.logical_shape()[-1] == group,
        "the weight must be one row of a multiple of 32 up to 128 elements, got {}",
        in.weight.logical_shape());
    TT_FATAL(width % group == 0, "the weight's {} elements must divide the last dimension {}", group, width);
    TT_FATAL(
        gate_shape.rank() == shape.rank() && in.gate.padded_shape().volume() / in.gate.padded_shape()[-1] ==
                                                 in.input.padded_shape().volume() / in.input.padded_shape()[-1],
        "gate shape {} must have the input's {} leading dimensions",
        gate_shape,
        shape);
    TT_FATAL(
        params.gate_offset % TILE_WIDTH == 0 && params.gate_offset + width <= gate_shape[-1],
        "gate columns {} .. {} must be tile aligned and within its {}",
        params.gate_offset,
        params.gate_offset + width,
        gate_shape[-1]);
    TT_FATAL(is_float(params.dtype), "the output must be BFLOAT16 or FLOAT32");
}

RMSNormGatedOperation::spec_return_value_t RMSNormGatedOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return TensorSpec(
        in.input.logical_shape(), TensorLayout(params.dtype, PageConfig(Layout::TILE), in.input.memory_config()));
}

RMSNormGatedOperation::tensor_return_value_t RMSNormGatedOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(params, in), in.input.device());
}

ProgramDescriptor RMSNormGatedOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& output_tensor) {
    const auto& input = in.input.mesh_tensor();
    const auto& gate = in.gate.mesh_tensor();
    const auto& weight = in.weight.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    auto* device = &input.mutable_device();

    // Work units are output tiles, row-major: tile row u / Nt, column u % Nt,
    // in group (u % Nt) / Gt.
    const uint32_t Nt = input.logical_shape()[-1] / TILE_WIDTH;
    const uint32_t Gt = weight.logical_shape().volume() / TILE_WIDTH;
    const uint32_t units = input.padded_shape().volume() / (TILE_HEIGHT * TILE_WIDTH);
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min(units, static_cast<uint32_t>(grid.x * grid.y));
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const uint32_t base = units / num_cores;
    const uint32_t extra = units % num_cores;

    ProgramDescriptor program;
    auto make_cb = [&](uint32_t index, uint32_t tiles, DataType dtype) {
        const auto format = datatype_to_dataformat_converter(dtype);
        const uint32_t page_size = tt::tile_size(format);
        program.cbs.push_back(CBDescriptor{
            .total_size = tiles * page_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = page_size}}}});
    };
    make_cb(0, Gt, input.dtype());  // the unit's input group
    make_cb(1, 1, gate.dtype());    // its gate tile
    make_cb(2, 1, weight.dtype());  // its weight tile
    make_cb(3, 1, params.dtype);    // its output tile

    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_input", 0},
        {"cb_gate", 1},
        {"cb_weight", 2},
        {"cb_output", 3},
        {"Nt", Nt},
        {"Gt", Gt},
        {"gate_Nt", gate.padded_shape()[-1] / TILE_WIDTH},
        {"gate_offset_tiles", params.gate_offset / TILE_WIDTH},
        {"inv_width", std::bit_cast<uint32_t>(1.0f / static_cast<float>(Gt * TILE_WIDTH))},
        {"eps", std::bit_cast<uint32_t>(params.epsilon)},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/rms_norm_gated/device/kernels/";

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(input).append_to(reader_args);
    TensorAccessorArgs(gate).append_to(reader_args);
    TensorAccessorArgs(weight).append_to(reader_args);
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_rms_norm_gated.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = ReaderConfigDescriptor{}};
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(output).append_to(writer_args);
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_rms_norm_gated.cpp",
        .core_ranges = cores,
        .compile_time_args = writer_args,
        .named_compile_time_args = common_args,
        .config = WriterConfigDescriptor{}};
    // FP32 operands unpack straight to DST, keeping all their bits.
    ComputeConfigDescriptor compute_config{
        .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (auto [cb, dtype] : {std::pair{0, input.dtype()}, {1, gate.dtype()}, {2, weight.dtype()}}) {
        if (dtype == DataType::FLOAT32) {
            compute_config.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/rms_norm_gated.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = compute_config};

    const auto core_list = corerange_to_cores(cores, num_cores, /*row_wise=*/true);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t start = i * base + std::min(i, extra);
        const uint32_t count = base + (i < extra ? 1 : 0);
        reader.emplace_runtime_args(core_list[i], {start, count, input, gate, weight});
        writer.emplace_runtime_args(core_list[i], {start, count, output});
        compute.emplace_runtime_args(core_list[i], {start, count});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

ttnn::Tensor rms_norm_gated(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& weight,
    float epsilon,
    uint32_t gate_offset,
    std::optional<tt::tt_metal::DataType> dtype) {
    using Op = prim::RMSNormGatedOperation;
    return device_operation::launch<Op>(
        {.epsilon = epsilon, .gate_offset = gate_offset, .dtype = dtype.value_or(input.dtype())},
        {.input = input, .gate = gate, .weight = weight});
}

}  // namespace ttnn::experimental
