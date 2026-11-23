// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/l2_clip/l2_clip.hpp"

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

struct L2ClipParams {
    uint32_t group = 128;
    float threshold = 1.0f;
    float epsilon = 1e-6f;
    uint32_t offset = 0;
    uint32_t width = 0;
    DataType dtype = DataType::BFLOAT16;
};

struct L2ClipInputs {
    Tensor input;
    std::optional<Tensor> scale;
    std::optional<Tensor> row_scale;
};

struct L2ClipOperation {
    using operation_attributes_t = L2ClipParams;
    using tensor_args_t = L2ClipInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

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

void L2ClipOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    validate_tensor(in.input, "input");
    const uint32_t columns = in.input.logical_shape()[-1];
    TT_FATAL(
        params.group == 16 || (params.group % TILE_WIDTH == 0 && params.group <= 4 * TILE_WIDTH),
        "the group must be 16 or a multiple of 32 up to 128, got {}",
        params.group);
    TT_FATAL(
        params.offset % TILE_WIDTH == 0 && params.width % TILE_WIDTH == 0 && params.width > 0 &&
            params.offset + params.width <= columns && params.width % params.group == 0,
        "columns {} .. {} must be tile aligned, within the input's {} and whole groups of {}",
        params.offset,
        params.offset + params.width,
        columns,
        params.group);
    if (in.scale.has_value()) {
        validate_tensor(*in.scale, "scale");
        TT_FATAL(
            in.scale->logical_shape()[-1] == params.width && in.scale->logical_shape().volume() == params.width,
            "the scale must be one row of {} elements, got {}",
            params.width,
            in.scale->logical_shape());
    }
    if (in.row_scale.has_value()) {
        validate_tensor(*in.row_scale, "row_scale");
        const uint32_t rows = in.input.logical_shape().volume() / columns;
        TT_FATAL(
            in.row_scale->logical_shape()[-1] == 1 && in.row_scale->logical_shape().volume() == rows &&
                in.row_scale->padded_shape().volume() / TILE_WIDTH ==
                    in.input.padded_shape().volume() / in.input.padded_shape()[-1],
            "the row scale must be [{}, 1], got {}",
            rows,
            in.row_scale->logical_shape());
    }
    TT_FATAL(is_float(params.dtype), "the output must be BFLOAT16 or FLOAT32");
}

L2ClipOperation::spec_return_value_t L2ClipOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    auto shape = in.input.logical_shape();
    shape[-1] = params.width;
    return TensorSpec(shape, TensorLayout(params.dtype, PageConfig(Layout::TILE), in.input.memory_config()));
}

L2ClipOperation::tensor_return_value_t L2ClipOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(params, in), in.input.device());
}

ProgramDescriptor L2ClipOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& output_tensor) {
    const auto& input = in.input.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    // Absent operands stand in as the input in the kernels' arguments.
    const auto& scale = in.scale.has_value() ? in.scale->mesh_tensor() : input;
    const auto& row_scale = in.row_scale.has_value() ? in.row_scale->mesh_tensor() : input;
    auto* device = &input.mutable_device();

    // Work units are output tiles, row-major: tile row u / Nt, column u % Nt.
    const uint32_t Nt = params.width / TILE_WIDTH;
    const uint32_t in_Nt = input.padded_shape()[-1] / TILE_WIDTH;
    // A group's tiles (one for groups of 16 columns, two to a tile).
    const uint32_t Gt = std::max(1u, params.group / TILE_WIDTH);
    const uint32_t row_tiles = input.padded_shape().volume() / (input.padded_shape()[-1] * TILE_HEIGHT);
    const uint32_t units = row_tiles * Nt;
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
    make_cb(0, Gt, input.dtype());     // the unit's group of input tiles
    make_cb(1, Gt, scale.dtype());     // their scale tiles
    make_cb(2, 1, row_scale.dtype());  // the unit's row scale tile
    make_cb(3, 1, params.dtype);       // its output tile

    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_input", 0},
        {"cb_scale", 1},
        {"cb_row_scale", 2},
        {"cb_output", 3},
        {"Nt", Nt},
        {"in_Nt", in_Nt},
        {"offset_tiles", params.offset / TILE_WIDTH},
        {"Gt", Gt},
        {"half_tile_groups", params.group == 16 ? 1u : 0u},
        {"has_scale", in.scale.has_value() ? 1u : 0u},
        {"has_row_scale", in.row_scale.has_value() ? 1u : 0u},
        {"threshold", std::bit_cast<uint32_t>(params.threshold)},
        {"eps_squared", std::bit_cast<uint32_t>(params.epsilon * params.epsilon)},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/l2_clip/device/kernels/";

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(input).append_to(reader_args);
    TensorAccessorArgs(scale).append_to(reader_args);
    TensorAccessorArgs(row_scale).append_to(reader_args);
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_l2_clip.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = ReaderConfigDescriptor{}};
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(output).append_to(writer_args);
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_l2_clip.cpp",
        .core_ranges = cores,
        .compile_time_args = writer_args,
        .named_compile_time_args = common_args,
        .config = WriterConfigDescriptor{}};
    // FP32 operands unpack straight to DST, keeping all their bits.
    ComputeConfigDescriptor compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .math_approx_mode = false};
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (auto [cb, dtype] : {std::pair{0, input.dtype()}, {1, scale.dtype()}, {2, row_scale.dtype()}}) {
        if (dtype == DataType::FLOAT32) {
            compute_config.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/l2_clip.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = compute_config};

    const auto core_list = corerange_to_cores(cores, num_cores, /*row_wise=*/true);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t start = i * base + std::min(i, extra);
        const uint32_t count = base + (i < extra ? 1 : 0);
        reader.emplace_runtime_args(core_list[i], {start, count, input, scale, row_scale});
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

ttnn::Tensor l2_clip(
    const ttnn::Tensor& input,
    uint32_t group,
    float threshold,
    float epsilon,
    uint32_t offset,
    uint32_t width,
    const std::optional<ttnn::Tensor>& scale,
    const std::optional<ttnn::Tensor>& row_scale,
    std::optional<tt::tt_metal::DataType> dtype) {
    using Op = prim::L2ClipOperation;
    const uint32_t columns = input.logical_shape()[-1];
    return device_operation::launch<Op>(
        {.group = group,
         .threshold = threshold,
         .epsilon = epsilon,
         .offset = offset,
         .width = width != 0 ? width : columns - offset,
         .dtype = dtype.value_or(input.dtype())},
        {.input = input, .scale = scale, .row_scale = row_scale});
}

}  // namespace ttnn::experimental
