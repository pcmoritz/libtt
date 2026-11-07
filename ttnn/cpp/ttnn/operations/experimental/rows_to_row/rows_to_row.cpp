// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/rows_to_row/rows_to_row.hpp"

#include <algorithm>
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

struct RowsToRowParams {
    ttnn::Shape shape;
    MemoryConfig memory_config;
};

struct RowsToRowInputs {
    Tensor input;
};

struct RowsToRowOperation {
    using operation_attributes_t = RowsToRowParams;
    using tensor_args_t = RowsToRowInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Input tiles a core reads before it writes their rows: a few L1 pages.
constexpr uint32_t batch_tiles = 8;

void RowsToRowOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& in) {
    TT_FATAL(
        ttnn::experimental::is_rows_to_row(in.input, params.shape),
        "rows_to_row takes a tiled interleaved [..., R, N] tensor of 4-byte elements to its rows in one row, got {} "
        "to {}",
        in.input.logical_shape(),
        params.shape);
    TT_FATAL(
        params.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "rows_to_row writes an interleaved output, got {}",
        params.memory_config);
}

RowsToRowOperation::spec_return_value_t RowsToRowOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return TensorSpec(
        params.shape, TensorLayout(in.input.dtype(), PageConfig(Layout::TILE), params.memory_config));
}

RowsToRowOperation::tensor_return_value_t RowsToRowOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(params, in), in.input.device());
}

ProgramDescriptor RowsToRowOperation::create_descriptor(
    const operation_attributes_t&, const tensor_args_t& in, tensor_return_value_t& output_tensor) {
    const auto& input = in.input.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    auto* device = &input.mutable_device();

    // Work units are input tiles: input tile c's row r is output tile r * Nt + c.
    const uint32_t rows = input.logical_shape()[-2];
    const uint32_t Nt = input.logical_shape()[-1] / TILE_WIDTH;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min(Nt, static_cast<uint32_t>(grid.x * grid.y));
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const uint32_t base = Nt / num_cores;
    const uint32_t extra = Nt % num_cores;

    ProgramDescriptor program;
    const auto format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_bytes = tt::tile_size(format);
    program.cbs.push_back(CBDescriptor{
        .total_size = batch_tiles * tile_bytes,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0, .data_format = format, .page_size = tile_bytes}}}});

    std::vector<uint32_t> compile_args = {rows, Nt, batch_tiles};
    TensorAccessorArgs(input).append_to(compile_args);
    TensorAccessorArgs(output).append_to(compile_args);
    KernelDescriptor kernel{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/rows_to_row/device/kernels/dataflow/rows_to_row.cpp",
        .core_ranges = cores,
        .compile_time_args = compile_args,
        .config = ReaderConfigDescriptor{}};
    const auto core_list = corerange_to_cores(cores, num_cores, /*row_wise=*/true);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t start = i * base + std::min(i, extra);
        const uint32_t count = base + (i < extra ? 1 : 0);
        kernel.emplace_runtime_args(core_list[i], {start, count, input, output});
    }
    program.kernels.push_back(std::move(kernel));
    return program;
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

bool is_rows_to_row(const ttnn::Tensor& input, const ttnn::Shape& shape) {
    const auto& in_shape = input.logical_shape();
    if (in_shape.rank() < 2 || shape.rank() < 1 || input.layout() != Layout::TILE ||
        input.tensor_spec().tile() != Tile() || input.element_size() != 4 ||
        input.storage_type() != StorageType::DEVICE ||
        input.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    const uint32_t rows = in_shape[-2];
    const uint32_t width = in_shape[-1];
    if (rows < 2 || rows > TILE_HEIGHT || width % TILE_WIDTH != 0 || in_shape.volume() != rows * width ||
        shape[-1] != rows * width || shape.volume() != rows * width) {
        return false;
    }
    return true;
}

ttnn::Tensor rows_to_row(
    const ttnn::Tensor& input, const ttnn::Shape& shape, const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using Op = prim::RowsToRowOperation;
    return device_operation::launch<Op>(
        {.shape = shape, .memory_config = memory_config.value_or(input.memory_config())}, {.input = input});
}

}  // namespace ttnn::experimental
