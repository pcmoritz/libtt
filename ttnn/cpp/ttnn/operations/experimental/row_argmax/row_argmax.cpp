// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/row_argmax/row_argmax.hpp"

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

struct RowArgmaxParams {
    bool keep_dim = false;
    std::optional<float> floor;
    MemoryConfig memory_config;
};

struct RowArgmaxInputs {
    Tensor input;
};

struct RowArgmaxOperation {
    using operation_attributes_t = RowArgmaxParams;
    using tensor_args_t = RowArgmaxInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Input tiles a core reads before it scans them: a few L1 pages.
constexpr uint32_t batch_tiles = 8;

// The order-preserving unsigned key of a float's bits: negative values
// reversed below the positive ones.
uint32_t key_of(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

void RowArgmaxOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& in) {
    TT_FATAL(
        ttnn::experimental::is_row_argmax(in.input, -1),
        "row_argmax takes a tiled interleaved [..., R, N] BFLOAT16 or FLOAT32 tensor, R <= 32, got {}",
        in.input.logical_shape());
    TT_FATAL(
        params.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "row_argmax writes an interleaved output, got {}",
        params.memory_config);
}

RowArgmaxOperation::spec_return_value_t RowArgmaxOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const auto& shape = in.input.logical_shape();
    ttsl::SmallVector<uint32_t> dims(shape.cbegin(), shape.cend() - 1);
    if (params.keep_dim) {
        dims.push_back(1);
    }
    return TensorSpec(
        Shape(dims), TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), params.memory_config));
}

RowArgmaxOperation::tensor_return_value_t RowArgmaxOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(params, in), in.input.device());
}

ProgramDescriptor RowArgmaxOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& output_tensor) {
    const auto& input = in.input.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    auto* device = &input.mutable_device();

    // Work units are tiles of the row; every core scans all R rows of its tiles
    // and sends their best (key, index) to the first core, which merges them.
    const uint32_t rows = input.logical_shape()[-2];
    const uint32_t Nt = input.logical_shape()[-1] / TILE_WIDTH;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min(Nt, static_cast<uint32_t>(grid.x * grid.y));
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const uint32_t base = Nt / num_cores;
    const uint32_t extra = Nt % num_cores;
    const auto core_list = corerange_to_cores(cores, num_cores, /*row_wise=*/true);
    const CoreCoord leader = device->worker_core_from_logical_core(core_list[0]);

    ProgramDescriptor program;
    const auto format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_bytes = tt::tile_size(format);
    program.cbs.push_back(CBDescriptor{
        .total_size = batch_tiles * tile_bytes,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 0, .data_format = format, .page_size = tile_bytes}}}});
    // Every core's (key, index) pairs of the R rows, on the first core.
    const uint32_t slot_bytes = rows * 2 * sizeof(uint32_t);
    const uint32_t slots_bytes = (num_cores * slot_bytes + 63) / 64 * 64;
    program.cbs.push_back(CBDescriptor{
        .total_size = slots_bytes,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = 1, .data_format = tt::DataFormat::UInt32, .page_size = slots_bytes}}}});
    program.semaphores.push_back(SemaphoreDescriptor{.id = 0, .core_ranges = cores, .initial_value = 0});

    std::vector<uint32_t> compile_args = {
        rows,
        static_cast<uint32_t>(input.element_size()),
        batch_tiles,
        num_cores,
        params.floor.has_value(),
        params.floor.has_value() ? key_of(*params.floor) : 0u};
    TensorAccessorArgs(input).append_to(compile_args);
    TensorAccessorArgs(output).append_to(compile_args);
    KernelDescriptor kernel{
        .kernel_source = "ttnn/cpp/ttnn/operations/experimental/row_argmax/device/kernels/dataflow/row_argmax.cpp",
        .core_ranges = cores,
        .compile_time_args = compile_args,
        .config = ReaderConfigDescriptor{}};
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t start = i * base + std::min(i, extra);
        const uint32_t count = base + (i < extra ? 1 : 0);
        kernel.emplace_runtime_args(
            core_list[i],
            {start, count, i, static_cast<uint32_t>(leader.x), static_cast<uint32_t>(leader.y), input, output});
    }
    program.kernels.push_back(std::move(kernel));
    return program;
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

bool is_row_argmax(const ttnn::Tensor& input, int dim) {
    const auto& shape = input.logical_shape();
    if (shape.rank() < 2 || (dim != -1 && dim != static_cast<int>(shape.rank()) - 1) ||
        input.layout() != Layout::TILE || input.tensor_spec().tile() != Tile() ||
        (input.dtype() != DataType::BFLOAT16 && input.dtype() != DataType::FLOAT32) ||
        input.storage_type() != StorageType::DEVICE ||
        input.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return false;
    }
    const uint32_t rows = shape[-2];
    const uint32_t width = shape[-1];
    return rows >= 1 && rows <= TILE_HEIGHT && width % TILE_WIDTH == 0 && shape.volume() == rows * width;
}

ttnn::Tensor row_argmax(
    const ttnn::Tensor& input,
    bool keep_dim,
    std::optional<float> floor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    using Op = prim::RowArgmaxOperation;
    return device_operation::launch<Op>(
        {.keep_dim = keep_dim, .floor = floor, .memory_config = memory_config.value_or(DRAM_MEMORY_CONFIG)},
        {.input = input});
}

}  // namespace ttnn::experimental
