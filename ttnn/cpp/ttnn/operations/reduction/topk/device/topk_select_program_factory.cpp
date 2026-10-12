// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"
#include "ttnn/operations/reduction/topk/device/topk_select.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt_stl/assert.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {

constexpr uint32_t max_k = 64;
// Narrower segments leave the scan shorter than the merge and the dispatch.
constexpr uint32_t min_segment_tiles = 16;
// L1 for one core's face rows of input and for the candidates a group's
// first-segment core gathers. Unlike the sort networks' factories this path
// does not check the free L1, so its buffers stay small.
constexpr uint32_t max_input_bytes = 128 * 1024;
constexpr uint32_t max_candidate_bytes = 64 * 1024;
// A face row of 16 elements is 64 bytes in FP32. A BF16 one is 32 bytes, read
// as the 64-byte-aligned pair of rows it belongs to.
constexpr uint32_t face_row_read_bytes = 64;
constexpr uint32_t candidate_bytes = 8;  // a key and an index

// Candidates per list: an even number keeps the lists 16-byte aligned for the
// core-to-core writes.
uint32_t list_stride(uint32_t k) { return (k + 1) & ~1u; }

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

}  // namespace

std::optional<TopKSelectPlan> topk_select_plan(
    const Tensor& input,
    uint32_t k,
    bool custom_indices,
    bool preallocated_outputs,
    const MemoryConfig& output_memory_config,
    const CoreRangeSet& grid) {
    if (custom_indices || preallocated_outputs || output_memory_config.is_sharded() ||
        input.storage_type() != StorageType::DEVICE || input.layout() != Layout::TILE ||
        input.tensor_spec().tile() != Tile() || input.memory_config().is_sharded() ||
        (input.dtype() != DataType::FLOAT32 && input.dtype() != DataType::BFLOAT16)) {
        return std::nullopt;
    }
    const auto& logical = input.logical_shape();
    const auto& padded = input.padded_shape();
    if (logical.rank() != 4 || k < 1 || k > max_k || k > logical[3]) {
        return std::nullopt;
    }
    TopKSelectPlan plan;
    plan.k = k;
    plan.height = logical[2];
    plan.width = logical[3];
    plan.Ht = padded[2] / 32;
    plan.Wt = padded[3] / 32;
    plan.tile_rows = logical[0] * logical[1] * plan.Ht;
    const uint32_t cores = grid.num_cores();
    if (plan.tile_rows == 0 || plan.tile_rows > cores) {
        return std::nullopt;
    }
    // What the sort networks do well: many tile rows of a narrow width, where
    // each core sorts a whole tile row 32 rows at a time, and a narrow k above
    // a tile, where an insertion shifts lists of up to 64 on a scalar core.
    if (plan.Wt < 32 && (plan.tile_rows > 2 || k > 32)) {
        return std::nullopt;
    }
    const uint32_t per_tile_row = cores / plan.tile_rows;
    const uint32_t rows = plan.rows();
    plan.segments = std::min(per_tile_row, div_up(plan.Wt, min_segment_tiles));
    plan.segment_tiles = div_up(plan.Wt, plan.segments);
    plan.segments = div_up(plan.Wt, plan.segment_tiles);
    plan.row_groups = std::max(1u, std::min(rows, per_tile_row / plan.segments));
    plan.rows_per_core = div_up(rows, plan.row_groups);
    // A group's first-segment core gathers k candidates per row and segment.
    const uint32_t max_segments = max_candidate_bytes / (plan.rows_per_core * list_stride(k) * candidate_bytes);
    if (plan.segments > max_segments) {
        plan.segment_tiles = div_up(plan.Wt, std::max(1u, max_segments));
        plan.segments = div_up(plan.Wt, plan.segment_tiles);
    }
    if (plan.rows_per_core * plan.segment_tiles * 2 * face_row_read_bytes > max_input_bytes) {
        return std::nullopt;
    }
    return plan;
}

// A tile row's cores take a column segment and the rows g, g + row_groups, ...
// each. A row's best k of each segment goes to the core of the first segment
// in its group, which merges them; the finished rows go into the output tiles
// of the tile row's first core, which writes them out. There is no compute
// kernel.
ProgramDescriptor TopKDeviceOperation::TopKSelectProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& outputs) {
    const auto plan = topk_select_plan(
        tensor_args.input,
        args.k,
        tensor_args.indices.has_value(),
        tensor_args.preallocated_outputs.has_value(),
        args.output_memory_config,
        args.sub_core_grids);
    TT_FATAL(plan.has_value(), "TopK's selection factory got a call outside topk_select_plan");
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& values = std::get<0>(outputs).mesh_tensor();
    const auto& indices = std::get<1>(outputs).mesh_tensor();
    auto* device = tensor_args.input.device();

    const auto grid_cores = corerange_to_cores(args.sub_core_grids, plan->cores(), /*row_wise=*/true);
    std::vector<CoreRange> ranges;
    ranges.reserve(grid_cores.size());
    for (const auto& core : grid_cores) {
        ranges.emplace_back(core, core);
    }
    const CoreRangeSet all_cores = CoreRangeSet(ranges).merge_ranges();

    const uint32_t value_bytes = tensor_args.input.element_size();
    const uint32_t index_bytes = std::get<1>(outputs).element_size();
    const uint32_t k_tiles = div_up(args.k, 32);
    const tt::DataFormat value_format = datatype_to_dataformat_converter(std::get<0>(outputs).dtype());
    const tt::DataFormat index_format = datatype_to_dataformat_converter(std::get<1>(outputs).dtype());

    ProgramDescriptor desc;
    auto add_buffer = [&](uint32_t index, uint32_t bytes, tt::DataFormat format) {
        bytes = div_up(bytes, 64) * 64;
        desc.cbs.push_back(CBDescriptor{
            .total_size = bytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = bytes}}}});
    };
    constexpr uint32_t cb_in = 0, cb_candidates = 1, cb_rows = 2, cb_values = 16, cb_indices = 17;
    // A row staged as face-row chunks of 64 bytes, two per tile for the
    // values and two for the indices; it also holds the row's list of k.
    const uint32_t row_bytes = 2 * k_tiles * 2 * 64;
    add_buffer(cb_in, plan->rows_per_core * plan->segment_tiles * 2 * face_row_read_bytes, tt::DataFormat::UInt32);
    add_buffer(
        cb_candidates,
        plan->segments > 1 ? plan->rows_per_core * plan->segments * list_stride(args.k) * candidate_bytes : 64,
        tt::DataFormat::UInt32);
    add_buffer(cb_rows, plan->rows_per_core * row_bytes, tt::DataFormat::UInt32);
    add_buffer(cb_values, k_tiles * tt::tile_size(value_format), value_format);
    add_buffer(cb_indices, k_tiles * tt::tile_size(index_format), index_format);
    constexpr uint32_t merge_sem_id = 0, owner_sem_id = 1;
    for (uint32_t id : {merge_sem_id, owner_sem_id}) {
        desc.semaphores.push_back(SemaphoreDescriptor{.id = id, .core_ranges = all_cores, .initial_value = 0});
    }

    std::vector<uint32_t> compile_args;
    TensorAccessorArgs(input).append_to(compile_args);
    TensorAccessorArgs(values).append_to(compile_args);
    TensorAccessorArgs(indices).append_to(compile_args);
    const std::unordered_map<std::string, uint32_t> named = {
        {"k", args.k},
        {"largest", args.largest ? 1u : 0u},
        {"value_bytes", value_bytes},
        {"index_bytes", index_bytes},
        {"value_tile_bytes", static_cast<uint32_t>(tt::tile_size(value_format))},
        {"index_tile_bytes", static_cast<uint32_t>(tt::tile_size(index_format))},
        {"k_tiles", k_tiles},
        {"height", plan->height},
        {"width", plan->width},
        {"Ht", plan->Ht},
        {"Wt", plan->Wt},
        {"segments", plan->segments},
        {"segment_tiles", plan->segment_tiles},
        {"row_groups", plan->row_groups},
        {"cb_in", cb_in},
        {"cb_candidates", cb_candidates},
        {"cb_rows", cb_rows},
        {"cb_values", cb_values},
        {"cb_indices", cb_indices},
        {"merge_sem", merge_sem_id},
        {"owner_sem", owner_sem_id},
    };
    KernelDescriptor kernel;
    kernel.kernel_source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/topk_select.cpp";
    kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = all_cores;
    kernel.compile_time_args = compile_args;
    kernel.named_compile_time_args = {named.begin(), named.end()};
    kernel.config = ReaderConfigDescriptor{};

    const uint32_t per_tile_row = plan->cores_per_tile_row();
    for (uint32_t i = 0; i < grid_cores.size(); ++i) {
        const uint32_t tile_row = i / per_tile_row;
        const uint32_t worker = i % per_tile_row;
        const uint32_t group = worker % plan->row_groups;
        const CoreCoord owner = device->worker_core_from_logical_core(grid_cores[tile_row * per_tile_row]);
        const CoreCoord merger = device->worker_core_from_logical_core(grid_cores[tile_row * per_tile_row + group]);
        kernel.emplace_runtime_args(
            grid_cores[i],
            {input,
             values,
             indices,
             tile_row,
             worker / plan->row_groups,  // segment
             group,
             static_cast<uint32_t>(owner.x),
             static_cast<uint32_t>(owner.y),
             static_cast<uint32_t>(merger.x),
             static_cast<uint32_t>(merger.y)});
    }
    desc.kernels.push_back(std::move(kernel));
    return desc;
}

}  // namespace ttnn::prim
