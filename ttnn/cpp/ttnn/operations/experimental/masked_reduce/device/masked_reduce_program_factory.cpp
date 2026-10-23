// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/masked_reduce/device/masked_reduce_device_operation.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

// Dense tiles a core reads per NoC barrier; the input buffer holds two batches.
constexpr uint32_t batch_tiles = 8;

// A tile row holds `rows` rows of data (the rest is tile padding). With at most
// 16 of them, the reader packs the data rows of `pack` input tiles into one
// dense tile, so the kernel reads and masks only data: dense row
// jj * stride + m holds row m of the dense tile's jj-th input tile. The stride
// is the row count, rounded up to an even number for 16-bit elements so that
// every chunk the reader moves starts 64 bytes aligned (the DRAM read
// alignment).
static uint32_t stride_of(uint32_t rows, uint32_t elem_bytes) { return elem_bytes == 2 ? rows + (rows & 1) : rows; }
static uint32_t pack_of(uint32_t stride) { return stride <= TILE_HEIGHT / 2 ? TILE_HEIGHT / stride : 1; }

ttnn::device_operation::ProgramArtifacts MaskedReduceProgramFactory::create_program_artifacts(
    const MaskedReduceParams& params, const MaskedReduceInputs& tensor_args, std::vector<Tensor>& outputs) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& threshold = tensor_args.threshold.mesh_tensor();
    const auto& output = outputs.at(0).mesh_tensor();
    const auto& device = input.device();

    const uint32_t tile_rows = masked_reduce_tile_rows(tensor_args.input);
    const uint32_t H = input.logical_shape()[-2];
    const uint32_t Ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t cores_per_row = params.cores_per_row;
    const uint32_t num_cores = tile_rows * cores_per_row;
    // The partials a tile row's cores send to its master core: a 16-byte slot
    // per core and row, in partial_tiles FP32 tiles of the master's L1.
    const uint32_t partial_cols = masked_reduce_partial_columns(cores_per_row);
    const uint32_t partial_tiles = partial_cols / TILE_WIDTH;
    const auto grid = device.compute_with_storage_grid_size();
    TT_FATAL(num_cores <= grid.x * grid.y, "masked_reduce: {} cores do not fit the grid", num_cores);
    const auto cores = tt::tt_metal::corerange_to_cores(
        tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1})),
        num_cores,
        /*row_wise=*/true);
    std::vector<tt::tt_metal::CoreRange> ranges;
    for (const auto& core : cores) {
        ranges.emplace_back(core, core);
    }
    const tt::tt_metal::CoreRangeSet all_cores = tt::tt_metal::CoreRangeSet(ranges).merge_ranges();

    const tt::DataFormat in_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t in_tile_bytes = tt::tile_size(in_format);
    const uint32_t f32_tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    // Columns of the last tile that hold data; 0 when the width is whole tiles.
    const uint32_t last_valid = input.logical_shape()[-1] % TILE_WIDTH;

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName T_DFB{"t"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName PART_DFB{"part"};
    const SemaphoreSpecName DONE_SEM{"done"};
    const TensorParamName INPUT{"input"};
    const TensorParamName THRESHOLD{"threshold"};
    const TensorParamName OUTPUT{"output"};
    Group<DataflowBufferSpec> dataflow_buffers = {
        {.unique_id = IN_DFB,
         .entry_size = in_tile_bytes,
         .num_entries = 2 * batch_tiles,
         .data_format_metadata = in_format},
        {.unique_id = T_DFB, .entry_size = in_tile_bytes, .num_entries = 1, .data_format_metadata = in_format},
        {.unique_id = OUT_DFB,
         .entry_size = f32_tile_bytes,
         .num_entries = 1,
         .data_format_metadata = tt::DataFormat::Float32},
        {.unique_id = PART_DFB,
         .entry_size = f32_tile_bytes,
         .num_entries = partial_tiles,
         .data_format_metadata = tt::DataFormat::Float32},
    };
    // Every core counts the row's partials that arrived at its master.
    Group<SemaphoreSpec> semaphores = {{.unique_id = DONE_SEM, .target_nodes = all_cores}};
    const std::vector<std::string> runtime_arg_names = {
        "tile_row",
        "first_tile",
        "num_tiles",
        "rows",
        "stride",
        "pack",
        "has_last",
        "col",
        "is_master",
        "master_x",
        "master_y",
        "remote_workers"};
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/masked_reduce/device/kernels/";
    KernelSpec reader{
        .unique_id = READER,
        .source = kernel_dir + "dataflow/reader_masked_reduce.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = T_DFB, .accessor_name = "t", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = THRESHOLD, .accessor_name = "threshold"},
            },
        .compile_time_args =
            {{"batch_tiles", batch_tiles}, {"elem_bytes", in_tile_bytes / (TILE_HEIGHT * TILE_WIDTH)}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = kernel_dir + "dataflow/writer_masked_reduce.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = PART_DFB, .accessor_name = "part", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .semaphore_bindings = {SemaphoreBinding{.semaphore_spec_name = DONE_SEM, .accessor_name = "done"}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"cols", partial_cols}, {"partial_tiles", partial_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_writer_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    const DeviceComputeKernelConfig compute_kernel_config = ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = false,
    };
    auto compute_hw = ttnn::to_compute_hardware_config(device.arch(), compute_kernel_config);
    if (in_format == tt::DataFormat::Float32) {
        // FP32 tiles go straight to DST, where the SFPU subtracts them exactly.
        unpack_modes(compute_hw) = {
            {IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest},
            {T_DFB, tt::tt_metal::UnpackMode::UnpackToDest},
            {PART_DFB, tt::tt_metal::UnpackMode::UnpackToDest},
        };
    } else {
        // The partials are FP32 whatever the input: sum them exactly.
        unpack_modes(compute_hw) = {{PART_DFB, tt::tt_metal::UnpackMode::UnpackToDest}};
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kernel_dir + "compute/masked_reduce.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = T_DFB, .accessor_name = "t", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = PART_DFB, .accessor_name = "part", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .compile_time_args =
            {{"compare", static_cast<uint32_t>(params.compare)},
             {"mode", static_cast<uint32_t>(params.mode)},
             {"last_valid", last_valid},
             {"partial_tiles", partial_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = compute_hw,
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t tile_row = i / cores_per_row;
        const uint32_t worker = i % cores_per_row;
        // The row's first core gathers the partials and writes the result.
        const auto master = device.worker_core_from_logical_core(cores[tile_row * cores_per_row]);
        // Data rows of this tile row: the last tile row of a [..., H, W] batch
        // holds H's remainder.
        const uint32_t rows = (tile_row % Ht == Ht - 1 && H % TILE_HEIGHT != 0) ? H % TILE_HEIGHT : TILE_HEIGHT;
        const uint32_t stride = stride_of(rows, in_tile_bytes / (TILE_HEIGHT * TILE_WIDTH));
        const uint32_t pack = pack_of(stride);
        // The row's dense tiles split evenly over its cores, the first ones one
        // dense tile longer; only the row's last dense tile can be partial.
        const uint32_t dense_tiles = (Wt + pack - 1) / pack;
        const uint32_t base = dense_tiles / cores_per_row;
        const uint32_t extra = dense_tiles % cores_per_row;
        const uint32_t dense_count = base + (worker < extra ? 1 : 0);
        const uint32_t dense_first = worker * base + std::min(worker, extra);
        const uint32_t first = dense_first * pack;
        const uint32_t count = dense_count == 0 ? 0 : std::min(dense_count * pack, Wt - first);
        for (auto* run_args : {&reader_run_args, &writer_run_args, &compute_run_args}) {
            AddRuntimeArgsForNode(
                run_args->runtime_arg_values,
                cores[i],
                {{"tile_row", tile_row},
                 {"first_tile", tile_row * Wt + first},
                 {"num_tiles", count},
                 {"rows", rows},
                 {"stride", stride},
                 {"pack", pack},
                 {"has_last", count != 0 && first + count == Wt ? 1u : 0u},
                 {"col", 4 * worker},
                 {"is_master", worker == 0 ? 1u : 0u},
                 {"master_x", static_cast<uint32_t>(master.x)},
                 {"master_y", static_cast<uint32_t>(master.y)},
                 {"remote_workers", cores_per_row - 1}});
        }
    }
    ProgramSpec spec{
        .name = "masked_reduce",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = THRESHOLD, .spec = threshold.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {{INPUT, input}, {THRESHOLD, threshold}, {OUTPUT, output}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
