// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/quantize_rows/device/quantize_rows_device_operation.hpp"

#include <algorithm>
#include <bit>
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

// Dense tiles converted per register acquire: the divisor tile and three more
// fill the four FP32 tiles of a DST half.
constexpr uint32_t chunk_tiles = 3;

ttnn::device_operation::ProgramArtifacts QuantizeRowsProgramFactory::create_program_artifacts(
    const QuantizeRowsParams& params, const QuantizeRowsInputs& tensor_args, std::vector<Tensor>& outputs) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& q = outputs.at(0).mesh_tensor();
    const auto& scale = outputs.at(1).mesh_tensor();
    const auto& device = input.device();

    // The cores form a rectangle within the grid's first run of NoC-adjacent
    // columns and rows, which the reader walks by NoC coordinates (a
    // multicast spanning Blackhole's non-worker columns hung).
    const auto grid = device.compute_with_storage_grid_size();
    const auto origin = device.worker_core_from_logical_core({0, 0});
    uint32_t grid_x = 1, grid_y = 1;
    while (grid_x < grid.x && device.worker_core_from_logical_core({grid_x, 0}).x == origin.x + grid_x) {
        ++grid_x;
    }
    while (grid_y < grid.y && device.worker_core_from_logical_core({0, grid_y}).y == origin.y + grid_y) {
        ++grid_y;
    }
    const QuantizeRowsPlan plan = quantize_rows_plan(tensor_args.input, params.max_cores, grid_x, grid_y);
    const uint32_t num_cores = plan.cores();
    const tt::tt_metal::CoreRange rect({0, 0}, {plan.width - 1, plan.height - 1});
    const tt::tt_metal::CoreRangeSet all_cores(rect);
    const auto noc_start = device.worker_core_from_logical_core(rect.start_coord);
    const auto noc_end = device.worker_core_from_logical_core(rect.end_coord);

    const tt::DataFormat in_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t in_tile_bytes = tt::tile_size(in_format);
    const uint32_t f32_tile_bytes = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t q_tile_bytes = tt::tile_size(tt::DataFormat::UInt8);
    const bool e5m2 = params.dtype == tt::tt_metal::DataType::FP8_E5M2;
    const float fp8_max = e5m2 ? 57344.0f : 448.0f;

    // Dense tiles split evenly over the cores, the first ones one longer.
    const uint32_t base = plan.dense / num_cores;
    const uint32_t extra = plan.dense % num_cores;
    const uint32_t max_dense = base + (extra != 0 ? 1 : 0);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName PART_DFB{"part"};
    const DFBSpecName GATHER_DFB{"gather"};
    const DFBSpecName OUT_DFB{"out"};
    const DFBSpecName SCALE_DFB{"scale"};
    const SemaphoreSpecName DONE_SEM{"done"};
    const TensorParamName INPUT{"input"};
    const TensorParamName Q{"q"};
    const TensorParamName SCALE{"scale"};
    Group<DataflowBufferSpec> dataflow_buffers = {
        {.unique_id = IN_DFB, .entry_size = in_tile_bytes, .num_entries = max_dense, .data_format_metadata = in_format},
        {.unique_id = PART_DFB,
         .entry_size = f32_tile_bytes,
         .num_entries = 1,
         .data_format_metadata = tt::DataFormat::Float32},
        {.unique_id = GATHER_DFB,
         .entry_size = f32_tile_bytes,
         .num_entries = plan.gather_tiles(),
         .data_format_metadata = tt::DataFormat::Float32},
        {.unique_id = OUT_DFB,
         .entry_size = q_tile_bytes,
         .num_entries = 2 * chunk_tiles,
         .data_format_metadata = tt::DataFormat::UInt8},
        {.unique_id = SCALE_DFB,
         .entry_size = f32_tile_bytes,
         .num_entries = 1,
         .data_format_metadata = tt::DataFormat::Float32},
    };
    // Every core counts the other cores whose row maxima have arrived.
    Group<SemaphoreSpec> semaphores = {{.unique_id = DONE_SEM, .target_nodes = all_cores}};
    const std::vector<std::string> runtime_arg_names = {
        "first_dense", "num_dense", "last_count", "core", "self_x", "self_y", "writes_scale"};
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/quantize_rows/device/kernels/";
    KernelSpec reader{
        .unique_id = READER,
        .source = kernel_dir + "dataflow/reader_quantize_rows.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = PART_DFB, .accessor_name = "part", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = GATHER_DFB, .accessor_name = "gather", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .semaphore_bindings = {SemaphoreBinding{.semaphore_spec_name = DONE_SEM, .accessor_name = "done"}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"stride", plan.stride},
             {"pack", plan.pack},
             {"elem_bytes", in_tile_bytes / (TILE_HEIGHT * TILE_WIDTH)},
             {"cores", num_cores},
             {"slot_bytes", plan.slot_bytes()},
             {"gather_tiles", plan.gather_tiles()},
             {"x0", static_cast<uint32_t>(noc_start.x)},
             {"y0", static_cast<uint32_t>(noc_start.y)},
             {"x1", static_cast<uint32_t>(noc_end.x)},
             {"y1", static_cast<uint32_t>(noc_end.y)}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = kernel_dir + "dataflow/writer_quantize_rows.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = SCALE_DFB, .accessor_name = "scale", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = Q, .accessor_name = "q"},
                TensorBinding{.tensor_parameter_name = SCALE, .accessor_name = "scale"},
            },
        .compile_time_args = {{"stride", plan.stride}, {"pack", plan.pack}, {"chunk_tiles", chunk_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };
    const ttnn::ComputeKernelConfig compute_kernel_config{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = false,
    };
    auto compute_hw = ttnn::to_compute_hardware_config(compute_kernel_config);
    // FP32 tiles go straight to DST, unrounded; BF16 ones widen exactly.
    if (in_format == tt::DataFormat::Float32) {
        compute_hw.unpack_modes = {
            {GATHER_DFB, tt::tt_metal::UnpackMode::UnpackToDest}, {IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest}};
    } else {
        compute_hw.unpack_modes = {{GATHER_DFB, tt::tt_metal::UnpackMode::UnpackToDest}};
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kernel_dir + "compute/quantize_rows.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = PART_DFB, .accessor_name = "part", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = GATHER_DFB, .accessor_name = "gather", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = SCALE_DFB, .accessor_name = "scale", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"stride", plan.stride},
             {"pack", plan.pack},
             {"cores", num_cores},
             {"slot_bands", plan.slot_bytes() / 256},
             {"gather_tiles", plan.gather_tiles()},
             {"chunk_tiles", chunk_tiles},
             {"e5m2", e5m2 ? 1u : 0u},
             {"fp8_max_bits", std::bit_cast<uint32_t>(fp8_max)},
             {"min_scale_bits", std::bit_cast<uint32_t>(params.min_scale)}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = compute_hw,
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < num_cores; ++i) {
        const tt::tt_metal::CoreCoord core{i % plan.width, i / plan.width};
        const auto noc = device.worker_core_from_logical_core(core);
        const uint32_t num_dense = base + (i < extra ? 1 : 0);
        const uint32_t first_dense = i * base + std::min(i, extra);
        // Input tiles in the core's last dense tile: only the row's last dense
        // tile can hold fewer than `pack`.
        const uint32_t last_first_tile = (first_dense + num_dense - 1) * plan.pack;
        const uint32_t last_count = std::min(plan.pack, plan.tiles - last_first_tile);
        for (auto* run_args : {&reader_run_args, &writer_run_args, &compute_run_args}) {
            AddRuntimeArgsForNode(
                run_args->runtime_arg_values,
                core,
                {{"first_dense", first_dense},
                 {"num_dense", num_dense},
                 {"last_count", last_count},
                 {"core", i},
                 {"self_x", static_cast<uint32_t>(noc.x)},
                 {"self_y", static_cast<uint32_t>(noc.y)},
                 {"writes_scale", i == 0 ? 1u : 0u}});
        }
    }
    ProgramSpec spec{
        .name = "quantize_rows",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .semaphores = std::move(semaphores),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = Q, .spec = q.tensor_spec()},
                TensorParameter{.unique_id = SCALE, .spec = scale.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {{INPUT, input}, {Q, q}, {SCALE, scale}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
