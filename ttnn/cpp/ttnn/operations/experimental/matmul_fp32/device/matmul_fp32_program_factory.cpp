// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/matmul_fp32/device/matmul_fp32_program_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

namespace {

// Product of all but the last two dimensions.
uint32_t batch_size(const tt::tt_metal::Shape& shape) {
    uint32_t batch = 1;
    for (int i = 0; i < static_cast<int>(shape.rank()) - 2; ++i) {
        batch *= shape[i];
    }
    return batch;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts MatmulFp32ProgramFactory::create_program_artifacts(
    const MatmulFp32Params&, const MatmulFp32Inputs& tensor_args, std::vector<Tensor>& tensor_return_value) {
    const auto& a = tensor_args.a.mesh_tensor();
    const auto& b = tensor_args.b.mesh_tensor();
    const auto& output = tensor_return_value.at(0).mesh_tensor();
    const auto& device = a.device();

    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const uint32_t batch = batch_size(ashape);
    const bool bcast_b = batch_size(bshape) == 1;
    const uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    const uint32_t Kt = ashape[-1] / TILE_WIDTH;
    const uint32_t Nt = bshape[-1] / TILE_WIDTH;

    // The output tiles are split evenly over the cores. A core takes its tiles
    // in runs of up to run_width consecutive tiles of one output row, whose A
    // row it reads once and keeps in L1 for the run (kernels/matmul_fp32_runs.hpp).
    // A row too big for that is streamed one tile per k instead, for one
    // output tile at a time.
    constexpr uint32_t max_resident_a_bytes = 1 << 20;
    const uint32_t tile_size = tt::tile_size(tt::DataFormat::Float32);
    const bool a_resident = Kt * tile_size <= max_resident_a_bytes;
    const uint32_t run_width = a_resident ? std::min<uint32_t>(8, Nt) : 1;
    // Double buffered rows when they fit, else one row; two k's when streaming.
    const uint32_t in0_entries = !a_resident ? 2 : 2 * Kt * tile_size <= max_resident_a_bytes ? 2 * Kt : Kt;
    const auto grid = device.compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, batch * Mt * Nt);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};

    const auto fp32 = tt::DataFormat::Float32;
    Group<DataflowBufferSpec> dataflow_buffers = {
        {.unique_id = IN0_DFB, .entry_size = tile_size, .num_entries = in0_entries, .data_format_metadata = fp32},
        {.unique_id = IN1_DFB, .entry_size = tile_size, .num_entries = 2, .data_format_metadata = fp32},
        {.unique_id = OUT_DFB, .entry_size = tile_size, .num_entries = 2, .data_format_metadata = fp32},
    };

    const std::vector<std::string> common_arg_names = {"Mt", "Kt", "Nt", "bcast_b", "run_width", "a_resident"};
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/matmul_fp32/device/kernels/";
    KernelSpec reader{
        .unique_id = READER,
        .source = kernel_dir + "dataflow/reader_matmul_fp32.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"},
                TensorBinding{.tensor_parameter_name = IN1, .accessor_name = "in1"},
            },
        .compile_time_args = {{"a_last_ktile_w", static_cast<uint32_t>(a.logical_shape()[-1] % TILE_WIDTH)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_tile", "num_tiles"}, .common_runtime_arg_names = common_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = kernel_dir + "dataflow/writer_matmul_fp32.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_tile", "num_tiles"}, .common_runtime_arg_names = common_arg_names},
        .hw_config = ttnn::create_writer_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // FP32 DST, with the inputs unpacked straight to DST so they stay FP32.
    const DeviceComputeKernelConfig compute_kernel_config = ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = false,
    };
    auto compute_hw = ttnn::to_compute_hardware_config(device.arch(), compute_kernel_config);
    unpack_modes(compute_hw) = {
        {IN0_DFB, tt::tt_metal::UnpackMode::UnpackToDest},
        {IN1_DFB, tt::tt_metal::UnpackMode::UnpackToDest},
    };
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kernel_dir + "compute/matmul_fp32.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_tile", "num_tiles"}, .common_runtime_arg_names = common_arg_names},
        .hw_config = compute_hw,
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (auto* run_args : {&reader_run_args, &writer_run_args, &compute_run_args}) {
        run_args->common_runtime_arg_values = {
            {"Mt", Mt}, {"Kt", Kt}, {"Nt", Nt}, {"bcast_b", static_cast<uint32_t>(bcast_b)}, {"run_width", run_width},
            {"a_resident", static_cast<uint32_t>(a_resident)}};
    }
    const auto cores = tt::tt_metal::corerange_to_cores(all_cores, num_cores, /*row_wise=*/false);
    for (uint32_t i = 0, first = 0; i < num_cores; ++i) {
        const uint32_t count = core_group_1.contains(cores[i]) ? tiles_per_core_1 : tiles_per_core_2;
        for (auto* run_args : {&reader_run_args, &writer_run_args, &compute_run_args}) {
            AddRuntimeArgsForNode(run_args->runtime_arg_values, cores[i], {{"first_tile", first}, {"num_tiles", count}});
        }
        first += count;
    }

    ProgramSpec spec{
        .name = "matmul_fp32",
        .kernels = {std::move(reader), std::move(writer), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = IN0, .spec = a.tensor_spec()},
                TensorParameter{.unique_id = IN1, .spec = b.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {{IN0, a}, {IN1, b}, {OUTPUT, output}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
