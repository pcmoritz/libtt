// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/matmul_fp32/device/matmul_fp32_program_factory.hpp"

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

    const auto grid = device.compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, batch * Mt * Nt);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_1{"compute_1"};
    const KernelSpecName COMPUTE_2{"compute_2"};
    const DFBSpecName IN0_DFB{"in0"};
    const DFBSpecName IN1_DFB{"in1"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};

    const auto fp32 = tt::DataFormat::Float32;
    const uint32_t tile_size = tt::tile_size(fp32);
    Group<DataflowBufferSpec> dataflow_buffers = {
        {.unique_id = IN0_DFB, .entry_size = tile_size, .num_entries = 2, .data_format_metadata = fp32},
        {.unique_id = IN1_DFB, .entry_size = tile_size, .num_entries = 2, .data_format_metadata = fp32},
        {.unique_id = OUT_DFB, .entry_size = tile_size, .num_entries = 2, .data_format_metadata = fp32},
    };

    // matmul's reader and writer, unchanged.
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
                  "reader_bmm_8bank_output_tiles_partitioned_metal2.cpp",
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
        .compile_time_args =
            {
                {"in0_last_ktile_w", static_cast<uint32_t>(a.logical_shape()[-1] % TILE_WIDTH)},
                {"in0_last_ktile_h", 0u},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"output_tile_start_id", "num_output_tiles"},
                .common_runtime_arg_names = {"Mt", "Kt", "Nt", "MtKt", "KtNt", "batch", "bcast_B", "MtNt"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    reader_run_args.common_runtime_arg_values = {
        {"Mt", Mt},
        {"Kt", Kt},
        {"Nt", Nt},
        {"MtKt", Mt * Kt},
        {"KtNt", Kt * Nt},
        {"batch", batch},
        {"bcast_B", static_cast<uint32_t>(bcast_b)},
        {"MtNt", Mt * Nt},
    };
    KernelRunArgs writer_run_args{.kernel = WRITER};
    const auto cores = tt::tt_metal::corerange_to_cores(all_cores, num_cores, /*row_wise=*/false);
    for (uint32_t i = 0, start = 0; i < num_cores; ++i) {
        const uint32_t tiles = core_group_1.contains(cores[i]) ? tiles_per_core_1 : tiles_per_core_2;
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            cores[i],
            {{"output_tile_start_id", start}, {"num_output_tiles", tiles}});
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, cores[i], {{"num_pages", tiles}, {"start_id", start}});
        start += tiles;
    }

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
    auto make_compute = [&](KernelSpecName unique_id, uint32_t num_tiles) {
        return KernelSpec{
            .unique_id = std::move(unique_id),
            .source = "ttnn/cpp/ttnn/operations/experimental/matmul_fp32/device/kernels/compute/matmul_fp32.cpp",
            .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings =
                {
                    DFBBinding{
                        .dfb_spec_name = IN0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = IN1_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
                    DFBBinding{
                        .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
                },
            .compile_time_args = {{"Kt", Kt}, {"num_tiles", num_tiles}},
            .hw_config = compute_hw,
        };
    };

    Group<KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute(COMPUTE_1, tiles_per_core_1));
    Group<WorkUnitSpec> work_units;
    work_units.push_back({.name = "core_group_1", .kernels = {READER, WRITER, COMPUTE_1}, .target_nodes = core_group_1});
    if (!core_group_2.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_2, tiles_per_core_2));
        work_units.push_back(
            {.name = "core_group_2", .kernels = {READER, WRITER, COMPUTE_2}, .target_nodes = core_group_2});
    }

    ProgramSpec spec{
        .name = "matmul_fp32",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = IN0, .spec = a.tensor_spec()},
                TensorParameter{.unique_id = IN1, .spec = b.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {{IN0, a}, {IN1, b}, {OUTPUT, output}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
