// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/rng_bits/device/rng_bits_program_factory.hpp"

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

ttnn::device_operation::ProgramArtifacts RngBitsProgramFactory::create_program_artifacts(
    const RngBitsParams& params, const RngBitsInputs& tensor_args, std::vector<Tensor>& tensor_return_value) {
    const auto& state = tensor_args.state.mesh_tensor();
    const auto& bits = tensor_return_value.at(0).mesh_tensor();
    const auto& new_state = tensor_return_value.at(1).mesh_tensor();
    const auto& device = state.device();

    // The bits are [batch..., H, W] (H = 1 for a vector), in tiles of the
    // padded [..., Ht * 32, Wt * 32]. Element (r, c) of tile (b, tr, tc) is
    // element (b * H + tr * 32 + r) * W + tc * 32 + c in row-major order.
    const auto& logical = bits.logical_shape();
    const auto& padded = bits.padded_shape();
    const uint32_t W = logical[-1];
    const uint32_t H = logical.rank() >= 2 ? logical[-2] : 1;
    const uint32_t Wt = padded[-1] / TILE_WIDTH;
    const uint32_t Ht = padded[-2] / TILE_HEIGHT;
    const uint32_t num_tiles = bits.physical_volume() / (TILE_HEIGHT * TILE_WIDTH);
    // w1 advances by one per four elements.
    const uint32_t advance = static_cast<uint32_t>((logical.volume() + 3) / 4);

    const auto grid = device.compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, num_tiles);

    const KernelSpecName MOVER{"mover"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName STATE_DFB{"state"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName STATE{"state"};
    const TensorParamName BITS{"bits"};
    const TensorParamName NEW_STATE{"new_state"};

    // DRAM pages read into L1 need 64-byte aligned entries.
    const uint32_t state_page_bytes = state.tensor_spec().compute_page_size_bytes();
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::UInt32);
    Group<DataflowBufferSpec> dataflow_buffers = {
        {.unique_id = STATE_DFB,
         .entry_size = (state_page_bytes + 63) / 64 * 64,
         .num_entries = 1,
         .data_format_metadata = tt::DataFormat::UInt32},
        {.unique_id = OUT_DFB, .entry_size = tile_bytes, .num_entries = 2, .data_format_metadata = tt::DataFormat::UInt32},
    };

    const std::vector<std::string> common_arg_names = {"W", "H", "Wt", "Ht", "advance"};
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/rng_bits/device/kernels/";
    KernelSpec mover{
        .unique_id = MOVER,
        .source = kernel_dir + "dataflow/rng_bits_mover.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STATE_DFB, .accessor_name = "state", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = STATE, .accessor_name = "state"},
                TensorBinding{.tensor_parameter_name = BITS, .accessor_name = "bits"},
                TensorBinding{.tensor_parameter_name = NEW_STATE, .accessor_name = "new_state"},
            },
        .compile_time_args = {{"state_page_bytes", state_page_bytes}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_tile", "num_tiles", "write_state"},
             .common_runtime_arg_names = common_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    // 32-bit DST for the integer words. Four tiles fit in each half of DST.
    const bool philox = params.algorithm == RngAlgorithm::Philox;
    const DeviceComputeKernelConfig compute_kernel_config = ttnn::ComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = false,
        .dst_full_sync_en = false,
    };
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kernel_dir + "compute/rng_bits.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STATE_DFB, .accessor_name = "state", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .compile_time_args =
            {{"philox", static_cast<uint32_t>(philox)},
             {"width", params.width},
             {"state_words", static_cast<uint32_t>(state.logical_shape()[-1])}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"first_tile", "num_tiles"}, .common_runtime_arg_names = common_arg_names},
        .hw_config = ttnn::to_compute_hardware_config(device.arch(), compute_kernel_config),
    };

    KernelRunArgs mover_run_args{.kernel = MOVER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (auto* run_args : {&mover_run_args, &compute_run_args}) {
        run_args->common_runtime_arg_values = {{"W", W}, {"H", H}, {"Wt", Wt}, {"Ht", Ht}, {"advance", advance}};
    }
    const auto cores = tt::tt_metal::corerange_to_cores(all_cores, num_cores, /*row_wise=*/false);
    for (uint32_t i = 0, first = 0; i < num_cores; ++i) {
        const uint32_t count = core_group_1.contains(cores[i]) ? tiles_per_core_1 : tiles_per_core_2;
        AddRuntimeArgsForNode(
            mover_run_args.runtime_arg_values,
            cores[i],
            {{"first_tile", first}, {"num_tiles", count}, {"write_state", static_cast<uint32_t>(i == 0)}});
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, cores[i], {{"first_tile", first}, {"num_tiles", count}});
        first += count;
    }

    ProgramSpec spec{
        .name = "rng_bits",
        .kernels = {std::move(mover), std::move(compute)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = STATE, .spec = state.tensor_spec()},
                TensorParameter{.unique_id = BITS, .spec = bits.tensor_spec()},
                TensorParameter{.unique_id = NEW_STATE, .spec = new_state.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {MOVER, COMPUTE}, .target_nodes = all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(mover_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args = {{STATE, state}, {BITS, bits}, {NEW_STATE, new_state}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
