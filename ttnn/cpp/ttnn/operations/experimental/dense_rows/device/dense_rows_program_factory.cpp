// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/dense_rows/device/dense_rows_device_operation.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

// Output tiles a core assembles per read barrier; the buffer holds two batches.
constexpr uint32_t batch_tiles = 8;

ttnn::device_operation::ProgramArtifacts DenseRowsProgramFactory::create_program_artifacts(
    const DenseRowsParams& params, const DenseRowsInputs& tensor_args, std::vector<Tensor>& outputs) {
    const auto& input = tensor_args.input.mesh_tensor();
    const auto& output = outputs.at(0).mesh_tensor();
    const auto& device = input.device();
    const bool unpack = params.direction == DenseRowsDirection::Unpack;

    // The [..., H, W] side of the move: O outer matrices of H rows, each row N
    // chunks of 32. Its tiles are O * Ht rows of N; the chunk side's are O * H
    // matrices of Nt.
    const ttnn::Shape& padded = unpack ? output.logical_shape() : input.logical_shape();
    const uint32_t rank = padded.rank();
    const uint32_t H = rank >= 2 ? padded[-2] : 1;
    const uint32_t N = padded[-1] / TILE_WIDTH;
    uint32_t O = 1;
    for (uint32_t d = 0; d + 2 < rank; ++d) {
        O *= padded[d];
    }
    const uint32_t Ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t Nt = (N + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t num_pages = unpack ? O * Ht * N : O * H * Nt;

    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint32_t>(num_pages, grid.x * grid.y);
    const auto cores = tt::tt_metal::corerange_to_cores(
        tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1})),
        num_cores,
        /*row_wise=*/true);
    std::vector<tt::tt_metal::CoreRange> ranges;
    for (const auto& core : cores) {
        ranges.emplace_back(core, core);
    }
    const tt::tt_metal::CoreRangeSet all_cores = tt::tt_metal::CoreRangeSet(ranges).merge_ranges();

    const tt::DataFormat format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_bytes = tt::tile_size(format);
    const uint32_t row_bytes = tile_bytes / 64;

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    Group<DataflowBufferSpec> dataflow_buffers = {
        {.unique_id = OUT_DFB, .entry_size = tile_bytes, .num_entries = 2 * batch_tiles, .data_format_metadata = format},
    };
    const std::vector<std::string> runtime_arg_names = {"first_page", "num_pages", "H", "N", "Ht", "Nt"};
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/dense_rows/device/kernels/";
    KernelSpec reader{
        .unique_id = READER,
        .source = kernel_dir + "dataflow/reader_dense_rows.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"unpack", unpack ? 1u : 0u}, {"batch_tiles", batch_tiles}, {"row_bytes", row_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = kernel_dir + "dataflow/writer_dense_rows.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"batch_tiles", batch_tiles}, {"row_bytes", row_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = runtime_arg_names},
        .hw_config = ttnn::create_writer_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    const uint32_t base = num_pages / num_cores;
    const uint32_t extra = num_pages % num_cores;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t first = i * base + std::min(i, extra);
        const uint32_t count = base + (i < extra ? 1 : 0);
        for (auto* run_args : {&reader_run_args, &writer_run_args}) {
            AddRuntimeArgsForNode(
                run_args->runtime_arg_values,
                cores[i],
                {{"first_page", first}, {"num_pages", count}, {"H", H}, {"N", N}, {"Ht", Ht}, {"Nt", Nt}});
        }
    }
    ProgramSpec spec{
        .name = "dense_rows",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores}},
    };
    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {{INPUT, input}, {OUTPUT, output}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
