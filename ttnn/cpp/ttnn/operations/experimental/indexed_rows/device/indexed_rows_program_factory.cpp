// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/indexed_rows/device/indexed_rows_program_factory.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/indexed_rows/device/indexed_rows_device_operation.hpp"

using namespace tt::tt_metal::experimental;

namespace ttnn::experimental::prim {

namespace {

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/indexed_rows/device/kernels/dataflow/";

// DFB entries hold DRAM pages, whose reads need the L1 address to share the
// page's alignment (64 bytes on Blackhole), so entries are 64-byte multiples.
constexpr uint32_t kEntryAlign = 64;
uint32_t align_entry(uint32_t bytes) { return (bytes + kEntryAlign - 1) / kEntryAlign * kEntryAlign; }

std::string comma_list(const ttsl::SmallVector<uint32_t>& values) {
    std::string out;
    for (uint32_t v : values) {
        if (!out.empty()) {
            out += ',';
        }
        out += std::to_string(v);
    }
    return out;
}

// Element kinds of kernels/indexed_rows_common.hpp.
uint32_t elem_kind(tt::tt_metal::DataType dtype) {
    using tt::tt_metal::DataType;
    switch (dtype) {
        case DataType::FLOAT32: return 1;
        case DataType::BFLOAT16: return 2;
        case DataType::FLOAT16: return 3;
        case DataType::INT32: return 4;
        case DataType::UINT32: return 5;
        case DataType::UINT16: return 7;
        case DataType::INT8: return 8;
        case DataType::UINT8: return 9;
        default: return 0;
    }
}

// The geometry both programs share: pages move in chunks of up to 64 KiB (a
// column window is one chunk), `group` chunks at a time; index pages are read
// `index_batch` positions at a time (one read per component).
struct Geometry {
    IndexedRowsTable table;
    uint32_t Bi, M, K;
    uint32_t elem_bytes, page_bytes, chunk_bytes, entry_bytes, group;
    uint32_t index_batch;         // positions per streamed chunk
    uint32_t index_stride_bytes;  // one component's part of an index chunk, with slack
    uint32_t index_entry_bytes;   // a chunk of index positions: K components
    uint32_t source_entry_bytes;  // a chunk of column-mode source, with slack
    uint32_t pages;               // table pages per batch
    KernelSpec::CompileTimeArgs compile_time_args;
    KernelSpec::CompilerOptions compiler_options;
};

Geometry geometry(const IndexedRowsParams& params, const IndexedRowsInputs& in) {
    Geometry g;
    g.table = indexed_rows_table(params, in.table);
    const auto& index_shape = in.indices.logical_shape();
    g.Bi = index_shape[0];
    g.K = params.sizes.size();
    g.M = index_shape[2];
    g.elem_bytes = in.table.element_size();
    g.page_bytes = g.table.R * g.elem_bytes;
    g.pages = g.table.N * g.table.S;
    constexpr uint32_t kMaxChunk = 64 * 1024;
    g.chunk_bytes = std::min(g.page_bytes, kMaxChunk);
    // A column window lands at its source offset within a 64-byte granule and
    // is moved to its destination offset, so its entry has a granule of slack.
    const uint32_t column_entry = params.column ? g.table.column_window * g.elem_bytes + kEntryAlign : 0;
    g.entry_bytes = align_entry(std::max(g.chunk_bytes, column_entry));
    g.group = std::clamp<uint32_t>((256 * 1024) / g.entry_bytes, 1, 16);
    // Index positions (and, in column mode, their source) stream in chunks of
    // up to 16 KiB of indices and 64 KiB of source.
    const uint32_t index_bytes = g.K * sizeof(uint32_t);  // per position
    const uint32_t source_per_position = params.column ? indexed_rows_window(params) * g.table.column_window * g.elem_bytes : 0;
    g.index_batch = std::max<uint32_t>(1, (16 * 1024) / index_bytes);
    if (source_per_position > 0) {
        g.index_batch = std::clamp<uint32_t>((64 * 1024) / source_per_position, 1, g.index_batch);
    }
    g.index_batch = std::min<uint32_t>(g.index_batch, std::max<uint32_t>(g.M, 1));
    g.index_stride_bytes = align_entry(g.index_batch * sizeof(uint32_t) + kEntryAlign);
    g.index_entry_bytes = g.K * g.index_stride_bytes;
    g.source_entry_bytes = align_entry(std::max<uint32_t>(g.index_batch * source_per_position, 1) + kEntryAlign);

    const uint32_t row_components = g.K - (params.column ? 1 : 0);
    ttsl::SmallVector<uint32_t> strides(g.K, 0);
    uint32_t stride = 1;
    for (uint32_t k = row_components; k-- > 0;) {
        strides[k] = stride;
        stride *= params.sizes[k];
    }
    g.compiler_options.defines = {
        {"ROWS_SIZES", comma_list(params.sizes)},
        {"ROWS_WINDOWS", comma_list(params.windows)},
        {"ROWS_STRIDES", comma_list(strides)},
    };
    g.compile_time_args = {
        {"num_components", g.K},
        {"column", static_cast<uint32_t>(params.column)},
        {"index_unsigned", static_cast<uint32_t>(in.indices.dtype() == tt::tt_metal::DataType::UINT32)},
        {"fill_oob", static_cast<uint32_t>(params.fill_out_of_bounds)},
        {"index_shared", static_cast<uint32_t>(g.Bi == 1)},
        {"window", indexed_rows_window(params)},
        {"positions", g.M},
        {"rows_per_batch", g.table.N},
        {"pages_per_row", g.table.S},
        {"page_bytes", g.page_bytes},
        {"chunk_bytes", g.chunk_bytes},
        {"index_stride", g.index_stride_bytes / static_cast<uint32_t>(sizeof(uint32_t))},
        {"group", g.group},
        {"index_batch", g.index_batch},
        {"resident", 0u},  // rows_scatter sets it
        {"elem_kind", elem_kind(in.table.dtype())},
        {"elem_bytes", g.elem_bytes},
        {"reduce", static_cast<uint32_t>(params.reduce)},
    };
    return g;
}

// Assigns [first, first + count) of `work` items to the cores, evenly.
struct Split {
    uint32_t num_cores;
    CoreRangeSet all_cores;
    std::vector<std::pair<CoreCoord, std::pair<uint32_t, uint32_t>>> ranges;
};

Split split(const tt::tt_metal::distributed::MeshDevice& device, uint32_t work) {
    const auto grid = device.compute_with_storage_grid_size();
    const auto [num_cores, all_cores, core_group_1, core_group_2, per_core_1, per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, work);
    Split s{.num_cores = num_cores, .all_cores = all_cores};
    const auto cores = tt::tt_metal::corerange_to_cores(all_cores, num_cores, /*row_wise=*/false);
    uint32_t first = 0;
    for (const CoreCoord& core : cores) {
        const uint32_t count = core_group_1.contains(core) ? per_core_1 : per_core_2;
        s.ranges.emplace_back(core, std::make_pair(first, count));
        first += count;
    }
    return s;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts RowsGatherProgramFactory::create_program_artifacts(
    const IndexedRowsParams& params, const IndexedRowsInputs& in, std::vector<Tensor>& outputs) {
    const auto& table = in.table.mesh_tensor();
    const auto& indices = in.indices.mesh_tensor();
    const auto& output = outputs.at(0).mesh_tensor();
    const auto& device = table.device();
    const Geometry g = geometry(params, in);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName ROWS_DFB{"rows"};
    const DFBSpecName INDEX_DFB{"index"};
    const TensorParamName TABLE{"table"};
    const TensorParamName INDICES{"indices"};
    const TensorParamName OUTPUT{"output"};

    Group<DataflowBufferSpec> dataflow_buffers = {
        {.unique_id = ROWS_DFB, .entry_size = g.entry_bytes, .num_entries = 2 * g.group},
        {.unique_id = INDEX_DFB, .entry_size = g.index_entry_bytes, .num_entries = 1},
    };

    KernelSpec reader{
        .unique_id = READER,
        .source = std::string(kKernelDir) + "rows_gather_reader.cpp",
        .compiler_options = g.compiler_options,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = ROWS_DFB, .accessor_name = "rows", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TABLE, .accessor_name = "table"},
                TensorBinding{.tensor_parameter_name = INDICES, .accessor_name = "indices"},
            },
        .compile_time_args = g.compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = {"first_item", "num_items"}},
        .hw_config = ttnn::create_reader_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(kKernelDir) + "rows_gather_writer.cpp",
        .compiler_options = g.compiler_options,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = ROWS_DFB, .accessor_name = "rows", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = g.compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = {"first_item", "num_items"}},
        .hw_config = ttnn::create_writer_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    const Split s = split(device, g.table.B * g.M);
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (const auto& [core, range] : s.ranges) {
        for (auto* run_args : {&reader_run_args, &writer_run_args}) {
            AddRuntimeArgsForNode(
                run_args->runtime_arg_values, core, {{"first_item", range.first}, {"num_items", range.second}});
        }
    }

    ProgramSpec spec{
        .name = "rows_gather",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = TABLE, .spec = table.tensor_spec()},
                TensorParameter{.unique_id = INDICES, .spec = indices.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {READER, WRITER}, .target_nodes = s.all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {{TABLE, table}, {INDICES, indices}, {OUTPUT, output}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts RowsScatterProgramFactory::create_program_artifacts(
    const IndexedRowsParams& params, const IndexedRowsInputs& in, std::vector<Tensor>& outputs) {
    const auto& table = in.table.mesh_tensor();
    const auto& indices = in.indices.mesh_tensor();
    const auto& source = in.source->mesh_tensor();
    const auto& output = outputs.at(0).mesh_tensor();
    const auto& device = table.device();
    Geometry g = geometry(params, in);

    // Owned pages that fit L1 are held there and written once after the
    // updates; otherwise they stream to the output and updates read-modify-write
    // DRAM. The split is even, so the largest share decides for every core.
    const Split s = split(device, g.table.B * g.pages);
    uint32_t max_pages = 0;
    for (const auto& [core, range] : s.ranges) {
        max_pages = std::max(max_pages, range.second);
    }
    constexpr uint32_t kResidentBudget = 512 * 1024;
    const bool resident = g.page_bytes <= g.chunk_bytes && max_pages * g.entry_bytes <= kResidentBudget;
    g.compile_time_args["resident"] = static_cast<uint32_t>(resident);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName ROWS_DFB{"rows"};
    const DFBSpecName INDEX_DFB{"index"};
    const DFBSpecName STREAM_DFB{"stream"};
    const DFBSpecName SCRATCH_DFB{"scratch"};
    const TensorParamName TABLE{"table"};
    const TensorParamName INDICES{"indices"};
    const TensorParamName SOURCE{"source"};
    const TensorParamName OUTPUT{"output"};

    Group<DataflowBufferSpec> dataflow_buffers = {
        {.unique_id = ROWS_DFB, .entry_size = g.entry_bytes, .num_entries = resident ? max_pages : 2 * g.group},
        {.unique_id = INDEX_DFB, .entry_size = g.index_entry_bytes, .num_entries = 1},
        {.unique_id = STREAM_DFB, .entry_size = g.source_entry_bytes, .num_entries = 1},
        {.unique_id = SCRATCH_DFB, .entry_size = g.entry_bytes, .num_entries = 2},
    };

    KernelSpec reader{
        .unique_id = READER,
        .source = std::string(kKernelDir) + "rows_scatter_reader.cpp",
        .compiler_options = g.compiler_options,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = ROWS_DFB, .accessor_name = "rows", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = TABLE, .accessor_name = "table"}},
        .compile_time_args = g.compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = {"first_page", "num_pages"}},
        .hw_config = ttnn::create_reader_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(kKernelDir) + "rows_scatter_writer.cpp",
        .compiler_options = g.compiler_options,
        .dfb_bindings =
            {
                DFBBinding{.dfb_spec_name = ROWS_DFB, .accessor_name = "rows", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{.dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = INDEX_DFB, .accessor_name = "index", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = STREAM_DFB, .accessor_name = "stream", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = STREAM_DFB, .accessor_name = "stream", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = SCRATCH_DFB, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = SCRATCH_DFB, .accessor_name = "scratch", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INDICES, .accessor_name = "indices"},
                TensorBinding{.tensor_parameter_name = SOURCE, .accessor_name = "source"},
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args = g.compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = {"first_page", "num_pages"}},
        .hw_config = ttnn::create_writer_datamovement_config(device.arch(), /*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (const auto& [core, range] : s.ranges) {
        for (auto* run_args : {&reader_run_args, &writer_run_args}) {
            AddRuntimeArgsForNode(
                run_args->runtime_arg_values, core, {{"first_page", range.first}, {"num_pages", range.second}});
        }
    }

    ProgramSpec spec{
        .name = "rows_scatter",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = TABLE, .spec = table.tensor_spec()},
                TensorParameter{.unique_id = INDICES, .spec = indices.tensor_spec()},
                TensorParameter{.unique_id = SOURCE, .spec = source.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {{.name = "main", .kernels = {READER, WRITER}, .target_nodes = s.all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args = {{TABLE, table}, {INDICES, indices}, {SOURCE, source}, {OUTPUT, output}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::experimental::prim
