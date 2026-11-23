// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/attn_res/attn_res.hpp"

#include <algorithm>
#include <bit>
#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
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

struct AttnResParams {
    float eps_res = 1e-6f;
    float eps_norm = 1e-6f;
    bool normalize = true;
    DataType dtype = DataType::BFLOAT16;
};

struct AttnResInputs {
    std::vector<Tensor> sources;
    std::optional<Tensor> addend;
    Tensor queries;
    std::optional<Tensor> weight;
};

struct AttnResOperation {
    using operation_attributes_t = AttnResParams;
    using tensor_args_t = AttnResInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

constexpr uint32_t max_sources = 8;
constexpr uint32_t max_queries = 3;
constexpr uint32_t max_tokens = 4;
// Each core takes up to two tile columns; a gather tile holds eight cores'
// rows (a band of four rows of each face pair).
constexpr uint32_t tiles_per_core = 2;
constexpr uint32_t slots_per_gather_tile = 8;
constexpr uint32_t max_cores = 48;

bool is_float(DataType dtype) { return dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32; }

std::optional<std::string> tensor_reason(const Tensor& tensor, const char* name) {
    if (tensor.storage_type() != StorageType::DEVICE || tensor.buffer() == nullptr) {
        return fmt::format("{} must be on device", name);
    }
    if (tensor.layout() != Layout::TILE || tensor.tensor_spec().tile() != Tile()) {
        return fmt::format("{} must use 32x32 tiles", name);
    }
    if (tensor.memory_config().buffer_type() != BufferType::DRAM ||
        tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return fmt::format("{} must be interleaved in DRAM", name);
    }
    if (!is_float(tensor.dtype())) {
        return fmt::format("{} must be BFLOAT16 or FLOAT32", name);
    }
    return std::nullopt;
}

// Rows of a tensor whose padded shape is one tile row of `width` columns.
std::optional<uint32_t> one_tile_row(const Tensor& tensor, uint32_t width) {
    const auto& logical = tensor.logical_shape();
    const auto& padded = tensor.padded_shape();
    if (logical[-1] != width || padded.volume() != static_cast<uint64_t>(TILE_HEIGHT) * width) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(logical.volume() / width);
}

std::optional<std::string> attn_res_reason(const AttnResInputs& in) {
    const auto& sources = in.sources;
    if (sources.empty() || sources.size() > max_sources) {
        return fmt::format("1 to {} sources, got {}", max_sources, sources.size());
    }
    const uint32_t width = sources[0].logical_shape()[-1];
    if (width % TILE_WIDTH != 0) {
        return fmt::format("the width {} must be a multiple of {}", width, TILE_WIDTH);
    }
    const uint32_t cores = (width / TILE_WIDTH + tiles_per_core - 1) / tiles_per_core;
    if (cores > max_cores) {
        return fmt::format("a width of {} needs {} cores, more than {}", width, cores, max_cores);
    }
    const auto tokens = one_tile_row(sources[0], width);
    if (!tokens.has_value() || *tokens == 0 || *tokens > max_tokens) {
        return fmt::format("sources must be [T, W] with T <= {}, got {}", max_tokens, sources[0].logical_shape());
    }
    for (const auto& source : sources) {
        if (auto reason = tensor_reason(source, "a source")) {
            return reason;
        }
        if (source.logical_shape() != sources[0].logical_shape() || source.dtype() != sources[0].dtype()) {
            return "sources must share a shape and a dtype";
        }
    }
    if (in.addend.has_value()) {
        if (auto reason = tensor_reason(*in.addend, "the addend")) {
            return reason;
        }
        if (in.addend->logical_shape() != sources[0].logical_shape()) {
            return "the addend must have the sources' shape";
        }
    }
    if (auto reason = tensor_reason(in.queries, "the queries")) {
        return reason;
    }
    const auto queries = one_tile_row(in.queries, width);
    if (!queries.has_value() || *queries == 0 || *queries > max_queries) {
        return fmt::format("queries must be [Q, W] with Q <= {}, got {}", max_queries, in.queries.logical_shape());
    }
    if (in.weight.has_value()) {
        if (auto reason = tensor_reason(*in.weight, "the weight")) {
            return reason;
        }
        if (in.weight->logical_shape() != in.queries.logical_shape()) {
            return "the weight must have the queries' shape";
        }
    }
    return std::nullopt;
}

void AttnResOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    if (auto reason = attn_res_reason(in)) {
        TT_THROW("attn_res: {}", *reason);
    }
    TT_FATAL(is_float(params.dtype), "attn_res: the output must be BFLOAT16 or FLOAT32");
}

AttnResOperation::spec_return_value_t AttnResOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const auto& shape = in.sources[0].logical_shape();
    const uint32_t width = shape[-1];
    const uint32_t tokens = shape.volume() / width;
    const uint32_t queries = in.queries.logical_shape().volume() / width;
    return TensorSpec(
        ttnn::Shape({queries, tokens, width}),
        TensorLayout(params.dtype, PageConfig(Layout::TILE), in.sources[0].memory_config()));
}

AttnResOperation::tensor_return_value_t AttnResOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(params, in), in.sources[0].device());
}

ProgramDescriptor AttnResOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& output_tensor) {
    const auto& first_source = in.sources[0].mesh_tensor();
    const auto& queries = in.queries.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    // Absent operands stand in as the queries in the kernels' arguments.
    const auto& addend = in.addend.has_value() ? in.addend->mesh_tensor() : queries;
    const auto& weight = in.weight.has_value() ? in.weight->mesh_tensor() : queries;
    auto* device = &first_source.mutable_device();

    const uint32_t n = in.sources.size();
    const uint32_t width = first_source.logical_shape()[-1];
    const uint32_t Wt = width / TILE_WIDTH;
    const uint32_t tokens = first_source.logical_shape().volume() / width;
    const uint32_t num_queries = queries.logical_shape().volume() / width;
    const bool has_addend = in.addend.has_value();
    const bool has_weight = in.weight.has_value();
    const uint32_t num_cores = (Wt + tiles_per_core - 1) / tiles_per_core;
    const uint32_t gather_tiles = (num_cores + slots_per_gather_tile - 1) / slots_per_gather_tile;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    TT_FATAL(num_cores <= grid.x * grid.y, "attn_res: {} cores do not fit the grid", num_cores);
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    TT_FATAL(cores.ranges().size() <= 2, "attn_res: the cores must form at most two rectangles");
    const auto core_list = corerange_to_cores(cores, num_cores, /*row_wise=*/true);

    ProgramDescriptor program;
    auto make_cb = [&](uint32_t index, uint32_t tiles, DataType dtype) {
        const auto format = datatype_to_dataformat_converter(dtype);
        const uint32_t page_size = tt::tile_size(format);
        program.cbs.push_back(CBDescriptor{
            .total_size = tiles * page_size,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index), .data_format = format, .page_size = page_size}}}});
    };
    constexpr uint32_t cb_sources = 0, cb_addend = 1, cb_queries = 2, cb_weight = 3, cb_row1 = 4, cb_stash = 5,
                       cb_gather1 = 6, cb_row2 = 7, cb_gather2 = 8, cb_mix = 9, cb_output = 16;
    make_cb(cb_sources, n * tiles_per_core, first_source.dtype());
    make_cb(cb_addend, tiles_per_core, addend.dtype());
    make_cb(cb_queries, tiles_per_core, queries.dtype());
    make_cb(cb_weight, tiles_per_core, weight.dtype());
    make_cb(cb_row1, 1, DataType::FLOAT32);
    make_cb(cb_stash, 2, DataType::FLOAT32);
    make_cb(cb_gather1, gather_tiles, DataType::FLOAT32);
    make_cb(cb_row2, 1, DataType::FLOAT32);
    make_cb(cb_gather2, gather_tiles, DataType::FLOAT32);
    make_cb(cb_mix, 1, DataType::FLOAT32);
    make_cb(cb_output, 1, params.dtype);

    constexpr uint32_t written_sem = 0, valid_sem = 1;
    for (uint32_t id : {written_sem, valid_sem}) {
        program.semaphores.push_back(SemaphoreDescriptor{.id = id, .core_ranges = cores, .initial_value = 0});
    }

    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_sources", cb_sources},
        {"cb_addend", cb_addend},
        {"cb_queries", cb_queries},
        {"cb_weight", cb_weight},
        {"cb_row1", cb_row1},
        {"cb_stash", cb_stash},
        {"cb_gather1", cb_gather1},
        {"cb_row2", cb_row2},
        {"cb_gather2", cb_gather2},
        {"cb_mix", cb_mix},
        {"cb_output", cb_output},
        {"num_sources", n},
        {"num_queries", num_queries},
        {"tokens", tokens},
        {"has_addend", has_addend ? 1u : 0u},
        {"has_weight", has_weight ? 1u : 0u},
        {"normalize", params.normalize ? 1u : 0u},
        {"num_cores", num_cores},
        {"gather_tiles", gather_tiles},
        {"Wt", Wt},
        {"inv_width", std::bit_cast<uint32_t>(1.0f / static_cast<float>(width))},
        {"eps_res", std::bit_cast<uint32_t>(params.eps_res)},
        {"eps_norm", std::bit_cast<uint32_t>(params.eps_norm)},
        {"written_sem", written_sem},
        {"valid_sem", valid_sem},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/attn_res/device/kernels/";

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(first_source).append_to(reader_args);
    TensorAccessorArgs(addend).append_to(reader_args);
    TensorAccessorArgs(queries).append_to(reader_args);
    TensorAccessorArgs(weight).append_to(reader_args);
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_attn_res.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}};
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(output).append_to(writer_args);
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_attn_res.cpp",
        .core_ranges = cores,
        .compile_time_args = writer_args,
        .named_compile_time_args = common_args,
        // NOC 0, whose multicast rectangles run from the start to the end
        // coordinates.
        .config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}};
    // Every operand and exchange unpacks into DST in full; FP32 ones straight
    // to DST. All eight FP32 DST tiles: the program keeps its operands there
    // between the exchanges' packs.
    ComputeConfigDescriptor compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .math_approx_mode = false};
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (auto [cb, dtype] : {std::pair{cb_sources, first_source.dtype()},
                             {cb_addend, addend.dtype()},
                             {cb_queries, queries.dtype()},
                             {cb_weight, weight.dtype()},
                             {cb_stash, DataType::FLOAT32},
                             {cb_gather1, DataType::FLOAT32},
                             {cb_gather2, DataType::FLOAT32},
                             {cb_mix, DataType::FLOAT32}}) {
        if (dtype == DataType::FLOAT32) {
            compute_config.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
        }
    }
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/attn_res.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = compute_config};

    // The first core multicasts the gathered rows to the rectangles of the
    // core set (the full rows of the grid and the partial last row), whose
    // destination counts leave the first core out.
    const CoreCoord gather = device->worker_core_from_logical_core(core_list[0]);
    std::vector<uint32_t> rectangles;
    for (const CoreRange& range : cores.ranges()) {
        const CoreCoord start = device->worker_core_from_logical_core(range.start_coord);
        const CoreCoord end = device->worker_core_from_logical_core(range.end_coord);
        const uint32_t dests = range.size() - (range.contains(core_list[0]) ? 1 : 0);
        rectangles.insert(
            rectangles.end(),
            {static_cast<uint32_t>(start.x),
             static_cast<uint32_t>(start.y),
             static_cast<uint32_t>(end.x),
             static_cast<uint32_t>(end.y),
             dests});
    }
    rectangles.resize(10, 0);

    using Arg = std::variant<uint32_t, std::reference_wrapper<const MeshTensor>>;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t start = i * tiles_per_core;
        const uint32_t count = std::min(tiles_per_core, Wt - start);
        std::vector<Arg> r = {start, count, std::cref(queries), std::cref(addend), std::cref(weight)};
        for (const auto& source : in.sources) {
            r.emplace_back(std::cref(source.mesh_tensor()));
        }
        reader.emplace_runtime_args(core_list[i], r);
        std::vector<Arg> w = {
            i, start, count, std::cref(output), static_cast<uint32_t>(gather.x), static_cast<uint32_t>(gather.y)};
        for (uint32_t value : rectangles) {
            w.emplace_back(value);
        }
        writer.emplace_runtime_args(core_list[i], w);
        compute.emplace_runtime_args(core_list[i], {count});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

std::optional<std::string> attn_res_reason_of(
    const std::vector<Tensor>& sources,
    const std::optional<Tensor>& addend,
    const Tensor& queries,
    const std::optional<Tensor>& weight) {
    return attn_res_reason({.sources = sources, .addend = addend, .queries = queries, .weight = weight});
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

std::optional<std::string> attn_res_unsupported_reason(
    const std::vector<ttnn::Tensor>& sources,
    const std::optional<ttnn::Tensor>& addend,
    const ttnn::Tensor& queries,
    const std::optional<ttnn::Tensor>& weight) {
    return prim::attn_res_reason_of(sources, addend, queries, weight);
}

ttnn::Tensor attn_res(
    const std::vector<ttnn::Tensor>& sources,
    const std::optional<ttnn::Tensor>& addend,
    const ttnn::Tensor& queries,
    const std::optional<ttnn::Tensor>& weight,
    float eps_res,
    float eps_norm,
    bool normalize,
    std::optional<tt::tt_metal::DataType> dtype) {
    using Op = prim::AttnResOperation;
    const auto out_dtype = dtype.value_or(weight.has_value() ? DataType::BFLOAT16 : DataType::FLOAT32);
    return device_operation::launch<Op>(
        {.eps_res = eps_res, .eps_norm = eps_norm, .normalize = normalize, .dtype = out_dtype},
        {.sources = sources, .addend = addend, .queries = queries, .weight = weight});
}

}  // namespace ttnn::experimental
