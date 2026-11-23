// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/window_bias/window_bias.hpp"

#include <bit>
#include <functional>
#include <string>
#include <utility>
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

struct WindowBiasParams {
    uint32_t heads = 0;
    float sink_scale = 1.0f;
};

struct WindowBiasInputs {
    Tensor rel_r;
    Tensor table;
    Tensor positions;
    std::optional<Tensor> sink;
};

struct WindowBiasOperation {
    using operation_attributes_t = WindowBiasParams;
    using tensor_args_t = WindowBiasInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// A table row: 16 FP32 values, 64 bytes, one row of an FP32 face.
constexpr uint32_t rel_dim = 16;

bool tiled_dram(const Tensor& t) {
    return t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr && t.layout() == Layout::TILE &&
           t.tensor_spec().tile() == Tile() && t.memory_config().buffer_type() == BufferType::DRAM &&
           t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
}

std::optional<std::string> window_bias_reason(
    const Tensor& rel_r, const Tensor& table, const Tensor& positions, uint32_t heads) {
    if (!tiled_dram(rel_r) || !tiled_dram(table) || !tiled_dram(positions)) {
        return "rel_r, the table and the positions must be tiled and interleaved in DRAM";
    }
    if (rel_r.dtype() != DataType::BFLOAT16 || table.dtype() != DataType::FLOAT32 ||
        (positions.dtype() != DataType::INT32 && positions.dtype() != DataType::UINT32)) {
        return "rel_r must be BFLOAT16, the table FLOAT32 and the positions INT32";
    }
    const auto& r = rel_r.logical_shape();
    if (heads == 0 || heads > TILE_HEIGHT || r.rank() != 2 || r[0] == 0 || r[0] > TILE_HEIGHT ||
        r[1] != heads * rel_dim) {
        return fmt::format("rel_r must be [T <= 32, heads * 16] with heads <= 32, got {} for {} heads", r, heads);
    }
    const auto& t = table.logical_shape();
    if (t.rank() != 2 || t[1] != rel_dim || t[0] == 0 || t[0] % (2 * TILE_HEIGHT) != 0) {
        return fmt::format("the table must be [2W, 16] with W a multiple of 32, got {}", t);
    }
    const auto& p = positions.logical_shape();
    if (p.volume() != r[0] || p[-1] != r[0]) {
        return fmt::format("the positions must be [{}], got {}", r[0], p);
    }
    return std::nullopt;
}

void WindowBiasOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& in) {
    if (auto reason = window_bias_reason(in.rel_r, in.table, in.positions, params.heads)) {
        TT_THROW("window_bias: {}", *reason);
    }
    if (in.sink.has_value()) {
        const auto& s = *in.sink;
        TT_FATAL(
            tiled_dram(s) && s.dtype() == DataType::BFLOAT16 && s.logical_shape().rank() == 2 &&
                s.logical_shape()[0] == in.rel_r.logical_shape()[0] && s.logical_shape()[1] == params.heads,
            "window_bias: the sink must be tiled [{}, {}] BFLOAT16, got {}",
            in.rel_r.logical_shape()[0],
            params.heads,
            s.logical_shape());
    }
}

WindowBiasOperation::spec_return_value_t WindowBiasOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const uint32_t T = in.rel_r.logical_shape()[0];
    const uint32_t W = in.table.logical_shape()[0] / 2;
    auto spec = [&](uint32_t width) {
        return TensorSpec(
            ttnn::Shape({T, 1, params.heads, width}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in.rel_r.memory_config()));
    };
    spec_return_value_t specs = {spec(W)};
    if (in.sink.has_value()) {
        specs.push_back(spec(TILE_WIDTH));
    }
    return specs;
}

WindowBiasOperation::tensor_return_value_t WindowBiasOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(params, in)) {
        outputs.push_back(create_device_tensor(spec, in.rel_r.device()));
    }
    return outputs;
}

ProgramDescriptor WindowBiasOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& outputs) {
    const auto& rel_r = in.rel_r.mesh_tensor();
    const auto& table = in.table.mesh_tensor();
    const auto& positions = in.positions.mesh_tensor();
    const bool has_sink = in.sink.has_value();
    const auto& sink = has_sink ? in.sink->mesh_tensor() : rel_r;
    const auto& bias = outputs.at(0).mesh_tensor();
    const auto& sink_out = has_sink ? outputs.at(1).mesh_tensor() : bias;
    const uint32_t T = rel_r.logical_shape()[0];
    const uint32_t window_tiles = table.logical_shape()[0] / (2 * TILE_HEIGHT);
    const uint32_t rel_tiles = (params.heads * rel_dim + TILE_WIDTH - 1) / TILE_WIDTH;

    // Core i computes key tile i of every token's window.
    const auto grid = in.rel_r.device()->compute_with_storage_grid_size();
    TT_FATAL(window_tiles <= grid.x * grid.y, "window_bias: {} key tiles exceed the grid", window_tiles);
    const CoreRangeSet cores = num_cores_to_corerangeset(window_tiles, grid, /*row_wise=*/true);
    const auto core_list = corerange_to_cores(cores, window_tiles, /*row_wise=*/true);

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
    constexpr uint32_t cb_positions = 0, cb_rel_r = 1, cb_a = 2, cb_b = 3, cb_sink_in = 4, cb_sink = 5,
                       cb_bias = 16, cb_sink_out = 17;
    make_cb(cb_positions, 1, positions.dtype());
    make_cb(cb_rel_r, rel_tiles, DataType::BFLOAT16);
    make_cb(cb_a, 2, DataType::BFLOAT16);
    make_cb(cb_b, 2, DataType::FLOAT32);
    make_cb(cb_bias, 2, DataType::BFLOAT16);
    if (has_sink) {
        make_cb(cb_sink_in, 1, DataType::BFLOAT16);
        make_cb(cb_sink, 2, DataType::BFLOAT16);
        make_cb(cb_sink_out, 2, DataType::BFLOAT16);
    }

    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_positions", cb_positions},
        {"cb_rel_r", cb_rel_r},
        {"cb_a", cb_a},
        {"cb_b", cb_b},
        {"cb_sink_in", cb_sink_in},
        {"cb_sink", cb_sink},
        {"cb_bias", cb_bias},
        {"cb_sink_out", cb_sink_out},
        {"tokens", T},
        {"heads", params.heads},
        {"rel_tiles", rel_tiles},
        {"window_tiles", window_tiles},
        {"has_sink", has_sink ? 1u : 0u},
        {"sink_scale", std::bit_cast<uint32_t>(params.sink_scale)},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/window_bias/device/kernels/";
    std::vector<uint32_t> reader_args;
    for (const auto* t : {&rel_r, &table, &positions, &sink}) {
        TensorAccessorArgs(*t).append_to(reader_args);
    }
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_window_bias.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = ReaderConfigDescriptor{}};
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(bias).append_to(writer_args);
    TensorAccessorArgs(sink_out).append_to(writer_args);
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_window_bias.cpp",
        .core_ranges = cores,
        .compile_time_args = writer_args,
        .named_compile_time_args = common_args,
        .config = WriterConfigDescriptor{}};
    ComputeConfigDescriptor compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,
        .math_approx_mode = false};
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/window_bias.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = compute_config};

    // The first core also writes the sink tiles.
    for (uint32_t i = 0; i < window_tiles; ++i) {
        const uint32_t sinks = has_sink && i == 0 ? 1u : 0u;
        reader.emplace_runtime_args(
            core_list[i], {i, sinks, std::cref(rel_r), std::cref(table), std::cref(positions), std::cref(sink)});
        writer.emplace_runtime_args(core_list[i], {i, sinks, std::cref(bias), std::cref(sink_out)});
        compute.emplace_runtime_args(core_list[i], {sinks});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

std::optional<std::string> window_bias_reason_of(
    const Tensor& rel_r, const Tensor& table, const Tensor& positions, uint32_t heads) {
    return window_bias_reason(rel_r, table, positions, heads);
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

std::vector<ttnn::Tensor> window_bias(
    const ttnn::Tensor& rel_r,
    const ttnn::Tensor& table,
    const ttnn::Tensor& positions,
    uint32_t heads,
    const std::optional<ttnn::Tensor>& sink,
    float sink_scale) {
    using Op = prim::WindowBiasOperation;
    return device_operation::launch<Op>(
        {.heads = heads, .sink_scale = sink_scale},
        {.rel_r = rel_r, .table = table, .positions = positions, .sink = sink});
}

std::optional<std::string> window_bias_unsupported_reason(
    const ttnn::Tensor& rel_r, const ttnn::Tensor& table, const ttnn::Tensor& positions, uint32_t heads) {
    return prim::window_bias_reason_of(rel_r, table, positions, heads);
}

}  // namespace ttnn::experimental
