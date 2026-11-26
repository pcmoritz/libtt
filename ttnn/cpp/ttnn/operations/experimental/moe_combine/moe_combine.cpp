// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/moe_combine/moe_combine.hpp"

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

struct MoeCombineParams {
    DataType dtype = DataType::BFLOAT16;
};

struct MoeCombineInputs {
    Tensor slots;
    Tensor weights;
};

struct MoeCombineOperation {
    using operation_attributes_t = MoeCombineParams;
    using tensor_args_t = MoeCombineInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

bool dram_interleaved(const Tensor& t) {
    return t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr &&
           t.memory_config().buffer_type() == BufferType::DRAM &&
           t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
}

std::optional<std::string> moe_combine_reason(const Tensor& slots, const Tensor& weights) {
    if (!dram_interleaved(slots) || slots.layout() != Layout::ROW_MAJOR || slots.dtype() != DataType::BFLOAT16) {
        return "the slots must be BFLOAT16 row-major and interleaved in DRAM";
    }
    if (!dram_interleaved(weights) || weights.layout() != Layout::TILE || weights.tensor_spec().tile() != Tile() ||
        (weights.dtype() != DataType::BFLOAT16 && weights.dtype() != DataType::FLOAT32)) {
        return "the weights must be tiled BFLOAT16 or FLOAT32 interleaved in DRAM";
    }
    const auto& s = slots.logical_shape();
    const auto& w = weights.logical_shape();
    if (s.rank() != 2 || w.rank() != 2 || w[0] == 0 || w[1] == 0 || s[0] != w[0] * w[1] || s[0] > TILE_HEIGHT ||
        s[1] == 0 || s[1] % TILE_WIDTH != 0) {
        return fmt::format(
            "the slots must be [k * T <= 32, H % 32 == 0] for weights [T, k], got slots {} and weights {}", s, w);
    }
    return std::nullopt;
}

void MoeCombineOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    if (auto reason = moe_combine_reason(in.slots, in.weights)) {
        TT_THROW("moe_combine: {}", *reason);
    }
    TT_FATAL(
        params.dtype == DataType::BFLOAT16 || params.dtype == DataType::FLOAT32,
        "moe_combine: the output must be BFLOAT16 or FLOAT32");
}

MoeCombineOperation::spec_return_value_t MoeCombineOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return TensorSpec(
        ttnn::Shape({in.weights.logical_shape()[0], in.slots.logical_shape()[1]}),
        TensorLayout(params.dtype, PageConfig(Layout::TILE), in.weights.memory_config()));
}

MoeCombineOperation::tensor_return_value_t MoeCombineOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(params, in), in.slots.device());
}

ProgramDescriptor MoeCombineOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& output) {
    const auto& slots = in.slots.mesh_tensor();
    const auto& weights = in.weights.mesh_tensor();
    const auto& out = output.mesh_tensor();
    const uint32_t T = weights.logical_shape()[0];
    const uint32_t k = weights.logical_shape()[1];
    const uint32_t H = slots.logical_shape()[1];
    const uint32_t Ht = H / TILE_WIDTH;

    const auto grid = in.slots.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint32_t>(Ht, grid.x * grid.y);
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
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
    // Weights (reader scratch), the placement, the column's slot rows as
    // they are read (row-major), as a tile, and the output.
    constexpr uint32_t cb_weights = 0, cb_placement = 1, cb_rows = 2, cb_slots = 3, cb_out = 16;
    make_cb(cb_weights, 1, weights.dtype());
    make_cb(cb_placement, 1, DataType::FLOAT32);
    make_cb(cb_rows, 1, DataType::BFLOAT16);
    make_cb(cb_slots, 2, DataType::BFLOAT16);
    make_cb(cb_out, 2, params.dtype);

    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_weights", cb_weights},
        {"cb_placement", cb_placement},
        {"cb_rows", cb_rows},
        {"cb_slots", cb_slots},
        {"cb_out", cb_out},
        {"tokens", T},
        {"experts", k},
        {"slot_rows", T * k},
        {"row_bytes", H * 2},
        {"num_cores", num_cores},
        {"Ht", Ht},
        {"fp32_weights", weights.dtype() == DataType::FLOAT32 ? 1u : 0u},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/moe_combine/device/kernels/";
    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(slots).append_to(reader_args);
    TensorAccessorArgs(weights).append_to(reader_args);
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_moe_combine.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = ReaderConfigDescriptor{}};
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(out).append_to(writer_args);
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_moe_combine.cpp",
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
        .kernel_source = kernel_dir + "compute/moe_combine.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = compute_config};

    for (uint32_t i = 0; i < num_cores; ++i) {
        reader.emplace_runtime_args(core_list[i], {i, std::cref(slots), std::cref(weights)});
        writer.emplace_runtime_args(core_list[i], {i, std::cref(out)});
        compute.emplace_runtime_args(core_list[i], {i});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

std::optional<std::string> moe_combine_reason_of(const Tensor& slots, const Tensor& weights) {
    return moe_combine_reason(slots, weights);
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

ttnn::Tensor moe_combine(const ttnn::Tensor& slots, const ttnn::Tensor& weights, DataType dtype) {
    using Op = prim::MoeCombineOperation;
    return device_operation::launch<Op>({.dtype = dtype}, {.slots = slots, .weights = weights});
}

std::optional<std::string> moe_combine_unsupported_reason(const ttnn::Tensor& slots, const ttnn::Tensor& weights) {
    return prim::moe_combine_reason_of(slots, weights);
}

}  // namespace ttnn::experimental
