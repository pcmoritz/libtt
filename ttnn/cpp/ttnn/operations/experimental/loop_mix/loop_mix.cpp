// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/loop_mix/loop_mix.hpp"

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

struct LoopMixParams {
    float epsilon = 1e-6f;
    float eps_norm = 0.0f;
};

struct LoopMixInputs {
    Tensor dest;
    Tensor messages;
    Tensor down;
    Tensor gain;
    Tensor forget;
    std::optional<Tensor> norm_weight;
};

struct LoopMixOperation {
    using operation_attributes_t = LoopMixParams;
    using tensor_args_t = LoopMixInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

constexpr uint32_t max_queries = 3;
constexpr uint32_t max_tokens = 4;
constexpr uint32_t tiles_per_core = 2;
constexpr uint32_t slots_per_gather_tile = 8;
constexpr uint32_t max_cores = 48;

std::optional<std::string> tensor_reason(const Tensor& tensor, const char* name, DataType dtype) {
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
    if (tensor.dtype() != dtype) {
        return fmt::format("{} must be {}", name, dtype);
    }
    return std::nullopt;
}

std::optional<std::string> loop_mix_reason(const LoopMixInputs& in) {
    if (auto r = tensor_reason(in.dest, "dest", DataType::FLOAT32)) {
        return r;
    }
    if (auto r = tensor_reason(in.messages, "messages", DataType::FLOAT32)) {
        return r;
    }
    for (auto [tensor, name] : {std::pair{&in.down, "down"}, {&in.gain, "gain"}, {&in.forget, "forget"}}) {
        if (auto r = tensor_reason(*tensor, name, DataType::BFLOAT16)) {
            return r;
        }
    }
    const auto& m = in.messages.logical_shape();
    const auto& d = in.dest.logical_shape();
    if (m.rank() != 3 || d.rank() != 3) {
        return "dest and messages must be [Q, T, W]";
    }
    const uint32_t Q = m[0], T = m[1], W = m[2];
    if (Q == 0 || Q > max_queries || T == 0 || T > max_tokens || W % (tiles_per_core * TILE_WIDTH) != 0) {
        return fmt::format("messages must be [Q <= 3, T <= 4, W % 64 == 0], got {}", m);
    }
    if ((d[0] != Q && d[0] != 1) || d[1] != T || d[2] != W) {
        return fmt::format("dest must be [{} or 1, {}, {}], got {}", Q, T, W, d);
    }
    if (W / TILE_WIDTH / tiles_per_core > max_cores) {
        return fmt::format("a width of {} needs more than {} cores", W, max_cores);
    }
    const auto& down = in.down.logical_shape();
    if (down.rank() != 3 || down[0] != Q || down[1] != 2 * W || (down[2] != 32 && down[2] != 64)) {
        return fmt::format("down must be [{}, {}, 32 or 64], got {}", Q, 2 * W, down);
    }
    const uint32_t R = down[2];
    for (const auto* t : {&in.gain, &in.forget}) {
        const auto& s = t->logical_shape();
        if (s.rank() != 3 || s[0] != Q || s[1] != R || s[2] != W) {
            return fmt::format("gain and forget must be [{}, {}, {}], got {}", Q, R, W, s);
        }
    }
    if (in.norm_weight.has_value()) {
        const auto& w = *in.norm_weight;
        if (w.storage_type() != StorageType::DEVICE || w.layout() != Layout::TILE ||
            (w.dtype() != DataType::FLOAT32 && w.dtype() != DataType::BFLOAT16) || w.logical_shape().rank() != 2 ||
            w.logical_shape()[0] != Q || w.logical_shape()[1] != W) {
            return fmt::format("the norm weight must be tiled [{}, {}] FLOAT32 or BFLOAT16", Q, W);
        }
    }
    return std::nullopt;
}

void LoopMixOperation::validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t& in) {
    if (auto reason = loop_mix_reason(in)) {
        TT_THROW("loop_mix: {}", *reason);
    }
}

LoopMixOperation::spec_return_value_t LoopMixOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& in) {
    const auto& shape = in.messages.logical_shape();
    if (!in.norm_weight.has_value()) {
        return {TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), in.messages.memory_config()))};
    }
    const TensorSpec row(
        ttnn::Shape({shape[1], shape[2]}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), in.messages.memory_config()));
    return spec_return_value_t(shape[0], row);
}

LoopMixOperation::tensor_return_value_t LoopMixOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(params, in)) {
        outputs.push_back(create_device_tensor(spec, in.messages.device()));
    }
    return outputs;
}

ProgramDescriptor LoopMixOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& output_tensor) {
    const auto& dest = in.dest.mesh_tensor();
    const auto& messages = in.messages.mesh_tensor();
    const auto& down = in.down.mesh_tensor();
    const auto& gain = in.gain.mesh_tensor();
    const auto& forget = in.forget.mesh_tensor();
    const bool has_norm = in.norm_weight.has_value();
    const auto& norm_weight = has_norm ? in.norm_weight->mesh_tensor() : messages;
    auto* device = &messages.mutable_device();

    const auto& shape = messages.logical_shape();
    const uint32_t Q = shape[0], T = shape[1], W = shape[2];
    const uint32_t Qd = dest.logical_shape()[0];
    const uint32_t Wt = W / TILE_WIDTH;
    const uint32_t Rt = down.logical_shape()[2] / TILE_WIDTH;
    const uint32_t num_cores = Wt / tiles_per_core;
    const uint32_t gather1_tiles = (num_cores + slots_per_gather_tile - 1) / slots_per_gather_tile;
    // One gather tile per eight cores and output tile of the down projection.
    const uint32_t gather2_tiles = gather1_tiles * Q * Rt;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    TT_FATAL(num_cores <= grid.x * grid.y, "loop_mix: {} cores do not fit the grid", num_cores);
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    TT_FATAL(cores.ranges().size() <= 2, "loop_mix: the cores must form at most two rectangles");
    const auto core_list = corerange_to_cores(cores, num_cores, /*row_wise=*/true);
    constexpr uint32_t cnt = tiles_per_core;

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
    constexpr uint32_t cb_dest = 0, cb_messages = 1, cb_down = 2, cb_gain = 3, cb_forget = 4, cb_row1 = 5,
                       cb_gather1 = 6, cb_normed = 7, cb_partial = 8, cb_gather2 = 9, cb_hidden = 10,
                       cb_rstd = 11, cb_mixed = 12, cb_row3 = 13, cb_gather3 = 14, cb_norm_weight = 15,
                       cb_output = 16;
    make_cb(cb_dest, Qd * cnt, DataType::FLOAT32);
    make_cb(cb_messages, Q * cnt, DataType::FLOAT32);
    make_cb(cb_down, Q * 2 * cnt * Rt, DataType::BFLOAT16);
    make_cb(cb_gain, Q * Rt * cnt, DataType::BFLOAT16);
    make_cb(cb_forget, Q * Rt * cnt, DataType::BFLOAT16);
    make_cb(cb_row1, 1, DataType::FLOAT32);
    make_cb(cb_gather1, gather1_tiles, DataType::FLOAT32);
    make_cb(cb_normed, (Qd + Q) * cnt, DataType::BFLOAT16);
    make_cb(cb_partial, Q * Rt, DataType::FLOAT32);
    make_cb(cb_gather2, gather2_tiles, DataType::FLOAT32);
    make_cb(cb_hidden, Q * Rt, DataType::BFLOAT16);
    make_cb(cb_rstd, 1, DataType::FLOAT32);
    if (has_norm) {
        make_cb(cb_mixed, Q * cnt, DataType::FLOAT32);
        make_cb(cb_row3, 1, DataType::FLOAT32);
        make_cb(cb_gather3, gather1_tiles, DataType::FLOAT32);
        make_cb(cb_norm_weight, cnt, norm_weight.dtype());
        make_cb(cb_output, Q * cnt, DataType::BFLOAT16);
    } else {
        make_cb(cb_output, cnt, DataType::FLOAT32);
    }

    constexpr uint32_t written_sem = 0, valid_sem = 1;
    for (uint32_t id : {written_sem, valid_sem}) {
        program.semaphores.push_back(SemaphoreDescriptor{.id = id, .core_ranges = cores, .initial_value = 0});
    }

    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_dest", cb_dest},
        {"cb_messages", cb_messages},
        {"cb_down", cb_down},
        {"cb_gain", cb_gain},
        {"cb_forget", cb_forget},
        {"cb_row1", cb_row1},
        {"cb_gather1", cb_gather1},
        {"cb_normed", cb_normed},
        {"cb_partial", cb_partial},
        {"cb_gather2", cb_gather2},
        {"cb_hidden", cb_hidden},
        {"cb_output", cb_output},
        {"cb_rstd", cb_rstd},
        {"cb_mixed", cb_mixed},
        {"cb_row3", cb_row3},
        {"cb_gather3", cb_gather3},
        {"cb_norm_weight", cb_norm_weight},
        {"has_norm", has_norm ? 1u : 0u},
        {"eps_norm", std::bit_cast<uint32_t>(params.eps_norm)},
        {"num_queries", Q},
        {"num_dest", Qd},
        {"Rt", Rt},
        {"tokens", T},
        {"Wt", Wt},
        {"num_cores", num_cores},
        {"gather1_tiles", gather1_tiles},
        {"gather2_tiles", gather2_tiles},
        {"inv_width", std::bit_cast<uint32_t>(1.0f / static_cast<float>(W))},
        {"eps", std::bit_cast<uint32_t>(params.epsilon)},
        {"written_sem", written_sem},
        {"valid_sem", valid_sem},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/loop_mix/device/kernels/";

    std::vector<uint32_t> reader_args;
    for (const auto* t : {&dest, &messages, &down, &gain, &forget, &norm_weight}) {
        TensorAccessorArgs(*t).append_to(reader_args);
    }
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_loop_mix.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}};
    // Every output has the first one's layout.
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(output_tensor.at(0).mesh_tensor()).append_to(writer_args);
    // NOC 0, whose multicast rectangles run from the start to the end
    // coordinates.
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_loop_mix.cpp",
        .core_ranges = cores,
        .compile_time_args = writer_args,
        .named_compile_time_args = common_args,
        .config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}};
    ComputeConfigDescriptor compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .math_approx_mode = false};
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (norm_weight.dtype() == DataType::FLOAT32) {
        compute_config.unpack_to_dest_mode[cb_norm_weight] = UnpackToDestMode::UnpackToDestFp32;
    }
    for (uint32_t cb : {cb_dest, cb_messages, cb_gather1, cb_gather2, cb_rstd, cb_mixed, cb_gather3}) {
        compute_config.unpack_to_dest_mode[cb] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/loop_mix.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = compute_config};

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
        const uint32_t start = i * cnt;
        reader.emplace_runtime_args(
            core_list[i],
            {start,
             std::cref(dest),
             std::cref(messages),
             std::cref(down),
             std::cref(gain),
             std::cref(forget),
             std::cref(norm_weight)});
        std::vector<Arg> w = {i, start, static_cast<uint32_t>(gather.x), static_cast<uint32_t>(gather.y)};
        for (uint32_t value : rectangles) {
            w.emplace_back(value);
        }
        // The output addresses: one, or one per destination.
        for (const auto& out : output_tensor) {
            w.emplace_back(std::cref(out.mesh_tensor()));
        }
        writer.emplace_runtime_args(core_list[i], w);
        compute.emplace_runtime_args(core_list[i], {i == 0 ? 1u : 0u});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

std::optional<std::string> loop_mix_reason_of(const LoopMixInputs& in) { return loop_mix_reason(in); }

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

std::optional<std::string> loop_mix_unsupported_reason(
    const ttnn::Tensor& dest,
    const ttnn::Tensor& messages,
    const ttnn::Tensor& down,
    const ttnn::Tensor& gain,
    const ttnn::Tensor& forget) {
    return prim::loop_mix_reason_of(
        {.dest = dest, .messages = messages, .down = down, .gain = gain, .forget = forget});
}

std::vector<ttnn::Tensor> loop_mix(
    const ttnn::Tensor& dest,
    const ttnn::Tensor& messages,
    const ttnn::Tensor& down,
    const ttnn::Tensor& gain,
    const ttnn::Tensor& forget,
    float epsilon,
    const std::optional<ttnn::Tensor>& norm_weight,
    float eps_norm) {
    using Op = prim::LoopMixOperation;
    return device_operation::launch<Op>(
        {.epsilon = epsilon, .eps_norm = eps_norm},
        {.dest = dest, .messages = messages, .down = down, .gain = gain, .forget = forget, .norm_weight = norm_weight});
}

}  // namespace ttnn::experimental
