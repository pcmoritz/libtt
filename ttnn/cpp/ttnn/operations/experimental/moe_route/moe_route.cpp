// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/moe_route/moe_route.hpp"

#include <bit>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {
namespace {

struct MoeRouteParams {
    uint32_t k = 8;
    float scale = 1.0f;
};

struct MoeRouteInputs {
    Tensor logits;
    Tensor bias;
    std::optional<Tensor> num_valid;
};

struct MoeRouteOperation {
    using operation_attributes_t = MoeRouteParams;
    using tensor_args_t = MoeRouteInputs;
    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

constexpr uint32_t max_tokens = 4;
constexpr uint32_t max_experts = 256;
constexpr uint32_t max_k = 16;

bool tiled_dram(const Tensor& t) {
    return t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr && t.layout() == Layout::TILE &&
           t.tensor_spec().tile() == Tile() && t.memory_config().buffer_type() == BufferType::DRAM &&
           t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED;
}

std::optional<std::string> moe_route_reason(const Tensor& logits, const Tensor& bias, uint32_t k) {
    if (!tiled_dram(logits) || !tiled_dram(bias)) {
        return "logits and bias must be tiled and interleaved in DRAM";
    }
    if (logits.dtype() != DataType::FLOAT32) {
        return "logits must be FLOAT32";
    }
    if (bias.dtype() != DataType::FLOAT32 && bias.dtype() != DataType::BFLOAT16) {
        return "the bias must be FLOAT32 or BFLOAT16";
    }
    const auto& shape = logits.logical_shape();
    const uint32_t E = shape[-1];
    const uint32_t T = shape.volume() / E;
    if (shape.rank() != 2 || T == 0 || T > max_tokens || E % TILE_WIDTH != 0 || E > max_experts) {
        return fmt::format("logits must be [T <= 4, E % 32 == 0, E <= 256], got {}", shape);
    }
    if (bias.logical_shape().volume() != E || bias.logical_shape()[-1] != E) {
        return fmt::format("the bias must be [{}], got {}", E, bias.logical_shape());
    }
    if (k == 0 || k > max_k || k > E) {
        return fmt::format("k must be 1 to {}, got {}", max_k, k);
    }
    return std::nullopt;
}

void MoeRouteOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    if (auto reason = moe_route_reason(in.logits, in.bias, params.k)) {
        TT_THROW("moe_route: {}", *reason);
    }
    if (in.num_valid.has_value()) {
        TT_FATAL(
            tiled_dram(*in.num_valid) && in.num_valid->logical_shape().volume() == 1 &&
                (in.num_valid->dtype() == DataType::INT32 || in.num_valid->dtype() == DataType::UINT32),
            "moe_route: num_valid must be one tiled INT32 element");
    }
}

MoeRouteOperation::spec_return_value_t MoeRouteOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const uint32_t E = in.logits.logical_shape()[-1];
    const uint32_t T = in.logits.logical_shape().volume() / E;
    const ttnn::Shape shape({T, params.k});
    auto spec = [&](DataType dtype) {
        return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), in.logits.memory_config()));
    };
    return {spec(DataType::FLOAT32), spec(DataType::INT32)};
}

MoeRouteOperation::tensor_return_value_t MoeRouteOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(params, in)) {
        outputs.push_back(create_device_tensor(spec, in.logits.device()));
    }
    return outputs;
}

ProgramDescriptor MoeRouteOperation::create_descriptor(
    const operation_attributes_t& params, const tensor_args_t& in, tensor_return_value_t& outputs) {
    const auto& logits = in.logits.mesh_tensor();
    const auto& bias = in.bias.mesh_tensor();
    const auto& num_valid = in.num_valid.has_value() ? in.num_valid->mesh_tensor() : logits;
    const auto& weights = outputs.at(0).mesh_tensor();
    const auto& ids = outputs.at(1).mesh_tensor();
    const uint32_t E = logits.logical_shape()[-1];
    const uint32_t T = logits.logical_shape().volume() / E;
    const uint32_t Et = E / TILE_WIDTH;
    const CoreRangeSet cores(CoreRange({0, 0}, {0, 0}));

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
    constexpr uint32_t cb_logits = 0, cb_bias = 1, cb_num_valid = 2, cb_weights = 16, cb_ids = 17;
    make_cb(cb_logits, Et, DataType::FLOAT32);
    make_cb(cb_bias, Et, bias.dtype());
    make_cb(cb_num_valid, 1, num_valid.dtype());
    make_cb(cb_weights, 1, DataType::FLOAT32);
    make_cb(cb_ids, 1, DataType::INT32);

    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_logits", cb_logits},
        {"cb_bias", cb_bias},
        {"cb_num_valid", cb_num_valid},
        {"cb_weights", cb_weights},
        {"cb_ids", cb_ids},
        {"Et", Et},
        {"k", params.k},
        {"tokens", T},
        {"has_num_valid", in.num_valid.has_value() ? 1u : 0u},
        {"scale", std::bit_cast<uint32_t>(params.scale)},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/moe_route/device/kernels/";
    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(logits).append_to(reader_args);
    TensorAccessorArgs(bias).append_to(reader_args);
    TensorAccessorArgs(num_valid).append_to(reader_args);
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_moe_route.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = ReaderConfigDescriptor{}};
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(weights).append_to(writer_args);
    TensorAccessorArgs(ids).append_to(writer_args);
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_moe_route.cpp",
        .core_ranges = cores,
        .compile_time_args = writer_args,
        .named_compile_time_args = common_args,
        .config = WriterConfigDescriptor{}};
    ComputeConfigDescriptor compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = true,
        .math_approx_mode = false};
    compute_config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    compute_config.unpack_to_dest_mode[cb_logits] = UnpackToDestMode::UnpackToDestFp32;
    if (bias.dtype() == DataType::FLOAT32) {
        compute_config.unpack_to_dest_mode[cb_bias] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/moe_route.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = compute_config};

    const CoreCoord core(0, 0);
    reader.emplace_runtime_args(core, {logits, bias, num_valid});
    writer.emplace_runtime_args(core, {weights, ids});
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

std::optional<std::string> moe_route_reason_of(const Tensor& logits, const Tensor& bias, uint32_t k) {
    return moe_route_reason(logits, bias, k);
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

std::vector<ttnn::Tensor> moe_route(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& bias,
    uint32_t k,
    float scale,
    const std::optional<ttnn::Tensor>& num_valid) {
    using Op = prim::MoeRouteOperation;
    return device_operation::launch<Op>({.k = k, .scale = scale}, {.logits = logits, .bias = bias, .num_valid = num_valid});
}

std::optional<std::string> moe_route_unsupported_reason(
    const ttnn::Tensor& logits, const ttnn::Tensor& bias, uint32_t k) {
    return prim::moe_route_reason_of(logits, bias, k);
}

}  // namespace ttnn::experimental
