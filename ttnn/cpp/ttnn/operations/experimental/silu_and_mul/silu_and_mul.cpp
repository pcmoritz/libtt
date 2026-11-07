// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/silu_and_mul/silu_and_mul.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

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

struct SiluAndMulParams {
    MemoryConfig memory_config;
};

struct SiluAndMulInputs {
    Tensor product;
};

struct SiluAndMulOperation {
    using operation_attributes_t = SiluAndMulParams;
    using tensor_args_t = SiluAndMulInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

void SiluAndMulOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    const Tensor& product = in.product;
    TT_FATAL(product.storage_type() == StorageType::DEVICE && product.buffer() != nullptr, "the product must be on device");
    TT_FATAL(
        product.layout() == Layout::TILE && product.tensor_spec().tile() == Tile(), "the product must use 32x32 tiles");
    TT_FATAL(
        product.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "the product must be interleaved, got {}",
        product.memory_config());
    TT_FATAL(product.dtype() == DataType::BFLOAT16, "the product must be BFLOAT16");
    TT_FATAL(
        product.logical_shape()[-1] % (2 * TILE_WIDTH) == 0,
        "the product's last dimension must be twice a multiple of 32, got {}",
        product.logical_shape());
    TT_FATAL(
        params.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "silu_and_mul writes an interleaved output, got {}",
        params.memory_config);
}

SiluAndMulOperation::spec_return_value_t SiluAndMulOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    ttnn::Shape shape = in.product.logical_shape();
    shape[-1] /= 2;
    return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), params.memory_config));
}

SiluAndMulOperation::tensor_return_value_t SiluAndMulOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return create_device_tensor(compute_output_specs(params, in), in.product.device());
}

ProgramDescriptor SiluAndMulOperation::create_descriptor(
    const operation_attributes_t&, const tensor_args_t& in, tensor_return_value_t& output_tensor) {
    const auto& product = in.product.mesh_tensor();
    const auto& output = output_tensor.mesh_tensor();
    auto* device = &product.mutable_device();

    // Work units are output tiles, row-major: tile row u / Nt, column u % Nt.
    const uint32_t Nt = output.padded_shape()[-1] / TILE_WIDTH;
    const uint32_t units = output.padded_shape().volume() / (TILE_HEIGHT * TILE_WIDTH);
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min(units, static_cast<uint32_t>(grid.x * grid.y));
    const CoreRangeSet cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const uint32_t base = units / num_cores;
    const uint32_t extra = units % num_cores;

    ProgramDescriptor program;
    const uint32_t tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    for (uint8_t index : {0, 1, 2}) {  // gate, up and output tiles
        program.cbs.push_back(CBDescriptor{
            .total_size = 2 * tile_bytes,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index, .data_format = tt::DataFormat::Float16_b, .page_size = tile_bytes}}}});
    }
    const std::vector<std::pair<std::string, uint32_t>> common_args = {
        {"cb_gate", 0},
        {"cb_up", 1},
        {"cb_output", 2},
        {"Nt", Nt},
    };
    const std::string kernel_dir = "ttnn/cpp/ttnn/operations/experimental/silu_and_mul/device/kernels/";
    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(product).append_to(reader_args);
    KernelDescriptor reader{
        .kernel_source = kernel_dir + "dataflow/reader_silu_and_mul.cpp",
        .core_ranges = cores,
        .compile_time_args = reader_args,
        .named_compile_time_args = common_args,
        .config = ReaderConfigDescriptor{}};
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(output).append_to(writer_args);
    KernelDescriptor writer{
        .kernel_source = kernel_dir + "dataflow/writer_silu_and_mul.cpp",
        .core_ranges = cores,
        .compile_time_args = writer_args,
        .named_compile_time_args = common_args,
        .config = WriterConfigDescriptor{}};
    // As ttnn.silu and ttnn.multiply run with FP32 accumulation.
    KernelDescriptor compute{
        .kernel_source = kernel_dir + "compute/silu_and_mul.cpp",
        .core_ranges = cores,
        .named_compile_time_args = common_args,
        .config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false}};

    const auto core_list = corerange_to_cores(cores, num_cores, /*row_wise=*/true);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t start = i * base + std::min(i, extra);
        const uint32_t count = base + (i < extra ? 1 : 0);
        reader.emplace_runtime_args(core_list[i], {start, count, product});
        writer.emplace_runtime_args(core_list[i], {start, count, output});
        compute.emplace_runtime_args(core_list[i], {count});
    }
    program.kernels.push_back(std::move(reader));
    program.kernels.push_back(std::move(writer));
    program.kernels.push_back(std::move(compute));
    return program;
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::experimental {

Tensor silu_and_mul(const Tensor& product, const tt::tt_metal::MemoryConfig& memory_config) {
    using Op = prim::SiluAndMulOperation;
    return device_operation::launch<Op>({.memory_config = memory_config}, {.product = product});
}

}  // namespace ttnn::experimental
