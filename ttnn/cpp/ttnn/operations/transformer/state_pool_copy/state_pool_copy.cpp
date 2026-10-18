// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/transformer/state_pool_copy/state_pool_copy.hpp"

#include <algorithm>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device.hpp"
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {

struct StatePoolCopyParams {};

struct StatePoolCopyInputs {
    Tensor state_pool;
    Tensor sources;
    Tensor targets;
};

struct StatePoolCopyOperation {
    using operation_attributes_t = StatePoolCopyParams;
    using tensor_args_t = StatePoolCopyInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    static ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t& inputs) {
        return inputs.state_pool.tensor_spec();
    }
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t& inputs) {
        return inputs.state_pool;
    }
};

void validate_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must be allocated", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must be tiled", name);
    TT_FATAL(!tensor.is_sharded(), "{} must be interleaved", name);
}

void StatePoolCopyOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& inputs) {
    validate_tensor(inputs.state_pool, "state pool");
    validate_tensor(inputs.sources, "sources");
    validate_tensor(inputs.targets, "targets");
    TT_FATAL(
        inputs.sources.dtype() == DataType::INT32 && inputs.targets.dtype() == DataType::INT32,
        "sources and targets must be int32");
    const auto& shape = inputs.sources.logical_shape();
    TT_FATAL(shape.rank() == 1 && shape[0] > 0 && shape[0] <= 32, "sources must be a [N <= 32] vector");
    TT_FATAL(inputs.targets.logical_shape() == shape, "sources and targets must have the same shape");
    const auto& pool_shape = inputs.state_pool.logical_shape();
    TT_FATAL(pool_shape.rank() >= 2, "the state pool must have rank at least two");
    TT_FATAL(
        inputs.state_pool.buffer()->num_pages() % pool_shape[0] == 0,
        "each slot of the state pool must span whole pages");
}

ProgramDescriptor StatePoolCopyOperation::create_descriptor(
    const operation_attributes_t&, const tensor_args_t& inputs, tensor_return_value_t&) {
    const uint32_t count = inputs.sources.logical_shape()[0];
    const auto* pool = inputs.state_pool.buffer();
    const uint32_t pages_per_slot = pool->num_pages() / inputs.state_pool.logical_shape()[0];
    const uint32_t page_bytes = pool->page_size();
    const uint32_t total_pages = count * pages_per_slot;

    ProgramDescriptor program;
    auto* device = inputs.state_pool.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t num_workers = std::min(total_pages, static_cast<uint32_t>(grid.x * grid.y));
    CoreRangeSet cores = num_cores_to_corerangeset(num_workers, grid);

    // Two pages in flight: a read overlaps the previous page's write.
    constexpr uint32_t buffered_pages = 2;
    program.cbs = {
        CBDescriptor{
            .total_size = buffered_pages * page_bytes,
            .core_ranges = cores,
            .format_descriptors = {{
                .buffer_index = tt::CBIndex::c_0,
                .data_format = datatype_to_dataformat_converter(inputs.state_pool.dtype()),
                .page_size = page_bytes}}},
        CBDescriptor{
            .total_size = 2 * 4096,
            .core_ranges = cores,
            .format_descriptors = {{
                .buffer_index = tt::CBIndex::c_1, .data_format = tt::DataFormat::Int32, .page_size = 4096}}}};

    std::vector<uint32_t> compile_args = {count, pages_per_slot, num_workers, page_bytes, buffered_pages};
    TensorAccessorArgs(inputs.state_pool.buffer()).append_to(compile_args);
    TensorAccessorArgs(inputs.sources.buffer()).append_to(compile_args);
    TensorAccessorArgs(inputs.targets.buffer()).append_to(compile_args);

    KernelDescriptor kernel{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/transformer/state_pool_copy/device/kernels/dataflow/state_pool_copy.cpp",
        .core_ranges = cores,
        .compile_time_args = compile_args,
        .config = DataMovementConfigDescriptor{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}};
    for (uint32_t worker = 0; worker < num_workers; ++worker) {
        const CoreCoord core{worker / grid.y, worker % grid.y};
        kernel.emplace_runtime_args(
            core, {worker, inputs.state_pool.buffer(), inputs.sources.buffer(), inputs.targets.buffer()});
    }
    program.kernels.push_back(std::move(kernel));
    return program;
}

}  // namespace
}  // namespace ttnn::prim

namespace ttnn::transformer {

void state_pool_copy(const Tensor& state_pool, const Tensor& sources, const Tensor& targets) {
    using Op = prim::StatePoolCopyOperation;
    device_operation::launch<Op>({}, {.state_pool = state_pool, .sources = sources, .targets = targets});
}

}  // namespace ttnn::transformer
