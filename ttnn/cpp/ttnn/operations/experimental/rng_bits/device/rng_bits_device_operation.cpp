// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/rng_bits/device/rng_bits_device_operation.hpp"

#include <cstdint>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

RngBitsOperation::program_factory_t RngBitsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RngBitsProgramFactory{};
}

void RngBitsOperation::validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& in) {
    const Tensor& state = in.state;
    TT_FATAL(
        state.storage_type() == ttnn::StorageType::DEVICE && state.buffer() != nullptr,
        "rng_bits: the state must be an allocated device tensor");
    TT_FATAL(
        state.device()->arch() == tt::ARCH::BLACKHOLE, "rng_bits is only implemented for Blackhole, got {}",
        state.device()->arch());
    TT_FATAL(
        state.dtype() == DataType::UINT32,
        "rng_bits: the state must be UINT32, got {}",
        state.dtype());
    TT_FATAL(
        state.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "rng_bits: the state must be interleaved");
    // Low/high pairs in the first face of a tile, with optional leading unit dimensions.
    const auto& shape = state.logical_shape();
    TT_FATAL(state.layout() == Layout::TILE, "rng_bits: the state must be tiled");
    TT_FATAL(
        shape.rank() >= 2 && shape[-1] == 2 && shape[-2] == 2 && shape.volume() == 4,
        "rng_bits: expected [2, 2] low/high pairs, got {}",
        shape);
    TT_FATAL(
        state.buffer()->num_pages() == 1, "rng_bits: the state must be one page, got {}", state.buffer()->num_pages());

    TT_FATAL(attrs.shape.rank() >= 1, "rng_bits: the bits need rank 1 or more, got {}", attrs.shape);
    TT_FATAL(attrs.shape.volume() > 0, "rng_bits: the bits must not be empty, got {}", attrs.shape);
    TT_FATAL(
        attrs.shape.volume() < (uint64_t{1} << 32), "rng_bits: the bits must be fewer than 2^32 elements, got {}",
        attrs.shape);
    TT_FATAL(
        attrs.output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "rng_bits: the bits must be interleaved");
}

RngBitsOperation::spec_return_value_t RngBitsOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {
        tt::tt_metal::TensorSpec(
            attrs.shape,
            tt::tt_metal::TensorLayout(
                DataType::UINT32, tt::tt_metal::PageConfig(Layout::TILE), attrs.output_mem_config)),
        in.state.tensor_spec()};
}

RngBitsOperation::tensor_return_value_t RngBitsOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto specs = compute_output_specs(attrs, in);
    return {
        ttnn::create_device_tensor(specs[0], in.state.device()),
        ttnn::create_device_tensor(specs[1], in.state.device())};
}

}  // namespace ttnn::experimental::prim
