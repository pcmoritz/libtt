// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/dense_rows/device/dense_rows_device_operation.hpp"

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

using tt::constants::TILE_WIDTH;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

DenseRowsOperation::program_factory_t DenseRowsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return DenseRowsProgramFactory{};
}

void DenseRowsOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    const auto& shape = in.input.logical_shape();
    TT_FATAL(in.input.buffer() != nullptr, "dense_rows: the input must be an allocated device tensor");
    TT_FATAL(
        params.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "dense_rows: the result must be interleaved");
    const auto out_shape = compute_output_specs(params, in)[0].logical_shape();
    TT_FATAL(
        dense_rows_direction(in.input, out_shape) == params.direction,
        "dense_rows: cannot move a {} {} tensor to {}",
        shape,
        in.input.dtype(),
        out_shape);
}

DenseRowsOperation::spec_return_value_t DenseRowsOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const auto& shape = in.input.logical_shape();
    ttsl::SmallVector<uint32_t> dims(shape.cbegin(), shape.cend());
    if (params.direction == DenseRowsDirection::Pack) {
        TT_FATAL(dims.back() % TILE_WIDTH == 0, "dense_rows: {} does not split into rows of 32", shape);
        dims.back() /= TILE_WIDTH;
        dims.push_back(TILE_WIDTH);
    } else {
        TT_FATAL(dims.size() >= 2 && dims.back() == TILE_WIDTH, "dense_rows: {} is not in rows of 32", shape);
        dims.pop_back();
        dims.back() *= TILE_WIDTH;
    }
    return {tt::tt_metal::TensorSpec(
        ttnn::Shape(dims),
        tt::tt_metal::TensorLayout(
            in.input.dtype(), tt::tt_metal::PageConfig(Layout::TILE), params.output_mem_config))};
}

DenseRowsOperation::tensor_return_value_t DenseRowsOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(params, in)[0], in.input.device())};
}

}  // namespace ttnn::experimental::prim
