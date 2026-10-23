// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/masked_reduce/device/masked_reduce_device_operation.hpp"

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

// Every core owns a 16-byte slot per row of its master's partials (its row
// sums, in the first of four columns), so a tile row of C cores spans 4 * C
// columns, in whole tiles.
uint32_t masked_reduce_partial_columns(uint32_t cores_per_row) {
    return tt::round_up(4 * cores_per_row, tt::constants::TILE_WIDTH);
}

uint32_t masked_reduce_tile_rows(const Tensor& input) {
    const auto& padded = input.padded_shape();
    uint32_t rows = 1;
    for (int d = 0; d < static_cast<int>(padded.rank()) - 1; ++d) {
        rows *= padded[d];
    }
    return rows / tt::constants::TILE_HEIGHT;
}

MaskedReduceOperation::program_factory_t MaskedReduceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MaskedReduceProgramFactory{};
}

void MaskedReduceOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    auto check = [&](const Tensor& tensor, const char* what) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE && tensor.buffer() != nullptr,
            "masked_reduce: {} must be an allocated device tensor",
            what);
        TT_FATAL(tensor.layout() == Layout::TILE, "masked_reduce: {} must be tiled", what);
        TT_FATAL(
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "masked_reduce: {} must be interleaved",
            what);
        TT_FATAL(
            tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
            "masked_reduce: {} must be FP32 or BF16",
            what);
    };
    check(in.input, "input");
    check(in.threshold, "threshold");
    TT_FATAL(in.input.dtype() == in.threshold.dtype(), "masked_reduce: input and threshold must have one data type");
    const auto& shape = in.input.logical_shape();
    const auto& tshape = in.threshold.logical_shape();
    TT_FATAL(shape.rank() >= 2 && shape.rank() == tshape.rank(), "masked_reduce: input and threshold ranks differ");
    for (int d = 0; d < static_cast<int>(shape.rank()) - 1; ++d) {
        TT_FATAL(shape[d] == tshape[d], "masked_reduce: threshold dimension {} differs from the input's", d);
    }
    TT_FATAL(tshape[-1] == 1, "masked_reduce: threshold must have one column");
    TT_FATAL(params.cores_per_row >= 1, "masked_reduce: no cores");
}

MaskedReduceOperation::spec_return_value_t MaskedReduceOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const auto& shape = in.input.logical_shape();
    ttsl::SmallVector<uint32_t> dims(shape.cbegin(), shape.cend());
    dims.back() = 1;
    return {tt::tt_metal::TensorSpec(
        ttnn::Shape(dims),
        tt::tt_metal::TensorLayout(
            DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), params.output_mem_config))};
}

MaskedReduceOperation::tensor_return_value_t MaskedReduceOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(params, in)[0], in.input.device())};
}

}  // namespace ttnn::experimental::prim
