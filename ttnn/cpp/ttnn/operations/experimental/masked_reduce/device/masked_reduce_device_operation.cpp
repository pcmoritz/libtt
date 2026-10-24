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

bool masked_reduce_row_form(const Tensor& input, const Tensor& threshold) {
    return threshold.logical_shape().rank() + 1 == input.logical_shape().rank();
}

bool masked_reduce_chunked(const Tensor& input, const Tensor& threshold) {
    return threshold.logical_shape().rank() + 2 == input.logical_shape().rank();
}

uint32_t masked_reduce_matrices(const Tensor& input) {
    const auto& shape = input.logical_shape();
    uint32_t matrices = 1;
    for (int d = 0; d + 2 < static_cast<int>(shape.rank()); ++d) {
        matrices *= shape[d];
    }
    return matrices;
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
    const bool row_form = masked_reduce_row_form(in.input, in.threshold);
    if (masked_reduce_chunked(in.input, in.threshold)) {
        TT_FATAL(
            shape.rank() >= 3 && shape[-1] == tt::constants::TILE_WIDTH,
            "masked_reduce: a [..., H] threshold reduces a [..., H, N, 32] input, not {}",
            shape);
        for (int d = 0; d < static_cast<int>(tshape.rank()); ++d) {
            TT_FATAL(shape[d] == tshape[d], "masked_reduce: threshold dimension {} differs from the input's", d);
        }
    } else {
        TT_FATAL(
            shape.rank() >= 2 && (row_form || shape.rank() == tshape.rank()),
            "masked_reduce: the threshold must be [..., H, 1] or [..., H] for a [..., H, W] input");
        for (int d = 0; d < static_cast<int>(shape.rank()) - 1; ++d) {
            TT_FATAL(shape[d] == tshape[d], "masked_reduce: threshold dimension {} differs from the input's", d);
        }
        TT_FATAL(row_form || tshape[-1] == 1, "masked_reduce: threshold must have one column");
    }
    const bool chunked = masked_reduce_chunked(in.input, in.threshold);
    TT_FATAL(
        chunked || params.mode == MaskedMode::Count || params.mode == MaskedMode::Sum,
        "masked_reduce: Total, Max and ArgFirst take chunked inputs");
    TT_FATAL(
        params.mode == MaskedMode::ArgFirst
            ? params.output_dtype == DataType::INT32
            : params.output_dtype == DataType::FLOAT32 ||
                  (params.output_dtype == DataType::INT32 && params.mode == MaskedMode::Count),
        "masked_reduce: the result is FP32, or INT32 for counts and indices");
    TT_FATAL(params.cores_per_row >= 1, "masked_reduce: no cores");
}

MaskedReduceOperation::spec_return_value_t MaskedReduceOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return {tt::tt_metal::TensorSpec(
        in.threshold.logical_shape(),
        tt::tt_metal::TensorLayout(
            params.output_dtype, tt::tt_metal::PageConfig(Layout::TILE), params.output_mem_config))};
}

MaskedReduceOperation::tensor_return_value_t MaskedReduceOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(params, in)[0], in.input.device())};
}

}  // namespace ttnn::experimental::prim
