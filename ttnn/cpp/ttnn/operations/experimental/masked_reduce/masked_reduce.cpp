// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/masked_reduce/masked_reduce.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/masked_reduce/device/masked_reduce_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor masked_reduce(
    const ttnn::Tensor& input,
    const ttnn::Tensor& threshold,
    MaskedCompare compare,
    MaskedMode mode,
    tt::tt_metal::DataType dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    using Op = prim::MaskedReduceOperation;
    // The units of work: tile rows, or the [N, 32] matrices of a chunked input.
    const bool chunked = prim::masked_reduce_chunked(input, threshold);
    const uint32_t tile_rows =
        chunked ? prim::masked_reduce_matrices(input) : prim::masked_reduce_tile_rows(input);
    const uint32_t Wt = input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t cores = grid.x * grid.y;
    TT_FATAL(tile_rows >= 1 && tile_rows <= cores, "masked_reduce: {} tile rows for {} cores", tile_rows, cores);
    // Match the reader's dense packing. Otherwise thin, short rows launch
    // workers with no input tiles and still gather their zero partials.
    const uint32_t rows = std::min(input.logical_shape()[-2], tt::constants::TILE_HEIGHT);
    const uint32_t stride = input.dtype() == tt::tt_metal::DataType::BFLOAT16 ? rows + (rows & 1) : rows;
    const uint32_t pack = stride <= tt::constants::TILE_HEIGHT / 2 ? tt::constants::TILE_HEIGHT / stride : 1;
    const uint32_t dense_tiles =
        chunked ? input.padded_shape()[-2] / tt::constants::TILE_HEIGHT : (Wt + pack - 1) / pack;
    prim::MaskedReduceParams params{
        .compare = compare,
        .mode = mode,
        .cores_per_row = std::max(1u, std::min(cores / tile_rows, dense_tiles)),
        .output_dtype = dtype,
        .output_mem_config = memory_config.value_or(input.memory_config())};
    prim::MaskedReduceInputs inputs{.input = input, .threshold = threshold};
    return ttnn::device_operation::launch<Op>(params, inputs)[0];
}

}  // namespace ttnn::experimental
