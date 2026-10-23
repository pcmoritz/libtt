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
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    using Op = prim::MaskedReduceOperation;
    const uint32_t tile_rows = prim::masked_reduce_tile_rows(input);
    const uint32_t Wt = input.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t cores = grid.x * grid.y;
    TT_FATAL(tile_rows >= 1 && tile_rows <= cores, "masked_reduce: {} tile rows for {} cores", tile_rows, cores);
    prim::MaskedReduceParams params{
        .compare = compare,
        .mode = mode,
        .cores_per_row = std::max(1u, std::min(cores / tile_rows, Wt)),
        .output_mem_config = memory_config.value_or(input.memory_config())};
    prim::MaskedReduceInputs inputs{.input = input, .threshold = threshold};
    return ttnn::device_operation::launch<Op>(params, inputs)[0];
}

}  // namespace ttnn::experimental
