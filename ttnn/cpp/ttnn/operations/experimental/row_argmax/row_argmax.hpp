// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// The argmax over the last dimension of the rows of a tiled [..., R, N] tensor
// (leading dimensions 1, R <= 32, N a multiple of 32, BFLOAT16 or FLOAT32) as
// UINT32, row-major: [..., R]. Ties go to the smallest index. With a floor, it
// is the argmax of max(x, floor): index 0 where no element exceeds the floor,
// as a reduction with that init value and index 0 gives.
//
// Each core reads and scans only the R rows of its tiles: for a [1, N] FP32
// logits row 128 bytes per tile, where an untilize before argmax reads all of
// every 4 KB tile.
bool is_row_argmax(const ttnn::Tensor& input, int dim);

ttnn::Tensor row_argmax(
    const ttnn::Tensor& input,
    std::optional<float> floor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config);

}  // namespace ttnn::experimental
