// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// The rows of a tiled [..., R, N] tensor (leading dimensions 1, 2 <= R <= 32,
// N a multiple of 32, 4-byte elements) one after another in one row, as
// `shape` (its volume R * N in the last dimension, the others 1): a row-major
// reshape such as a gathered [4, N] to [1, 4 * N]. Each output tile is row r
// of an input tile, so the op moves 128 bytes per tile where the tiled reshape
// moves every element on its own; the output tiles' padding rows are not
// written.
bool is_rows_to_row(const ttnn::Tensor& input, const ttnn::Shape& shape);

ttnn::Tensor rows_to_row(
    const ttnn::Tensor& input, const ttnn::Shape& shape, const std::optional<tt::tt_metal::MemoryConfig>& memory_config);

}  // namespace ttnn::experimental
