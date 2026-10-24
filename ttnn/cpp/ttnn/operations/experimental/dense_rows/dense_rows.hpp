// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// ttnn::experimental::dense_rows moves a tiled tensor between a [..., H, W]
// shape and the [..., H, W / 32, 32] one that splits its rows into 32-element
// chunks. Both are the same row-major data, but in tiles a [2, W] tensor is
// sixteen times its size in padding, while [2, W / 32, 32] fills its tiles. A
// row of a tile is one 32-element chunk in either shape, so the op copies tile
// rows (two 16-element face rows each) straight from page to page, and fills
// the padding rows of the result's partial tiles with zeros.
#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

enum class DenseRowsDirection : uint32_t {
    // [..., H, W] -> [..., H, W / 32, 32]
    Pack = 0,
    // [..., H, N, 32] -> [..., H, N * 32]
    Unpack = 1,
};

// The direction in which reshaping `input` to `shape` is a dense_rows move, if
// it is one the op supports: an interleaved tiled device tensor of 32-bit
// elements in 32 x 32 tiles.
std::optional<DenseRowsDirection> dense_rows_direction(const ttnn::Tensor& input, const ttnn::Shape& shape);

ttnn::Tensor dense_rows(
    const ttnn::Tensor& input,
    DenseRowsDirection direction,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
