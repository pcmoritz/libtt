// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// L2 clipping of each group of `group` columns (a head), as Inkling 2 clips
// its queries, keys and relative-position projections:
//
//   v      = input[..., offset : offset + width] * scale
//   output = v * min(1, threshold / sqrt(sum_group(v^2) + epsilon^2)) * row_scale
//            * sigmoid(gate[..., gate_offset : gate_offset + width])
//
// in FP32, rounded once to `dtype` (BF16 or FP32; the input's by default).
// input is [..., N] (columns read from `offset` on, a multiple of 32; width
// defaults to the rest), BF16 or FP32, tiled and interleaved; group is 16 or
// a multiple of 32 up to 128, dividing width. scale (optional) is [width]
// and row_scale (optional) [rows, 1], one value per row of the input's
// leading dimensions; gate (optional) has the input's rows and is read from
// column gate_offset on (a multiple of 32). An infinite threshold skips the
// clip (Inkling's attention output gate). One program; each core makes whole
// output tiles and computes their rows' sums itself.
ttnn::Tensor l2_clip(
    const ttnn::Tensor& input,
    uint32_t group,
    float threshold,
    float epsilon,
    uint32_t offset = 0,
    uint32_t width = 0,
    const std::optional<ttnn::Tensor>& scale = std::nullopt,
    const std::optional<ttnn::Tensor>& row_scale = std::nullopt,
    std::optional<tt::tt_metal::DataType> dtype = std::nullopt,
    const std::optional<ttnn::Tensor>& gate = std::nullopt,
    uint32_t gate_offset = 0);

}  // namespace ttnn::experimental
