// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// The window bias of a learned relative-position attention bias for the
// causal decode SDPA (its attn_mask), in one program: for token t at position
// p and head h, column j of a window of W = 32 * window_tiles keys that ends
// at p's key tile (the first key 32 * max(0, p / 32 + 1 - window_tiles)) and
// whose distance to p is d = p - first - j,
//
//   bias[t, 0, h, j] = sum_f rel_r[t, h * 16 + f] * table[W - 1 - d, f]
//
// rel_r is [T, heads * 16] BF16, T <= 32, heads <= 32; table [2W, 16] FP32
// holds the relative projection by distance, reversed: row W - 1 - d is
// distance d's 16 values (zero where there is no bias), so each key tile's 32
// columns are 32 consecutive rows. positions are T INT32. Returns bias
// [T, 1, heads, W] BF16 and, with sink ([T, heads] BF16, per-token sinks),
// the SDPA's sink tiles [T, 1, heads, 32] BF16: sink[t, h] * sink_scale in
// column 0 of row h. One core per key tile.
std::vector<ttnn::Tensor> window_bias(
    const ttnn::Tensor& rel_r,
    const ttnn::Tensor& table,
    const ttnn::Tensor& positions,
    uint32_t heads,
    const std::optional<ttnn::Tensor>& sink = std::nullopt,
    float sink_scale = 1.0f);

std::optional<std::string> window_bias_unsupported_reason(
    const ttnn::Tensor& rel_r, const ttnn::Tensor& table, const ttnn::Tensor& positions, uint32_t heads);

}  // namespace ttnn::experimental
