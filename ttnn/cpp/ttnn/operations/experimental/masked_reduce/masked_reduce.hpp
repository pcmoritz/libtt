// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

enum class MaskedCompare : uint32_t { GT = 0, GE = 1, LT = 2, LE = 3 };
enum class MaskedMode : uint32_t { Count = 0, Sum = 1 };

// Per row of `input` ([..., H, W], tiled, FP32 or BF16), the number of
// elements that compare true against the row's threshold (`threshold` is
// [..., H, 1] in the input's type), or their sum:
//
//   out[..., h, 0] = sum_w (input[..., h, w] <cmp> threshold[..., h, 0]) ? (Count ? 1 : input[..., h, w]) : 0
//
// in FP32, as [..., H, 1]. One pass over the input: every core masks and
// accumulates a run of a tile row's tiles on the compute engine and writes
// one partial tile, and a reduction over the partials finishes the rows.
// This is what a reduction of a compare, or of a select on one, costs with
// one read of the input instead of three passes with two intermediates.
ttnn::Tensor masked_reduce(
    const ttnn::Tensor& input,
    const ttnn::Tensor& threshold,
    MaskedCompare compare,
    MaskedMode mode,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
