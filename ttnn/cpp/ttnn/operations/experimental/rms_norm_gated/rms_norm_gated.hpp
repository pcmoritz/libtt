// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// The gated RMS norm of linear-attention layers (the output of Qwen3.5 and
// Qwen3.8's gated DeltaNet), over each group of D columns of the last
// dimension N (D the weight's size: a head, so N = heads * D):
//
//   output = input * rsqrt(mean(input^2) + epsilon) * weight * silu(gate)
//
// in FP32, rounded once to `dtype` (BF16 or FP32; the input's by default).
// input is [..., N] and gate [..., N_gate] with the same leading dimensions,
// read from column gate_offset on (a multiple of 32), both BF16 or FP32,
// tiled and interleaved; D is a multiple of 32 up to 128 dividing N. One
// program; each core makes whole output tiles, computing their rows' rstd
// itself.
ttnn::Tensor rms_norm_gated(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& weight,
    float epsilon,
    uint32_t gate_offset = 0,
    std::optional<tt::tt_metal::DataType> dtype = std::nullopt);

}  // namespace ttnn::experimental
