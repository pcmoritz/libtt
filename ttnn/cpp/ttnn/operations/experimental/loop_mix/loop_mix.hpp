// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Inkling 2's loop-message controller for up to four tokens, in one program:
// for each destination q (Q <= 3) and token t,
//
//   hidden  = bf16([bf16(rms(dest[q])) | bf16(rms(messages[q]))] @ down[q])
//   gate    = tanh(bf16(hidden @ gain[q]))
//   forget  = tanh(bf16(hidden @ forget[q]))
//   out[q]  = dest[q] + (gate * messages[q] - forget * dest[q])
//
// with rms(x) = x * rsqrt(mean(x^2) + epsilon) over the width W; dest is
// [Qd, T, W] FP32 with Qd = Q or 1 (one destination for all), messages
// [Q, T, W] FP32, down [Q, 2W, R] and gain and forget [Q, R, W] BF16, R 32 or
// 64; the result is [Q, T, W] FP32. Matmuls accumulate BF16 products in FP32
// and round to BF16 as torch.bmm does; the rest is FP32. Tiled, interleaved
// in DRAM, W a multiple of 64.
//
// Each core takes two tile columns of W: its share of the RMS norms' sums of
// squares (gathered by the first core and multicast back), its share of the
// down projection's K (the first core sums the shares and multicasts
// hidden), and then its output columns.
ttnn::Tensor loop_mix(
    const ttnn::Tensor& dest,
    const ttnn::Tensor& messages,
    const ttnn::Tensor& down,
    const ttnn::Tensor& gain,
    const ttnn::Tensor& forget,
    float epsilon);

// Why loop_mix cannot run these operands, if it cannot.
std::optional<std::string> loop_mix_unsupported_reason(
    const ttnn::Tensor& dest,
    const ttnn::Tensor& messages,
    const ttnn::Tensor& down,
    const ttnn::Tensor& gain,
    const ttnn::Tensor& forget);

}  // namespace ttnn::experimental
