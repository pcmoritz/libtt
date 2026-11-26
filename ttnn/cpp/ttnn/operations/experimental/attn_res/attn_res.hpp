// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Block attention residuals (Inkling 2) for up to four tokens, in one
// program: for each query row q of `queries` and each token t,
//
//   r_i     = rsqrt(mean(s_i[t]^2) + eps_res)
//   p_i     = softmax_i(<s_i[t], queries[q]> * r_i)
//   mix     = sum_i p_i * (normalize ? r_i * s_i[t] : s_i[t])
//   out[q, t] = mix * rsqrt(mean(mix^2) + eps_norm) * weight[q]   (with a weight)
//   out[q, t] = mix                                               (without)
//
// over the sources s_i, the last one plus `addend` when there is one; all
// in FP32, rounded once to `dtype` (BF16 with a weight and FP32 without, by
// default). Sources (1 to 8) and the addend are [T, W] with T <= 4 and W a
// multiple of 32, one dtype for the sources (BF16 or FP32); queries and the
// weight are [Q, W] with Q <= 3, queries scaled by the caller (Inkling scales
// them by W^-0.5). The result is [Q, T, W]. Tiled, interleaved in DRAM.
//
// Each core takes two tile columns of W and computes its share of every
// row's sums of squares and query products; the first core gathers them and
// multicasts them back, and every core finishes the softmax itself. With a
// weight a second exchange gathers the mixture's sums of squares.
ttnn::Tensor attn_res(
    const std::vector<ttnn::Tensor>& sources,
    const std::optional<ttnn::Tensor>& addend,
    const ttnn::Tensor& queries,
    const std::optional<ttnn::Tensor>& weight,
    float eps_res,
    float eps_norm,
    bool normalize,
    std::optional<tt::tt_metal::DataType> dtype = std::nullopt);

// Why attn_res cannot run these operands, if it cannot.
std::optional<std::string> attn_res_unsupported_reason(
    const std::vector<ttnn::Tensor>& sources,
    const std::optional<ttnn::Tensor>& addend,
    const ttnn::Tensor& queries,
    const std::optional<ttnn::Tensor>& weight);

}  // namespace ttnn::experimental
