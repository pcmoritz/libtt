// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// A mixture-of-experts router's selection for up to four tokens, in one
// program (Inkling 2's and DeepSeek-V3's, without groups):
//
//   scores  = sigmoid(logits)                       [T, E]
//   ids     = top_k(scores + bias)                  [T, k], best first, the
//                                                   lower expert first among equals
//   weights = scores[ids] / sum(scores[ids]) * scale
//
// in FP32. logits are [T, E] FP32 with T <= 4 and E a multiple of 32 up to
// 256, bias [E] (FP32 or BF16), k <= 16. With num_valid (a one-element INT32
// tensor), rows from num_valid on get weight 0 and id -1 (padding). Returns
// {weights [T, k] FP32, ids [T, k] INT32}. One core.
std::vector<ttnn::Tensor> moe_route(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& bias,
    uint32_t k,
    float scale,
    const std::optional<ttnn::Tensor>& num_valid = std::nullopt);

std::optional<std::string> moe_route_unsupported_reason(
    const ttnn::Tensor& logits, const ttnn::Tensor& bias, uint32_t k);

}  // namespace ttnn::experimental
