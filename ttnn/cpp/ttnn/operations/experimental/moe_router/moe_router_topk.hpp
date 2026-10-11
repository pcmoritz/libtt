// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <optional>
#include <vector>
#include "ttnn/tensor/tensor.hpp"
namespace ttnn::experimental {
// A mixture-of-experts router's top k: for each row of `logits` ([rows, width],
// rows <= 32, width <= 512), the indices of its k largest values, largest
// first (the lower index first among equal values), and the softmax of those
// k values: {weights, indices}, both [rows, k], in `weights_dtype` (FLOAT32 or
// BFLOAT16) and `indices_dtype` (INT32 or UINT32). With a `mask` ([1, 1] or
// [rows, 1]), every index of a row whose mask is zero is the [1, 1] `fill`.
bool moe_router_topk_supported(const Tensor &logits, uint32_t k,
                               const std::optional<Tensor> &mask,
                               const std::optional<Tensor> &fill);
std::vector<Tensor> moe_router_topk(const Tensor &logits, uint32_t k,
                                    tt::tt_metal::DataType weights_dtype,
                                    tt::tt_metal::DataType indices_dtype,
                                    const std::optional<Tensor> &mask,
                                    const std::optional<Tensor> &fill);
} // namespace ttnn::experimental
