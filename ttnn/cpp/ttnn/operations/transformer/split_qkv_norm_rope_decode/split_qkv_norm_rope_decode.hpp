// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::transformer {

// A decode step's attention prologue as PyTorch models trace it, in one
// program. The fused QKV projection qkv [B, (num_q_heads + 2 * num_kv_heads)
// * head_dim] is split into heads, each q head is normalized by rms_norm with
// q_weight and each k head with k_weight, and both are rotated by
// rotary_embedding with token b's row of cos and sin [1, 1 or B, 1,
// head_dim]. Every value is bit for bit that of those ttnn ops, each with
// HiFi4 and FP32 accumulation.
//
// Returns q [1, B, num_q_heads, q_width], zero padded from head_dim, in
// q_memory_config, then for each token b its k and v heads side by side,
// [1, 1, num_kv_heads, 2 * head_dim], in kv_memory_config: in one tensor
// [1, B, num_kv_heads, 2 * head_dim] if batched_key_values.
std::vector<Tensor> split_qkv_norm_rope_decode(
    const Tensor& qkv,
    const Tensor& q_weight,
    const Tensor& k_weight,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    float epsilon,
    uint32_t q_width,
    const tt::tt_metal::MemoryConfig& q_memory_config,
    const tt::tt_metal::MemoryConfig& kv_memory_config,
    bool batched_key_values = false);

}  // namespace ttnn::transformer
