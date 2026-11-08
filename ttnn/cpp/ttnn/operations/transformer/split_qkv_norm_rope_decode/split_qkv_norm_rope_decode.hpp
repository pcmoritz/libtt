// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::transformer {

// A decode step's attention prologue as PyTorch models trace it, in one
// program. The fused QKV projection qkv [B, W] holds num_q_heads q heads, each
// followed by its gate head in gated attention (as W tells), then num_kv_heads
// k heads and num_kv_heads v heads, all head_dim wide, the length of q_weight.
// Each q head is normalized by rms_norm with q_weight and each k head with
// k_weight (BFLOAT16 or FLOAT32), and the first R dims of both are rotated by
// rotary_embedding with token b's row of cos and sin [1, 1 or B, 1, R], the
// rest passing through. Every value is bit for bit that of those ttnn ops,
// each with HiFi4 and FP32 accumulation.
//
// Returns q [1, B, num_q_heads, q_width], zero padded from head_dim, in
// q_memory_config, and every token's k and v heads side by side, [1, B,
// num_kv_heads, 2 * head_dim], in kv_memory_config.
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
    const tt::tt_metal::MemoryConfig& kv_memory_config);

}  // namespace ttnn::transformer
