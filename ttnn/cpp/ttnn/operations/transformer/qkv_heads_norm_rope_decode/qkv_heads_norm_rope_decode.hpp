// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::transformer {

// The attention prologue of a decode step with q and k norms in one program:
// nlp_create_qkv_heads_decode of an interleaved fused QKV projection
// [1, 1, B, (num_q_heads + 2 * num_kv_heads) * head_dim], then RMS
// normalization of every q and k head, with row h of norm_weight
// [num_q_heads + num_kv_heads, head_dim] for q head h and row num_q_heads + h
// for k head h, then the rotary embedding x * cos + rotate_half(x) * sin with
// row 0 of cos and sin, as rotary_embedding with token_index 0 applies it.
// Returns {q, k, v} with the shapes and memory configs of
// nlp_create_qkv_heads_decode.
//
// cos and sin are one row of rotary_dim elements, a multiple of 64 of at most
// head_dim: only the first rotary_dim elements of a head are rotated and the
// rest pass through normalized (partial rotary, as in Qwen3.5). With
// gated_query, each
// q head of qkv is followed by an output gate head of head_dim, which is
// skipped: qkv is [1, 1, B, (2 * num_q_heads + 2 * num_kv_heads) * head_dim].
std::vector<Tensor> nlp_create_qkv_heads_decode_norm_rope(
    const Tensor& qkv,
    const Tensor& norm_weight,
    const Tensor& cos,
    const Tensor& sin,
    uint32_t num_q_heads,
    uint32_t num_kv_heads,
    float epsilon,
    bool gated_query = false);

}  // namespace ttnn::transformer
