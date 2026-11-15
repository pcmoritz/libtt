// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::transformer {

// The output gate of gated attention (Qwen3.5 and Qwen3.8), as PyTorch models
// trace it: each head of the attention output times the sigmoid of its gate
// head,
//
//   output[t, h * D + d] = attention[0, t, h, value_offset + d]
//                          * bf16(sigmoid(fp32(gate[t, h * gate_stride + gate_offset + d])))
//
// in BF16, rounding as the separate typecast, sigmoid, typecast and multiply
// do (sigmoid in FP32, then the BF16 roundings to nearest even). attention is
// [1, T, H, W] (paged SDPA decode's output) and gate [T, N] (the QKV
// projection, each q head followed by its gate head), both BF16, tiled and
// interleaved in DRAM, T <= 32 tokens; D, value_offset, gate_offset and
// gate_stride are multiples of 32. Returns [T, H * D] in memory_config (tiled,
// interleaved). One program; each core makes whole output tiles.
Tensor attention_output_gate(
    const Tensor& attention,
    const Tensor& gate,
    uint32_t head_dim,
    uint32_t value_offset,
    uint32_t gate_offset,
    uint32_t gate_stride,
    const tt::tt_metal::MemoryConfig& memory_config);

}  // namespace ttnn::transformer
