// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental {

// PyTorch's SiluAndMul of a BF16 [..., 2 N] product, its gate half first, in
// one program:
//
//   output = bf16(silu(fp32(product[..., :N]))) * product[..., N:]
//
// rounding as its ops do on TT (the silu in FP32, then the BF16 roundings to
// nearest even of the typecast and the multiply). The product is tiled and
// interleaved, N a multiple of 32; the output is [..., N] in memory_config.
// Each core makes whole output tiles.
Tensor silu_and_mul(const Tensor& product, const tt::tt_metal::MemoryConfig& memory_config);

}  // namespace ttnn::experimental
