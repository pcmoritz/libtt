// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

enum class RngAlgorithm : uint32_t { Threefry = 0, Philox = 1 };

// StableHLO's rng_bit_generator with device state: returns {bits, new_state}.
// The state is a tiled UINT32 [words, 2] tensor of uint64 low/high pairs
// (2 <= words <= 16). w0, w1, w2 are the low words and h0, h1, h2 the high
// words. Element i of bits, in row-major order, is the first word of:
//
//   Threefry-2x32-20: key (w0 ^ h1, h0), counter (i, w1)
//   Philox-4x32-10:   key (w0, h0 ^ h2), counter (i, w1, w2, h1)
//
// Missing w2/h2 are zero. bits is a UINT32 TILE tensor of `shape` (rank >= 1).
// new_state increments the second uint64 state element by one modulo 2^64.
// The kernels read state on the device, so a cached program draws new bits
// from each new state.
std::vector<ttnn::Tensor> rng_bits(
    const ttnn::Tensor& state,
    const ttnn::Shape& shape,
    RngAlgorithm algorithm,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
