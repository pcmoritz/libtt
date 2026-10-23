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
// The state is a tiled UINT32 [2, 2] tensor of two uint64 low/high pairs:
// w0, w1 are the low words and h0, h1 the high words. Element i of bits, in
// row-major order, is the first word of:
//
//   Threefry-2x32-20: key Threefry((w0, h0), (h1, 0x1BD11BDA)), counter (i, w1)
//   Philox-4x32-10:   key (w0, h0), counter (i, w1, 0, h1)
//
// Every state word feeds the bits, so different states draw different streams.
// bits is a UINT32 TILE tensor of `shape` (rank >= 1).
// Scalar outputs arrive from the compiler as [1]. Threefry key derivation
// returns both words and runs once per core, before generating output tiles.
// new_state increments the second uint64 state element by one modulo 2^64.
// The kernels read state on the device, so a cached program draws new bits
// from each new state.
std::vector<ttnn::Tensor> rng_bits(
    const ttnn::Tensor& state,
    const ttnn::Shape& shape,
    RngAlgorithm algorithm,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
