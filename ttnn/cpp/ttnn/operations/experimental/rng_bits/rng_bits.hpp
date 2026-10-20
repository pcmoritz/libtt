// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

enum class RngAlgorithm : uint32_t { Threefry = 0, Philox = 1 };

// StableHLO's rng_bit_generator with the state on the device: returns
// {bits, new_state}. state is a UINT32 (or INT32) tensor of at least two
// elements along its last dimension, [w0, w1, w2, ...]; the key is (w0, 0), and
// element i of bits (in row-major order) is the first word of
//
//   Threefry-2x32, 20 rounds:  counter (i, w1)
//   Philox-4x32, 10 rounds:    counter (i, w1, w2 or 0, 0)
//
// so the same state gives the same bits. bits is a UINT32 TILE tensor of
// `shape` (rank 1 or more) holding each word's low `width` bits (8, 16 or 32).
// new_state is state with w1 advanced by ceil(elements / 4), modulo 2^32, so
// the next call uses other counters. The kernels read the state on the device,
// so a cached program draws new bits from each new state.
std::vector<ttnn::Tensor> rng_bits(
    const ttnn::Tensor& state,
    const ttnn::Shape& shape,
    RngAlgorithm algorithm,
    uint32_t width = 32,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
