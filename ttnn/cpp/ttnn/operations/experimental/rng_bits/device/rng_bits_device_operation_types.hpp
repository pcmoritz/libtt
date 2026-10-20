// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/rng_bits/rng_bits.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RngBitsParams {
    ttnn::Shape shape;
    RngAlgorithm algorithm;
    uint32_t width;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct RngBitsInputs {
    Tensor state;
};

}  // namespace ttnn::experimental::prim
