// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/rng_bits/rng_bits.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/rng_bits/device/rng_bits_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> rng_bits(
    const ttnn::Tensor& state,
    const ttnn::Shape& shape,
    RngAlgorithm algorithm,
    uint32_t width,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    using Op = prim::RngBitsOperation;
    return ttnn::device_operation::launch<Op>(
        {.shape = shape,
         .algorithm = algorithm,
         .width = width,
         .output_mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG)},
        {.state = state});
}

}  // namespace ttnn::experimental
