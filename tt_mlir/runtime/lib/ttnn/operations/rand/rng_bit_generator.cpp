// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/rand/rng_bit_generator.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/rng_bits/rng_bits.hpp"

namespace tt::runtime::ttnn::operations::rand {

void run(const ::tt::target::ttnn::RngBitGeneratorOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &state =
      tensorPool.getTTNNTensorAndValidate(op->initial_state());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  std::vector<::ttnn::Tensor> outputs = ::ttnn::experimental::rng_bits(
      state, ::tt::runtime::ttnn::operations::utils::toTTNNShape(*op->size()),
      op->philox() ? ::ttnn::experimental::RngAlgorithm::Philox
                   : ::ttnn::experimental::RngAlgorithm::Threefry,
      op->width(), outputMemoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->output_state(), outputs[1]);
  tensorPool.insertTTNNTensorAndValidate(op->out(), outputs[0]);
}

} // namespace tt::runtime::ttnn::operations::rand
