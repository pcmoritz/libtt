// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/attention_output_gate.h"

#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/transformer/attention_output_gate/attention_output_gate.hpp"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::AttentionOutputGateOp *op,
         ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      utils::createMemoryConfigIfNeeded(
          utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(memoryConfig, "attention_output_gate writes a device tensor");
  ::ttnn::Tensor out = ::ttnn::transformer::attention_output_gate(
      pool.getTTNNTensorAndValidate(op->attention()),
      pool.getTTNNTensorAndValidate(op->gate()), op->head_dim(),
      op->value_offset(), op->gate_offset(), op->gate_stride(),
      *memoryConfig);
  pool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::transformer
