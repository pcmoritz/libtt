// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/split_qkv_norm_rope_decode.h"

#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/transformer/split_qkv_norm_rope_decode/split_qkv_norm_rope_decode.hpp"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::SplitQKVNormRopeDecodeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  auto tensor = [&](const ::tt::target::ttnn::TensorRef *ref)
      -> const ::ttnn::Tensor & { return pool.getTTNNTensorAndValidate(ref); };
  auto memoryConfig = [](const ::tt::target::ttnn::TensorRef *ref) {
    std::optional<::ttnn::MemoryConfig> config =
        utils::createMemoryConfigIfNeeded(utils::getTensorRefMemoryConfig(ref));
    LOG_ASSERT(config, "split_qkv_norm_rope_decode writes device tensors");
    return *config;
  };
  // query is [1, B, num_q_heads, q_width].
  const uint32_t qWidth = op->query()->desc()->shape()->Get(3);
  std::vector<::ttnn::Tensor> outputs =
      ::ttnn::transformer::split_qkv_norm_rope_decode(
          tensor(op->input()), tensor(op->q_weight()), tensor(op->k_weight()),
          tensor(op->cos_cache()), tensor(op->sin_cache()), op->num_q_heads(),
          op->num_kv_heads(), op->epsilon(), qWidth,
          memoryConfig(op->query()), memoryConfig(op->key_values()));
  pool.insertTTNNTensorAndValidate(op->query(), outputs.at(0));
  pool.insertTTNNTensorAndValidate(op->key_values(), outputs.at(1));
}
} // namespace tt::runtime::ttnn::operations::transformer
