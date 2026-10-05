// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/nlp_create_qkv_heads_decode_norm_rope.h"

#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/transformer/qkv_heads_norm_rope_decode/qkv_heads_norm_rope_decode.hpp"

namespace tt::runtime::ttnn::operations::transformer {
void run(const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeNormRopeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  auto tensor = [&](const ::tt::target::ttnn::TensorRef *ref)
      -> const ::ttnn::Tensor & { return pool.getTTNNTensorAndValidate(ref); };
  std::vector<::ttnn::Tensor> outputs =
      ::ttnn::transformer::nlp_create_qkv_heads_decode_norm_rope(
          tensor(op->input()), tensor(op->norm_weight()), tensor(op->cos_cache()),
          tensor(op->sin_cache()), op->num_heads(), op->num_kv_heads(),
          op->epsilon());
  pool.insertTTNNTensorAndValidate(op->q_out(), outputs.at(0));
  pool.insertTTNNTensorAndValidate(op->k_out(), outputs.at(1));
  pool.insertTTNNTensorAndValidate(op->v_out(), outputs.at(2));
}
} // namespace tt::runtime::ttnn::operations::transformer
