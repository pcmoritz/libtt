// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "operations/reduction/moe_router_topk.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/moe_router/moe_router_topk.hpp"
namespace tt::runtime::ttnn::operations::reduction {
void run(const ::tt::target::ttnn::MoeRouterTopKOp *op,
         ProgramContext &context) {
  ProgramTensorPool &pool = context.getTensorPool();
  // The op reads standard tiles from interleaved memory.
  auto standard = [](::ttnn::Tensor tensor) {
    if (tensor.layout() != ::ttnn::Layout::TILE) {
      tensor = ::ttnn::to_layout(tensor, ::ttnn::Layout::TILE);
    }
    if (tensor.memory_config().is_sharded()) {
      tensor = ::ttnn::to_memory_config(tensor, ::ttnn::DRAM_MEMORY_CONFIG);
    }
    return tensor;
  };
  ::ttnn::Tensor logits = standard(pool.getTTNNTensorAndValidate(op->logits()));
  std::optional<::ttnn::Tensor> mask;
  std::optional<::ttnn::Tensor> fill;
  if (op->mask()) {
    mask = standard(pool.getTTNNTensorAndValidate(op->mask()));
    if (mask->element_size() < 2) {
      mask = ::ttnn::typecast(*mask, ::ttnn::DataType::INT32);
    }
    fill = standard(pool.getTTNNTensorAndValidate(op->fill()));
  }
  auto outputs = ::ttnn::experimental::moe_router_topk(
      logits, op->k(), utils::getDataType(op->weights()),
      utils::getDataType(op->indices()), mask, fill);
  pool.insertTTNNTensorAndValidate(op->weights(), outputs.at(0));
  pool.insertTTNNTensorAndValidate(op->indices(), outputs.at(1));
}
} // namespace tt::runtime::ttnn::operations::reduction
