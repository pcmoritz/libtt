// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/masked_reduce.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/masked_reduce/masked_reduce.hpp"

namespace tt::runtime::ttnn::operations::reduction {
namespace {

using ::ttnn::experimental::MaskedCompare;
using ::ttnn::experimental::MaskedMode;
using ::tt::target::ttnn::MaskedCompareType;
using ::tt::target::ttnn::MaskedModeType;

// The flatbuffer enums list the cases in ttnn's order.
static_assert(static_cast<uint32_t>(MaskedCompare::GT) == static_cast<uint32_t>(MaskedCompareType::GT));
static_assert(static_cast<uint32_t>(MaskedCompare::GE) == static_cast<uint32_t>(MaskedCompareType::GE));
static_assert(static_cast<uint32_t>(MaskedCompare::LT) == static_cast<uint32_t>(MaskedCompareType::LT));
static_assert(static_cast<uint32_t>(MaskedCompare::LE) == static_cast<uint32_t>(MaskedCompareType::LE));
static_assert(static_cast<uint32_t>(MaskedMode::Count) == static_cast<uint32_t>(MaskedModeType::Count));
static_assert(static_cast<uint32_t>(MaskedMode::Sum) == static_cast<uint32_t>(MaskedModeType::Sum));

} // namespace

void run(const ::tt::target::ttnn::MaskedReduceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &threshold =
      tensorPool.getTTNNTensorAndValidate(op->threshold());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  ::ttnn::Tensor out = ::ttnn::experimental::masked_reduce(
      input, threshold, static_cast<MaskedCompare>(op->compare()),
      static_cast<MaskedMode>(op->mode()), outputMemoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::reduction
