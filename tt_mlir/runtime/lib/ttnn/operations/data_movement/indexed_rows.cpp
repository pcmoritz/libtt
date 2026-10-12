// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/indexed_rows.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/indexed_rows/indexed_rows.hpp"

namespace tt::runtime::ttnn::operations::data_movement {
namespace {

::ttnn::experimental::IndexedRowsSpec
toSpec(const ::flatbuffers::Vector<uint32_t> *sizes,
       const ::flatbuffers::Vector<uint32_t> *windows) {
  ::ttnn::experimental::IndexedRowsSpec spec;
  spec.sizes.assign(sizes->begin(), sizes->end());
  spec.windows.assign(windows->begin(), windows->end());
  return spec;
}

::ttnn::experimental::RowsReduce
toReduce(::tt::target::ttnn::ScatterReduceType reduce) {
  switch (reduce) {
  case ::tt::target::ttnn::ScatterReduceType::Sum:
    return ::ttnn::experimental::RowsReduce::Add;
  case ::tt::target::ttnn::ScatterReduceType::Prod:
    return ::ttnn::experimental::RowsReduce::Mul;
  case ::tt::target::ttnn::ScatterReduceType::Max:
    return ::ttnn::experimental::RowsReduce::Max;
  case ::tt::target::ttnn::ScatterReduceType::Min:
    return ::ttnn::experimental::RowsReduce::Min;
  case ::tt::target::ttnn::ScatterReduceType::Invalid:
    return ::ttnn::experimental::RowsReduce::None;
  }
  LOG_FATAL("unsupported rows_scatter reduction");
}

} // namespace

void run(const ::tt::target::ttnn::RowsGatherOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &table =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &indices =
      tensorPool.getTTNNTensorAndValidate(op->indices());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  ::ttnn::Tensor out = ::ttnn::experimental::rows_gather(
      table, indices, toSpec(op->sizes(), op->windows()),
      op->fill_out_of_bounds(), outputMemoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::RowsScatterOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &table =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &indices =
      tensorPool.getTTNNTensorAndValidate(op->indices());
  const ::ttnn::Tensor &source =
      tensorPool.getTTNNTensorAndValidate(op->source());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  ::ttnn::Tensor out = ::ttnn::experimental::rows_scatter(
      table, indices, source, toSpec(op->sizes(), op->windows()),
      toReduce(op->reduce()), outputMemoryConfig);
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
