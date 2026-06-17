// SPDX-License-Identifier: Apache-2.0

#include "operations/debug/debug.h"

namespace tt::runtime::ttnn::operations::debug {

namespace {

void forwardTensor(const ::tt::target::ttnn::TensorRef *operand,
                   const ::tt::target::ttnn::TensorRef *result,
                   ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &tensor =
      tensorPool.getTTNNTensorAndValidate(operand);
  tensorPool.insertTTNNTensorAndValidate(result, tensor);
}

} // namespace

void run(const ::tt::target::ttnn::AnnotateOp *op, ProgramContext &context) {
  forwardTensor(op->operand(), op->result(), context);
}

void run(const ::tt::target::ttnn::BreakpointOp *, ProgramContext &) {}

void run(const ::tt::target::ttnn::PrintOp *, ProgramContext &) {}

void run(const ::tt::target::ttnn::MemorySnapshotOp *, ProgramContext &) {}

void run(const ::tt::target::ttnn::RegionStartOp *op, ProgramContext &context) {
  forwardTensor(op->operand(), op->result(), context);
}

void run(const ::tt::target::ttnn::RegionEndOp *op, ProgramContext &context) {
  forwardTensor(op->operand(), op->result(), context);
}

} // namespace tt::runtime::ttnn::operations::debug
