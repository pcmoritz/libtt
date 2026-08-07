// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/mlir_native/func_call.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace tt::runtime::ttnn::operations::mlir_native {
namespace {

// A nested body program may normally deallocate its inputs after their last
// local use. Preserve them until the next loop state has been collected.
class ScopedTensorRetention {
public:
  explicit ScopedTensorRetention(std::vector<::tt::runtime::Tensor> &tensors) {
    retained.reserve(tensors.size());
    for (auto &tensor : tensors) {
      auto &wrapper = tensor.as<TTNNTensorWrapper>(DeviceRuntime::TTNN);
      retained.emplace_back(&wrapper, wrapper.shouldRetain());
      wrapper.setRetain(true);
    }
  }

  ~ScopedTensorRetention() {
    for (auto it = retained.rbegin(); it != retained.rend(); ++it) {
      it->first->setRetain(it->second);
    }
  }

private:
  std::vector<std::pair<TTNNTensorWrapper *, bool>> retained;
};

int64_t getScalarInteger(const ::tt::runtime::Tensor &tensor) {
  auto hostTensors = ::tt::runtime::ttnn::toHost(
      tensor, /*untilize=*/true, /*blocking=*/true);
  LOG_ASSERT(hostTensors.size() == 1,
             "Counted loop bounds must have one shard");
  const ::ttnn::Tensor &hostTensor =
      hostTensors.front()
          .as<TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();

  switch (hostTensor.dtype()) {
  case ::ttnn::DataType::INT32:
    return utils::getScalarFromTensor<int32_t>(hostTensor);
  case ::ttnn::DataType::UINT32:
    return static_cast<int32_t>(
        utils::getScalarFromTensor<uint32_t>(hostTensor));
  case ::ttnn::DataType::UINT16:
    return static_cast<int16_t>(
        utils::getScalarFromTensor<uint16_t>(hostTensor));
  case ::ttnn::DataType::UINT8:
    return static_cast<int8_t>(utils::getScalarFromTensor<uint8_t>(hostTensor));
  case ::ttnn::DataType::FLOAT32:
    return utils::getScalarFromTensor<float>(hostTensor) != 0;
  case ::ttnn::DataType::BFLOAT16:
    return static_cast<float>(utils::getScalarFromTensor<bfloat16>(hostTensor)) !=
           0;
  case ::ttnn::DataType::FLOAT16:
    return static_cast<float>(
               utils::getScalarFromTensor<::tt::tt_metal::float16>(hostTensor)) !=
           0;
  default:
    LOG_FATAL("Unsupported loop scalar data type");
  }
}

} // namespace

void run(const ::tt::target::ttnn::WhileLoopOp *op,
         ProgramContext &context) {
  const size_t stateSize = op->state_count();
  LOG_ASSERT(op->inputs()->size() >= stateSize,
             "While loop has fewer inputs than state values");
  LOG_ASSERT(op->output_indices()->size() == op->outputs()->size(),
             "While loop output mapping arity mismatch");

  // The mutable loop state is the prefix; captures remain in the suffix.
  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(op->inputs()->size());
  for (const auto *input : *op->inputs()) {
    inputs.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(input));
  }
  std::vector<const void *> inputHandles;
  inputHandles.reserve(inputs.size());
  for (const auto &tensor : inputs) {
    inputHandles.push_back(tensor.handle.get());
  }

  auto getBound = [&](const ::tt::target::ttnn::LoopBound *bound) {
    LOG_ASSERT(bound, "Counted loop is missing a bound");
    auto index = bound->input_index();
    if (!index) {
      return bound->constant();
    }
    LOG_ASSERT(static_cast<size_t>(*index) < op->inputs()->size(),
               "While loop bound index is out of range");
    return getScalarInteger(inputs[*index]);
  };

  auto execute = [&](uint32_t programId) {
    ScopedTensorRetention retention(inputs);
    ProgramExecutor program(context.getDeviceHandle(),
                            context.getExecutableHandle(), programId, inputs,
                            /*constEvalProgram=*/false);
    program.execute();
    return program.gatherOutputTensors();
  };
  auto executeBody = [&] {
    std::vector<::tt::runtime::Tensor> bodyOutputs = execute(op->program_id());
    LOG_ASSERT(bodyOutputs.size() == op->output_indices()->size(),
               "While loop body output arity mismatch");
    for (size_t i = 0; i < bodyOutputs.size(); ++i) {
      uint32_t stateIndex = op->output_indices()->Get(i);
      LOG_ASSERT(stateIndex < stateSize,
                 "While loop output index is out of range");
      inputs[stateIndex] = std::move(bodyOutputs[i]);
    }
  };

  if (op->condition_program_id() >= 0) {
    while (true) {
      std::vector<::tt::runtime::Tensor> conditionOutputs =
          execute(static_cast<uint32_t>(op->condition_program_id()));
      LOG_ASSERT(conditionOutputs.size() == 1,
                 "Loop condition must return one value");
      if (getScalarInteger(conditionOutputs.front()) == 0) {
        break;
      }
      executeBody();
    }
  } else {
    int64_t initial = getBound(op->initial());
    int64_t limit = getBound(op->limit());
    uint64_t tripCount = 0;
    if (initial < limit) {
      int64_t step = getBound(op->step());
      LOG_ASSERT(step > 0, "Counted loop requires a positive step");
      uint64_t distance =
          static_cast<uint64_t>(limit) - static_cast<uint64_t>(initial);
      tripCount = (distance + static_cast<uint64_t>(step) - 1) /
                  static_cast<uint64_t>(step);
    }
    for (uint64_t iteration = 0; iteration < tripCount; ++iteration) {
      executeBody();
    }
  }

  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    uint32_t stateIndex = op->output_indices()->Get(i);
    LOG_ASSERT(stateIndex < stateSize,
               "While loop output index is out of range");
    // A zero-iteration loop can return an input directly. Keep that shared
    // tensor alive when the caller deallocates the input after this operation.
    if (std::find(inputHandles.begin(), inputHandles.end(),
                  inputs[stateIndex].handle.get()) != inputHandles.end()) {
      inputs[stateIndex]
          .as<TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .setRetain(true);
    }
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), inputs[stateIndex]);
  }
}

} // namespace tt::runtime::ttnn::operations::mlir_native
