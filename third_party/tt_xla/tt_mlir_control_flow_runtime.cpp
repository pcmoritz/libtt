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

// Preserve inputs while nested programs execute. Outputs that alias an input
// can then be retained when they escape to the caller.
class NestedProgramTensorRetention {
public:
  explicit NestedProgramTensorRetention(
      std::vector<::tt::runtime::Tensor> &tensors)
      : tensors(tensors), originalTensors(tensors) {
    originalRetain.reserve(tensors.size());
    for (const auto &tensor : tensors) {
      originalRetain.push_back(
          tensor.as<TTNNTensorWrapper>(DeviceRuntime::TTNN).shouldRetain());
    }
    for (auto &tensor : tensors) {
      retain(tensor);
    }
  }

  ~NestedProgramTensorRetention() { restore(); }

  void retain(::tt::runtime::Tensor &tensor) {
    tensor.as<TTNNTensorWrapper>(DeviceRuntime::TTNN).setRetain(true);
  }

  void retainIfInputAlias(::tt::runtime::Tensor &tensor) {
    if (std::any_of(originalTensors.begin(), originalTensors.end(),
                    [&](const ::tt::runtime::Tensor &input) {
                      return input.handle.get() == tensor.handle.get();
                    })) {
      retain(tensor);
    }
  }

  void restore() {
    if (restored) {
      return;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      tensors[i].as<TTNNTensorWrapper>(DeviceRuntime::TTNN).setRetain(
          originalRetain[i]);
      originalTensors[i]
          .as<TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .setRetain(originalRetain[i]);
    }
    restored = true;
  }

private:
  std::vector<::tt::runtime::Tensor> &tensors;
  std::vector<::tt::runtime::Tensor> originalTensors;
  std::vector<bool> originalRetain;
  bool restored = false;
};

::ttnn::Tensor getHostScalar(const ::tt::runtime::Tensor &tensor) {
  auto hostTensors = ::tt::runtime::ttnn::toHost(
      tensor, /*untilize=*/true, /*blocking=*/true);
  LOG_ASSERT(hostTensors.size() == 1,
             "Control-flow scalar must have one shard");
  return hostTensors.front()
      .as<TTNNTensorWrapper>(DeviceRuntime::TTNN)
      .getTensor();
}

int64_t readIntegerScalar(const ::tt::runtime::Tensor &tensor) {
  const ::ttnn::Tensor hostTensor = getHostScalar(tensor);
  switch (hostTensor.dtype()) {
  case ::ttnn::DataType::INT32:
    return utils::getScalarFromTensor<int32_t>(hostTensor);
  case ::ttnn::DataType::UINT32:
    return static_cast<int64_t>(
        utils::getScalarFromTensor<uint32_t>(hostTensor));
  case ::ttnn::DataType::UINT16:
    return static_cast<int64_t>(
        utils::getScalarFromTensor<uint16_t>(hostTensor));
  case ::ttnn::DataType::UINT8:
    return static_cast<int64_t>(
        utils::getScalarFromTensor<uint8_t>(hostTensor));
  default:
    LOG_FATAL("Unsupported control-flow integer scalar data type");
  }
}

bool readConditionScalar(const ::tt::runtime::Tensor &tensor) {
  const ::ttnn::Tensor hostTensor = getHostScalar(tensor);
  switch (hostTensor.dtype()) {
  case ::ttnn::DataType::INT32:
    return utils::getScalarFromTensor<int32_t>(hostTensor) != 0;
  case ::ttnn::DataType::UINT32:
    return utils::getScalarFromTensor<uint32_t>(hostTensor) != 0;
  case ::ttnn::DataType::UINT16:
    return utils::getScalarFromTensor<uint16_t>(hostTensor) != 0;
  case ::ttnn::DataType::UINT8:
    return utils::getScalarFromTensor<uint8_t>(hostTensor) != 0;
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
    LOG_FATAL("Unsupported loop condition scalar data type");
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
  NestedProgramTensorRetention retention(inputs);

  auto getBound = [&](const ::tt::target::ttnn::LoopBound *bound) {
    LOG_ASSERT(bound, "Counted loop is missing a bound");
    auto index = bound->input_index();
    if (!index) {
      return bound->constant();
    }
    LOG_ASSERT(static_cast<size_t>(*index) < op->inputs()->size(),
               "While loop bound index is out of range");
    return readIntegerScalar(inputs[*index]);
  };

  auto execute = [&](uint32_t programId) {
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
      retention.retain(bodyOutputs[i]);
      inputs[stateIndex] = std::move(bodyOutputs[i]);
    }
  };

  if (op->condition_program_id() >= 0) {
    while (true) {
      std::vector<::tt::runtime::Tensor> conditionOutputs =
          execute(static_cast<uint32_t>(op->condition_program_id()));
      LOG_ASSERT(conditionOutputs.size() == 1,
                 "Loop condition must return one value");
      if (!readConditionScalar(conditionOutputs.front())) {
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

  retention.restore();

  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    uint32_t stateIndex = op->output_indices()->Get(i);
    LOG_ASSERT(stateIndex < stateSize,
               "While loop output index is out of range");
    // A loop output may directly alias an input even after the body ran. Keep
    // that shared tensor alive when the caller deallocates the input.
    retention.retainIfInputAlias(inputs[stateIndex]);
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), inputs[stateIndex]);
  }
}

void run(const ::tt::target::ttnn::CaseOp *op, ProgramContext &context) {
  LOG_ASSERT(op->branch_program_ids() &&
                 op->branch_program_ids()->size() != 0,
             "Case operation must have at least one branch");

  ::tt::runtime::Tensor index =
      context.getTensorPool().getRuntimeTensorAndValidate(op->index());
  int64_t branchIndex = readIntegerScalar(index);
  // StableHLO selects the last branch for any out-of-range index, including a
  // negative index.
  size_t selectedBranch = op->branch_program_ids()->size() - 1;
  if (branchIndex >= 0 &&
      static_cast<uint64_t>(branchIndex) < op->branch_program_ids()->size()) {
    selectedBranch = static_cast<size_t>(branchIndex);
  }

  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(op->inputs()->size());
  for (const auto *input : *op->inputs()) {
    inputs.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(input));
  }
  NestedProgramTensorRetention retention(inputs);

  ProgramExecutor program(
      context.getDeviceHandle(), context.getExecutableHandle(),
      op->branch_program_ids()->Get(selectedBranch), inputs,
      /*constEvalProgram=*/false);
  program.execute();
  std::vector<::tt::runtime::Tensor> outputs = program.gatherOutputTensors();
  LOG_ASSERT(outputs.size() == op->outputs()->size(),
             "Case branch output arity mismatch");
  // Restore original input flags first, then re-pin aliases that escape as
  // outputs. The retention destructor sees `restored` and is therefore a no-op.
  retention.restore();
  for (size_t i = 0; i < outputs.size(); ++i) {
    retention.retainIfInputAlias(outputs[i]);
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }
}

} // namespace tt::runtime::ttnn::operations::mlir_native
