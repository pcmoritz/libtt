// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/mlir_native/func_call.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/distributed.hpp>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace tt::runtime::ttnn::operations::trace {
// Whether a trace's run-and-capture program is executing (the uncaptured
// warmup run or the captured run); see capture_or_execute_trace.cpp.
bool inTraceCaptureProgram();
} // namespace tt::runtime::ttnn::operations::trace

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

template <typename ReadScalar>
auto readHostScalar(const ::tt::runtime::Tensor &tensor, ReadScalar read) {
  auto hostTensors = ::tt::runtime::ttnn::toHost(
      tensor, /*untilize=*/true, /*blocking=*/true);
  LOG_ASSERT(!hostTensors.empty(),
             "Control-flow scalar must have a local shard");
  auto value = read(hostTensors.front()
                        .as<TTNNTensorWrapper>(DeviceRuntime::TTNN)
                        .getTensor());
  for (size_t i = 1; i < hostTensors.size(); ++i) {
    LOG_ASSERT(value == read(hostTensors[i]
                                 .as<TTNNTensorWrapper>(DeviceRuntime::TTNN)
                                 .getTensor()),
               "Control-flow scalar replicas must agree across local devices");
  }
  return value;
}

int64_t readIntegerScalar(const ::tt::runtime::Tensor &tensor) {
  return readHostScalar(tensor, [](const ::ttnn::Tensor &hostTensor) -> int64_t {
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
  });
}

bool readConditionScalar(const ::tt::runtime::Tensor &tensor) {
  return readHostScalar(tensor, [](const ::ttnn::Tensor &hostTensor) -> bool {
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
  });
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

// A case whose selector lives on device, inside a trace's run-and-capture
// program. A trace cannot read the selector, so both branch programs run,
// between markers that make the captured trace branch on the selector at
// replay (TraceBranchBegin/Else/End). Each branch copies its results into one
// result buffer, so the case has a single result whichever branch replays.
// The warmup run takes the same path, so both programs compile. The branch
// programs must be free of side effects that the other branch would not
// perform.
void runTraceBranch(const ::tt::target::ttnn::CaseOp *op,
                    ProgramContext &context,
                    const ::tt::runtime::Tensor &index) {
  LOG_ASSERT(op->branch_program_ids()->size() == 2,
             "A trace can branch on a device value between two programs only");
  ::ttnn::MeshDevice &meshDevice =
      context.getDeviceHandle().as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::tt::tt_metal::distributed::MeshCommandQueue &cq =
      meshDevice.mesh_command_queue(0);
  const bool capturing = cq.trace_id().has_value();

  // The prefetcher reads the predicate from page 0 of an interleaved DRAM
  // buffer, as a 32-bit word or its low 16 bits.
  ::ttnn::Tensor predicate =
      index.as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor();
  switch (predicate.dtype()) {
  case ::ttnn::DataType::BFLOAT16:
  case ::ttnn::DataType::UINT16:
  case ::ttnn::DataType::FLOAT16:
  case ::ttnn::DataType::INT32:
  case ::ttnn::DataType::UINT32:
  case ::ttnn::DataType::FLOAT32:
    break;
  default:
    predicate = ::ttnn::typecast(predicate, ::ttnn::DataType::UINT32);
  }
  if (!predicate.memory_config().is_dram() ||
      predicate.memory_config().is_sharded()) {
    predicate = ::ttnn::to_memory_config(
        predicate, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                        ::ttnn::BufferType::DRAM});
  }
  const bool predicateIs16Bit = predicate.dtype() == ::ttnn::DataType::BFLOAT16 ||
                                predicate.dtype() == ::ttnn::DataType::UINT16 ||
                                predicate.dtype() == ::ttnn::DataType::FLOAT16;

  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(op->inputs()->size());
  for (const auto *input : *op->inputs()) {
    inputs.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(input));
  }
  NestedProgramTensorRetention retention(inputs);

  auto runBranch = [&](size_t branch) {
    ProgramExecutor program(
        context.getDeviceHandle(), context.getExecutableHandle(),
        op->branch_program_ids()->Get(branch), inputs,
        /*constEvalProgram=*/false);
    program.execute();
    std::vector<::tt::runtime::Tensor> outputs = program.gatherOutputTensors();
    LOG_ASSERT(outputs.size() == op->outputs()->size(),
               "Case branch output arity mismatch");
    return outputs;
  };

  // The branch for a nonzero predicate first; its outputs give the result
  // buffers their specs.
  if (capturing) {
    ::tt::tt_metal::distributed::TraceBranchBegin(cq, predicate.mesh_buffer(),
                                                  predicateIs16Bit);
  }
  std::vector<::tt::runtime::Tensor> taken = runBranch(1);
  std::vector<::tt::runtime::Tensor> results;
  results.reserve(taken.size());
  for (const ::tt::runtime::Tensor &output : taken) {
    const ::ttnn::Tensor &tensor =
        output.as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor();
    ::ttnn::Tensor result =
        ::ttnn::create_device_tensor(tensor.tensor_spec(), &meshDevice);
    ::ttnn::copy(tensor, result);
    results.push_back(utils::createRuntimeTensorFromTTNN(result));
  }
  taken.clear();

  if (capturing) {
    ::tt::tt_metal::distributed::TraceBranchElse(cq);
  }
  std::vector<::tt::runtime::Tensor> other = runBranch(0);
  for (size_t i = 0; i < other.size(); ++i) {
    ::ttnn::copy(other[i].as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor(),
                 results[i].as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor());
  }
  other.clear();
  if (capturing) {
    ::tt::tt_metal::distributed::TraceBranchEnd(cq);
  }

  retention.restore();
  for (size_t i = 0; i < results.size(); ++i) {
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), results[i]);
  }
}

void run(const ::tt::target::ttnn::CaseOp *op, ProgramContext &context) {
  LOG_ASSERT(op->branch_program_ids() &&
                 op->branch_program_ids()->size() != 0,
             "Case operation must have at least one branch");

  ::tt::runtime::Tensor index =
      context.getTensorPool().getRuntimeTensorAndValidate(op->index());
  if (trace::inTraceCaptureProgram() &&
      index.as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor().storage_type() ==
          ::ttnn::StorageType::DEVICE) {
    runTraceBranch(op, context, index);
    return;
  }
  int64_t branchIndex = op->branch_program_ids()->size() == 2
                            ? static_cast<int64_t>(readConditionScalar(index))
                            : readIntegerScalar(index);
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
