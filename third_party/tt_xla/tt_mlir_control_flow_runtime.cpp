// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/mlir_native/func_call.h"
#include "operations/trace/capture_or_execute_trace.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace tt::runtime::ttnn::operations::mlir_native {
namespace {

// Preserve inputs while nested programs execute. Outputs that alias an input
// can then be retained when they escape to the caller.
class NestedProgramTensorRetention {
public:
  explicit NestedProgramTensorRetention(
      const std::vector<::tt::runtime::Tensor> &tensors)
      : originalTensors(tensors) {
    originalRetain.reserve(tensors.size());
    for (const auto &tensor : tensors) {
      originalRetain.push_back(
          tensor.as<TTNNTensorWrapper>(DeviceRuntime::TTNN).shouldRetain());
    }
    for (auto &tensor : originalTensors) {
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
    // Loop state slots can be replaced by different tensors, including cached
    // constants. Restore flags on the tensors we retained, never on whatever
    // now occupies their old slots.
    for (size_t i = 0; i < originalTensors.size(); ++i) {
      originalTensors[i]
          .as<TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .setRetain(originalRetain[i]);
    }
    restored = true;
  }

private:
  std::vector<::tt::runtime::Tensor> originalTensors;
  std::vector<bool> originalRetain;
  bool restored = false;
};

// A branch or loop body can return one tensor for several results, such as a
// one-step scan whose stacked output is its carry. The caller deallocates each
// result on its own, and the first deallocation would free the buffer the other
// results still name, so a tensor returned more than once is retained. It is
// freed with its last handle.
void retainRepeatedTensors(std::vector<::tt::runtime::Tensor> &tensors) {
  for (size_t i = 1; i < tensors.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (tensors[i].handle.get() == tensors[j].handle.get()) {
        tensors[i].as<TTNNTensorWrapper>(DeviceRuntime::TTNN).setRetain(true);
        break;
      }
    }
  }
}

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
    // Protect the current iteration's values only for the nested call. This
    // also preserves the ownership flags of constants returned by the body.
    NestedProgramTensorRetention iterationRetention(inputs);
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
      // A trace in the body publishes its output slots as they are, and its
      // next replay rewrites them. A carry outlives that replay, so it gets
      // its own copy; a body that forwards a value it computed the previous
      // iteration, (a, b) -> (a + b, a), would otherwise read the sum slot
      // as b after the next iteration overwrote it.
      auto &wrapper =
          bodyOutputs[i].as<TTNNTensorWrapper>(DeviceRuntime::TTNN);
      if (wrapper.isTraceOutput()) {
        bodyOutputs[i] = ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
            ::ttnn::clone(wrapper.getTensor(), std::nullopt, std::nullopt,
                          std::nullopt));
      }
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

  std::vector<::tt::runtime::Tensor> outputs;
  outputs.reserve(op->outputs()->size());
  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    uint32_t stateIndex = op->output_indices()->Get(i);
    LOG_ASSERT(stateIndex < stateSize,
               "While loop output index is out of range");
    outputs.push_back(inputs[stateIndex]);
  }
  retainRepeatedTensors(outputs);
  for (size_t i = 0; i < outputs.size(); ++i) {
    // A loop output may directly alias an input even after the body ran. Keep
    // that shared tensor alive when the caller deallocates the input.
    retention.retainIfInputAlias(outputs[i]);
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }
}

// Whether a branch program is one trace and returns the trace's results in
// order, so that the branch can run as a conditional trace.
static bool isSingleTraceProgram(const ::tt::target::ttnn::Program *program) {
  const ::tt::target::ttnn::CaptureOrExecuteTraceOp *trace = nullptr;
  for (const auto *operation : *program->operations()) {
    switch (operation->type_type()) {
    case ::tt::target::ttnn::OpType::GetDeviceOp:
    case ::tt::target::ttnn::OpType::DeallocateOp:
      break;
    case ::tt::target::ttnn::OpType::CaptureOrExecuteTraceOp:
      if (trace) {
        return false;
      }
      trace = operation->type_as_CaptureOrExecuteTraceOp();
      break;
    default:
      return false;
    }
  }
  if (!trace || trace->outputs()->size() != program->outputs()->size()) {
    return false;
  }
  for (size_t i = 0; i < program->outputs()->size(); ++i) {
    if (trace->outputs()->Get(i)->global_id() !=
        program->outputs()->Get(i)->global_id()) {
      return false;
    }
  }
  return true;
}

// Whether a conditional runs without reading its selector on host: a two-way
// conditional whose selector is on device and whose branches are single
// traces. Each branch then runs as a conditional trace on a predicate computed
// from the selector, so only the branch it selects runs on device, and both
// publish their results in the same slots.
static bool runsAsConditionalTraces(const ::tt::target::ttnn::CaseOp *op,
                                    ProgramContext &context,
                                    const ::ttnn::Tensor &selector) {
  if (selector.storage_type() != ::ttnn::StorageType::DEVICE ||
      op->branch_program_ids()->size() != 2) {
    return false;
  }
  return std::all_of(op->branch_program_ids()->begin(),
                     op->branch_program_ids()->end(), [&](uint32_t programId) {
                       return isSingleTraceProgram(
                           ::tt::runtime::ttnn::utils::getProgram(
                               context.getExecutableHandle(), programId));
                     });
}

// The predicate of the conditional traces of a two-way conditional: 1 for a
// nonzero selector, which selects the second branch, as a uint32 in
// interleaved DRAM, where the prefetcher reads it.
static ::ttnn::Tensor conditionalTracePredicate(const ::ttnn::Tensor &selector) {
  ::ttnn::Tensor predicate = selector;
  if (predicate.layout() != ::ttnn::Layout::TILE) {
    predicate = ::ttnn::to_layout(predicate, ::ttnn::Layout::TILE);
  }
  if (predicate.dtype() != ::ttnn::DataType::FLOAT32) {
    predicate = ::ttnn::typecast(predicate, ::ttnn::DataType::FLOAT32);
  }
  return ::ttnn::typecast(::ttnn::nez(predicate), ::ttnn::DataType::UINT32,
                          ::ttnn::DRAM_MEMORY_CONFIG);
}

void run(const ::tt::target::ttnn::CaseOp *op, ProgramContext &context) {
  LOG_ASSERT(op->branch_program_ids() &&
                 op->branch_program_ids()->size() != 0,
             "Case operation must have at least one branch");

  ::tt::runtime::Tensor index =
      context.getTensorPool().getRuntimeTensorAndValidate(op->index());
  const ::ttnn::Tensor &selector =
      ::tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(index);
  std::optional<::ttnn::Tensor> predicate;
  std::vector<uint32_t> branches;
  if (runsAsConditionalTraces(op, context, selector)) {
    predicate = conditionalTracePredicate(selector);
    branches = {0, 1};
  } else {
    int64_t branchIndex = op->branch_program_ids()->size() == 2
                              ? static_cast<int64_t>(readConditionScalar(index))
                              : readIntegerScalar(index);
    // StableHLO selects the last branch for any out-of-range index, including
    // a negative index.
    size_t selectedBranch = op->branch_program_ids()->size() - 1;
    if (branchIndex >= 0 &&
        static_cast<uint64_t>(branchIndex) < op->branch_program_ids()->size()) {
      selectedBranch = static_cast<size_t>(branchIndex);
    }
    branches = {static_cast<uint32_t>(selectedBranch)};
  }

  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(op->inputs()->size());
  for (const auto *input : *op->inputs()) {
    inputs.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(input));
  }
  NestedProgramTensorRetention retention(inputs);

  std::vector<::tt::runtime::Tensor> outputSlots;
  std::vector<::tt::runtime::Tensor> outputs;
  for (uint32_t branch : branches) {
    std::optional<trace::ConditionalTrace> conditional;
    std::optional<trace::ConditionalTraceScope> scope;
    if (predicate) {
      conditional.emplace(
          trace::ConditionalTrace{*predicate, branch, outputSlots});
      scope.emplace(*conditional);
    }
    ProgramExecutor program(context.getDeviceHandle(),
                            context.getExecutableHandle(),
                            op->branch_program_ids()->Get(branch), inputs,
                            /*constEvalProgram=*/false);
    program.execute();
    outputs = program.gatherOutputTensors();
  }
  LOG_ASSERT(outputs.size() == op->outputs()->size(),
             "Case branch output arity mismatch");
  // Restore original input flags first, then re-pin aliases that escape as
  // outputs. The retention destructor sees `restored` and is therefore a no-op.
  retention.restore();
  retainRepeatedTensors(outputs);
  for (size_t i = 0; i < outputs.size(); ++i) {
    retention.retainIfInputAlias(outputs[i]);
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }
}

} // namespace tt::runtime::ttnn::operations::mlir_native
