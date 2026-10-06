// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/mlir_native/func_call.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <map>
#include <set>
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

// Loop bodies are captured into a TT-Metal trace and replayed, which replaces
// the host dispatch of every body op with one replay per iteration. Set
// TT_RUNTIME_TRACE_LOOP_BODIES=0 to execute bodies op by op.
bool loopBodyTracingEnabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("TT_RUNTIME_TRACE_LOOP_BODIES");
    return env == nullptr || std::strcmp(env, "0") != 0;
  }();
  return enabled;
}

// Bodies whose captures failed once, keyed by executable and body program.
std::mutex untraceableBodiesMutex;
std::set<std::pair<uint64_t, uint32_t>> untraceableBodies;

// Iterations each condition-driven loop body ran the last time. A capture
// costs about one iteration of dispatch, so bodies of short loops are not
// traced again.
constexpr uint64_t kMinTracedIterations = 8;
std::mutex lastIterationsMutex;
std::map<std::pair<uint64_t, uint32_t>, uint64_t> lastIterations;

std::pair<uint64_t, uint32_t> bodyKey(ProgramContext &context,
                                      uint32_t programId) {
  return {context.getExecutableHandle().id(), programId};
}

// A trace records device commands only: anything that talks to the host
// (transfers, readbacks, nested control flow, CPU-hoisted ops) cannot be
// captured. Bodies with such ops, or with any result outside device memory,
// run op by op.
bool isTraceableBody(ProgramContext &context, uint32_t programId,
                     size_t stateSize) {
  {
    std::lock_guard lock(untraceableBodiesMutex);
    if (untraceableBodies.count(bodyKey(context, programId))) {
      return false;
    }
  }
  using ::tt::target::ttnn::OpType;
  const auto *program =
      utils::getProgram(context.getExecutableHandle(), programId);
  // Body inputs past the loop state are captures, which the loop never
  // changes.
  std::set<uint32_t> captureIds;
  for (size_t i = stateSize; i < program->inputs()->size(); ++i) {
    captureIds.insert(program->inputs()->Get(i)->global_id());
  }
  for (const auto *operation : *program->operations()) {
    switch (operation->type_type()) {
    case OpType::LoadCachedOp: {
      // Cached constants computed from captures are filled by the first
      // iteration and then only looked up. Ones computed from loop state would
      // rerun the constant program (and are keyed by tensor versions, which
      // slot updates do not bump), so such bodies are not traced.
      for (const auto *input : *operation->type_as_LoadCachedOp()->inputs()) {
        if (!captureIds.count(input->global_id())) {
          return false;
        }
      }
      continue;
    }
    case OpType::WhileLoopOp:
    case OpType::WhileOp:
    case OpType::CaseOp:
    case OpType::CpuOp:
    case OpType::FuncCallOp:
    case OpType::FromDeviceOp:
    case OpType::ToDeviceOp:
    case OpType::CaptureOrExecuteTraceOp:
    case OpType::BeginTraceCaptureOp:
    case OpType::EndTraceCaptureOp:
    case OpType::ExecuteTraceOp:
    case OpType::WriteTensorOp:
    case OpType::DumpTensorOp:
    case OpType::LoadTensorOp:
    case OpType::PrintOp:
    case OpType::BreakpointOp:
    case OpType::MemorySnapshotOp:
    case OpType::CreateGlobalSemaphoreOp:
    case OpType::ResetGlobalSemaphoreOp:
    case OpType::AllocateMoeComputeSemaphoreOp:
      return false;
    default:
      break;
    }
    try {
      auto opHandle = ::tt::runtime::utils::unsafeBorrowShared(
          const_cast<::tt::target::ttnn::Operation *>(operation));
      for (const auto &ref :
           getOpOutputRefs(OpContext(opHandle, DeviceRuntime::TTNN))) {
        if (!utils::inDeviceMemory(
                &ref.as<::tt::target::ttnn::TensorRef>(DeviceRuntime::TTNN))) {
          return false;
        }
      }
    } catch (const std::exception &) {
      // Op types the ref helper does not know are treated as untraceable.
      return false;
    }
  }
  return true;
}

bool isDeviceTensor(const ::tt::runtime::Tensor &tensor) {
  return tensor.as<TTNNTensorWrapper>(DeviceRuntime::TTNN)
             .getTensor()
             .storage_type() == ::ttnn::StorageType::DEVICE;
}

// Tensors that share storage share the mesh buffer object.
const void *bufferOf(const ::tt::runtime::Tensor &tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor();
  return ttnnTensor.storage_type() == ::ttnn::StorageType::DEVICE &&
                 ttnnTensor.is_allocated()
             ? &ttnnTensor.mesh_buffer()
             : nullptr;
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

  // After one untraced iteration has populated the program cache, the body is
  // captured once into a trace against persistent per-state slots, and every
  // further iteration is a single replay. The trace ends by copying each new
  // state value into its slot, so the next replay and the condition program
  // read the current state from the same buffers.
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();
  std::optional<::ttnn::MeshTraceId> bodyTrace;
  struct ReleaseTrace {
    ::ttnn::MeshDevice &device;
    std::optional<::ttnn::MeshTraceId> &trace;
    ~ReleaseTrace() {
      if (trace) {
        device.mesh_command_queue(0).finish();
        ::ttnn::operations::trace::release_trace(&device, *trace);
      }
    }
  } releaseTrace{meshDevice, bodyTrace};
  const std::vector<::tt::runtime::Tensor> callerInputs = inputs;
  bool captureAttempted = false;

  std::vector<uint32_t> updated(op->output_indices()->begin(),
                                op->output_indices()->end());

  // Runs the body on the state slots and writes each new state value back into
  // its slot. Results that are themselves state slots (e.g. a body that swaps
  // two values) are read before any slot is overwritten.
  auto runBodyIntoSlots = [&](std::vector<::tt::runtime::Tensor> &bodyOutputs) {
    bodyOutputs = execute(op->program_id());
    LOG_ASSERT(bodyOutputs.size() == updated.size(),
               "While loop body output arity mismatch");
    std::vector<std::pair<::ttnn::Tensor, uint32_t>> writes;
    for (size_t i = 0; i < bodyOutputs.size(); ++i) {
      const void *buffer = bufferOf(bodyOutputs[i]);
      if (buffer == bufferOf(inputs[updated[i]])) {
        continue;
      }
      ::ttnn::Tensor value =
          bodyOutputs[i].as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor();
      if (std::any_of(updated.begin(), updated.end(), [&](uint32_t state) {
            return bufferOf(inputs[state]) == buffer;
          })) {
        value = ::ttnn::clone(value, std::nullopt, std::nullopt, std::nullopt);
      }
      writes.emplace_back(std::move(value), updated[i]);
    }
    for (auto &[value, state] : writes) {
      const ::ttnn::Tensor &slot =
          inputs[state].as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor();
      if (value.layout() != slot.layout() || value.dtype() != slot.dtype() ||
          value.memory_config() != slot.memory_config()) {
        value = ::ttnn::to_layout(value, slot.layout(), slot.dtype(),
                                  slot.memory_config());
      }
      ::ttnn::copy(value, slot);
    }
  };

  // Runs the current iteration untraced against the state slots, which also
  // brings every program the trace will contain into the program cache (a
  // trace cannot load new binaries), then captures the body for the following
  // iterations.
  auto runIterationAndCaptureBody = [&] {
    // Replays write into the updated state buffers, so each must be private
    // to the loop: not shared with another state value, a capture, or a
    // tensor the enclosing program still owns.
    for (uint32_t state : updated) {
      const void *buffer = bufferOf(inputs[state]);
      bool shared = false;
      for (size_t j = 0; j < inputs.size(); ++j) {
        shared |= j != state && bufferOf(inputs[j]) == buffer;
      }
      for (const auto &callerInput : callerInputs) {
        shared |= bufferOf(callerInput) == buffer;
      }
      shared &= buffer != nullptr;
      if (shared) {
        inputs[state] = utils::createRuntimeTensorFromTTNN(::ttnn::clone(
            inputs[state].as<TTNNTensorWrapper>(DeviceRuntime::TTNN).getTensor(),
            std::nullopt, std::nullopt, std::nullopt));
        retention.retain(inputs[state]);
      }
    }

    // If writing a result into its slot is unsupported (e.g. a dtype the copy
    // ops do not handle), hand the results over as plain state instead; only
    // the loop's own slots were written, and they are all replaced.
    std::vector<::tt::runtime::Tensor> bodyOutputs;
    try {
      runBodyIntoSlots(bodyOutputs);
    } catch (const std::exception &error) {
      LOG_ASSERT(bodyOutputs.size() == updated.size(),
                 "While loop body failed: ", error.what());
      for (size_t i = 0; i < bodyOutputs.size(); ++i) {
        retention.retain(bodyOutputs[i]);
        inputs[updated[i]] = std::move(bodyOutputs[i]);
      }
      LOG_WARNING("Running while loop body op by op; cannot update its state "
                  "in place: ",
                  error.what());
      std::lock_guard lock(untraceableBodiesMutex);
      untraceableBodies.insert(bodyKey(context, op->program_id()));
      return;
    }
    bodyOutputs.clear();

    auto &queue = meshDevice.mesh_command_queue(0);
    queue.finish();
    ::ttnn::MeshTraceId trace = ::ttnn::operations::trace::begin_trace_capture(
        &meshDevice, ::ttnn::QueueId(0));
    try {
      runBodyIntoSlots(bodyOutputs);
      bodyOutputs.clear();
      ::ttnn::operations::trace::end_trace_capture(&meshDevice, trace,
                                                   ::ttnn::QueueId(0));
    } catch (const std::exception &error) {
      // Nothing was executed during the capture, so the slots still hold the
      // current state and the loop can continue op by op, provided the queue
      // has left capture mode. Otherwise later commands would be recorded
      // instead of executed, so fail loudly rather than return wrong results.
      try {
        if (queue.trace_id().has_value()) {
          ::ttnn::operations::trace::end_trace_capture(&meshDevice, trace,
                                                       ::ttnn::QueueId(0));
        }
        ::ttnn::operations::trace::release_trace(&meshDevice, trace);
      } catch (const std::exception &) {
      }
      if (queue.trace_id().has_value()) {
        throw;
      }
      LOG_WARNING("Running while loop body op by op; trace capture failed: ",
                  error.what());
      std::lock_guard lock(untraceableBodiesMutex);
      untraceableBodies.insert(bodyKey(context, op->program_id()));
      return;
    }
    bodyTrace = trace;
  };

  // `completed` iterations have run; `remaining` is known for counted loops.
  auto runIteration = [&](uint64_t completed,
                          std::optional<uint64_t> remaining) {
    // Decide once, after the first iteration: tracing pays off when at least
    // two replays follow the capture iteration.
    if (!captureAttempted && completed >= 1) {
      captureAttempted = true;
      std::optional<uint64_t> previous;
      if (!remaining) {
        std::lock_guard lock(lastIterationsMutex);
        auto it = lastIterations.find(bodyKey(context, op->program_id()));
        if (it != lastIterations.end()) {
          previous = it->second;
        }
      }
      if (remaining.value_or(3) >= 3 &&
          previous.value_or(kMinTracedIterations) >= kMinTracedIterations &&
          loopBodyTracingEnabled() &&
          meshDevice.get_program_cache().is_enabled() &&
          std::all_of(inputs.begin(), inputs.end(), isDeviceTensor) &&
          isTraceableBody(context, op->program_id(), stateSize)) {
        runIterationAndCaptureBody();
        return;
      }
    }
    if (bodyTrace) {
      ::ttnn::operations::trace::execute_trace(&meshDevice, *bodyTrace,
                                               ::ttnn::QueueId(0),
                                               /*blocking=*/false);
      return;
    }
    executeBody();
  };

  if (op->condition_program_id() >= 0) {
    for (uint64_t completed = 0;; ++completed) {
      std::vector<::tt::runtime::Tensor> conditionOutputs =
          execute(static_cast<uint32_t>(op->condition_program_id()));
      LOG_ASSERT(conditionOutputs.size() == 1,
                 "Loop condition must return one value");
      if (!readConditionScalar(conditionOutputs.front())) {
        std::lock_guard lock(lastIterationsMutex);
        lastIterations[bodyKey(context, op->program_id())] = completed;
        break;
      }
      runIteration(completed, std::nullopt);
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
      runIteration(iteration, tripCount - iteration);
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
