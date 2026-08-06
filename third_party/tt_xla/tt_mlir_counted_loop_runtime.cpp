// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/mlir_native/func_call.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

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
  default:
    LOG_FATAL("Unsupported counted loop bound data type");
  }
}

} // namespace

void run(const ::tt::target::ttnn::CountedLoopOp *op,
         ProgramContext &context) {
  const size_t stateSize = op->state_count();
  LOG_ASSERT(op->inputs()->size() >= stateSize,
             "Counted loop has fewer inputs than state values");
  LOG_ASSERT(op->output_indices()->size() == op->outputs()->size(),
             "Counted loop output mapping arity mismatch");

  std::vector<::tt::runtime::Tensor> state;
  state.reserve(stateSize);
  for (size_t i = 0; i < stateSize; ++i) {
    state.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(
            op->inputs()->Get(i)));
  }

  std::vector<::tt::runtime::Tensor> captures;
  captures.reserve(op->inputs()->size() - stateSize);
  for (size_t i = stateSize; i < op->inputs()->size(); ++i) {
    captures.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(
            op->inputs()->Get(i)));
  }

  auto getBound = [&](int32_t index, int64_t constant) {
    if (index < 0) {
      return constant;
    }
    LOG_ASSERT(static_cast<size_t>(index) < state.size(),
               "Counted loop bound index is out of range");
    return getScalarInteger(state[index]);
  };
  int64_t initial = getBound(op->initial_index(), op->initial());
  int64_t limit = getBound(op->limit_index(), op->limit());
  uint64_t tripCount = initial < limit
                           ? static_cast<uint64_t>(limit) -
                                 static_cast<uint64_t>(initial)
                           : 0;

  for (uint64_t iteration = 0; iteration < tripCount; ++iteration) {
    std::vector<::tt::runtime::Tensor> bodyInputs;
    bodyInputs.reserve(state.size() + captures.size());
    bodyInputs.insert(bodyInputs.end(), state.begin(), state.end());
    bodyInputs.insert(bodyInputs.end(), captures.begin(), captures.end());
    ProgramExecutor body(context.getDeviceHandle(),
                         context.getExecutableHandle(), op->program_id(),
                         bodyInputs,
                         /*constEvalProgram=*/false);
    ScopedTensorRetention retention(bodyInputs);
    body.execute();
    std::vector<::tt::runtime::Tensor> bodyOutputs =
        body.gatherOutputTensors();
    LOG_ASSERT(bodyOutputs.size() == op->output_indices()->size(),
               "Counted loop body output arity mismatch");
    for (size_t i = 0; i < bodyOutputs.size(); ++i) {
      uint32_t stateIndex = op->output_indices()->Get(i);
      LOG_ASSERT(stateIndex < state.size(),
                 "Counted loop output index is out of range");
      state[stateIndex] = std::move(bodyOutputs[i]);
    }
  }

  for (size_t i = 0; i < op->outputs()->size(); ++i) {
    uint32_t stateIndex = op->output_indices()->Get(i);
    LOG_ASSERT(stateIndex < state.size(),
               "Counted loop output index is out of range");
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), state[stateIndex]);
  }
}

} // namespace tt::runtime::ttnn::operations::mlir_native
