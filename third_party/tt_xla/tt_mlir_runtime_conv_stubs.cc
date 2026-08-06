// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv3d.h"
#include "operations/conv/prepare_conv3d_weights.h"

#include <stdexcept>
#include <string>

namespace tt::runtime::ttnn::operations::conv {
namespace {
[[noreturn]] void unsupported(const char *opName) {
  throw std::runtime_error(std::string(opName) +
                           " is not linked in this libtt build");
}
} // namespace

void run(const ::tt::target::ttnn::Conv3dOp *, ProgramContext &) {
  unsupported("ttnn Conv3dOp");
}

void run(const ::tt::target::ttnn::PrepareConv3dWeightsOp *,
         ProgramContext &) {
  unsupported("ttnn PrepareConv3dWeightsOp");
}
} // namespace tt::runtime::ttnn::operations::conv
