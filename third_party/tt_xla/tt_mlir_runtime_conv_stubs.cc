// SPDX-License-Identifier: Apache-2.0

#include "operations/conv/conv3d.h"
#include "operations/conv/conv2d.h"
#include "operations/conv/conv_transpose2d.h"
#include "operations/conv/prepare_conv2d_bias.h"
#include "operations/conv/prepare_conv2d_weights.h"
#include "operations/conv/prepare_conv3d_weights.h"
#include "operations/conv/prepare_conv_transpose2d_bias.h"
#include "operations/conv/prepare_conv_transpose2d_weights.h"

#include <stdexcept>
#include <string>

namespace tt::runtime::ttnn::operations::conv {
namespace {
[[noreturn]] void unsupported(const char *opName) {
  throw std::runtime_error(std::string(opName) +
                           " is not linked in this libtt build");
}
} // namespace

void run(const ::tt::target::ttnn::Conv2dOp *, ProgramContext &) {
  unsupported("ttnn Conv2dOp");
}

void run(const ::tt::target::ttnn::Conv3dOp *, ProgramContext &) {
  unsupported("ttnn Conv3dOp");
}

void run(const ::tt::target::ttnn::ConvTranspose2dOp *, ProgramContext &) {
  unsupported("ttnn ConvTranspose2dOp");
}

void run(const ::tt::target::ttnn::PrepareConv2dBiasOp *, ProgramContext &) {
  unsupported("ttnn PrepareConv2dBiasOp");
}

void run(const ::tt::target::ttnn::PrepareConv2dWeightsOp *,
         ProgramContext &) {
  unsupported("ttnn PrepareConv2dWeightsOp");
}

void run(const ::tt::target::ttnn::PrepareConv3dWeightsOp *,
         ProgramContext &) {
  unsupported("ttnn PrepareConv3dWeightsOp");
}

void run(const ::tt::target::ttnn::PrepareConvTranspose2dBiasOp *,
         ProgramContext &) {
  unsupported("ttnn PrepareConvTranspose2dBiasOp");
}

void run(const ::tt::target::ttnn::PrepareConvTranspose2dWeightsOp *,
         ProgramContext &) {
  unsupported("ttnn PrepareConvTranspose2dWeightsOp");
}
} // namespace tt::runtime::ttnn::operations::conv
