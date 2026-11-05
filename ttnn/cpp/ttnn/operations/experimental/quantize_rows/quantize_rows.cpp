// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/quantize_rows/quantize_rows.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/quantize_rows/device/quantize_rows_device_operation.hpp"

namespace ttnn::experimental {

std::vector<ttnn::Tensor> quantize_rows(
    const ttnn::Tensor& input,
    tt::tt_metal::DataType dtype,
    float min_scale,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    using Op = prim::QuantizeRowsOperation;
    prim::QuantizeRowsParams params{
        .dtype = dtype,
        .min_scale = min_scale,
        .output_mem_config = memory_config.value_or(input.memory_config())};
    return ttnn::device_operation::launch<Op>(params, prim::QuantizeRowsInputs{.input = input});
}

}  // namespace ttnn::experimental
