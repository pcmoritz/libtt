// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/dense_rows/dense_rows.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/dense_rows/device/dense_rows_device_operation.hpp"

namespace ttnn::experimental {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

std::optional<DenseRowsDirection> dense_rows_direction(const ttnn::Tensor& input, const ttnn::Shape& shape) {
    using tt::constants::TILE_WIDTH;
    if (input.storage_type() != ttnn::StorageType::DEVICE || input.layout() != Layout::TILE ||
        input.tensor_spec().tile().get_tile_shape() != std::array<uint32_t, 2>{32, 32} ||
        input.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED ||
        !(input.dtype() == DataType::FLOAT32 || input.dtype() == DataType::INT32 ||
          input.dtype() == DataType::UINT32) ||
        input.logical_volume() == 0) {
        return std::nullopt;
    }
    const auto& from = input.logical_shape();
    const auto same_prefix = [](const ttnn::Shape& a, const ttnn::Shape& b, size_t n) {
        for (size_t d = 0; d < n; ++d) {
            if (a[d] != b[d]) {
                return false;
            }
        }
        return true;
    };
    if (shape.rank() == from.rank() + 1 && from.rank() >= 1 && shape[-1] == TILE_WIDTH &&
        from[-1] == shape[-2] * TILE_WIDTH && same_prefix(from, shape, from.rank() - 1)) {
        return DenseRowsDirection::Pack;
    }
    if (from.rank() == shape.rank() + 1 && shape.rank() >= 1 && from[-1] == TILE_WIDTH &&
        shape[-1] == from[-2] * TILE_WIDTH && same_prefix(from, shape, shape.rank() - 1)) {
        return DenseRowsDirection::Unpack;
    }
    return std::nullopt;
}

ttnn::Tensor dense_rows(
    const ttnn::Tensor& input,
    DenseRowsDirection direction,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    using Op = prim::DenseRowsOperation;
    prim::DenseRowsParams params{
        .direction = direction, .output_mem_config = memory_config.value_or(input.memory_config())};
    return ttnn::device_operation::launch<Op>(params, prim::DenseRowsInputs{.input = input})[0];
}

}  // namespace ttnn::experimental
