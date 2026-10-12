// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/indexed_rows/indexed_rows.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/indexed_rows/device/indexed_rows_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor rows_gather(
    const ttnn::Tensor& table,
    const ttnn::Tensor& indices,
    const IndexedRowsSpec& spec,
    bool fill_out_of_bounds,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    using Op = prim::RowsGatherOperation;
    prim::IndexedRowsParams params{
        .sizes = spec.sizes,
        .windows = spec.windows,
        .column = spec.column,
        .fill_out_of_bounds = fill_out_of_bounds,
        .output_mem_config = memory_config.value_or(table.memory_config())};
    prim::IndexedRowsInputs inputs{.table = table, .indices = indices};
    // Nothing to gather: the kernels have no work to split.
    if (indices.logical_shape()[2] == 0 || table.logical_volume() == 0) {
        prim::validate_indexed_rows(params, inputs, /*scatter=*/false);
        return Op::create_output_tensors(params, inputs)[0];
    }
    return ttnn::device_operation::launch<Op>(params, inputs)[0];
}

ttnn::Tensor rows_scatter(
    const ttnn::Tensor& table,
    const ttnn::Tensor& indices,
    const ttnn::Tensor& source,
    const IndexedRowsSpec& spec,
    RowsReduce reduce,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    using Op = prim::RowsScatterOperation;
    prim::IndexedRowsParams params{
        .sizes = spec.sizes,
        .windows = spec.windows,
        .column = spec.column,
        .reduce = reduce,
        .output_mem_config = memory_config.value_or(table.memory_config())};
    prim::IndexedRowsInputs inputs{.table = table, .indices = indices, .source = source};
    if (table.logical_volume() == 0) {
        prim::validate_indexed_rows(params, inputs, /*scatter=*/true);
        return Op::create_output_tensors(params, inputs)[0];
    }
    return ttnn::device_operation::launch<Op>(params, inputs)[0];
}

}  // namespace ttnn::experimental
