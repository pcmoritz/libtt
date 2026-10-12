// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/indexed_rows/device/indexed_rows_device_operation.hpp"

#include <cstdint>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

uint32_t indexed_rows_window(const IndexedRowsParams& params) {
    uint32_t window = 1;
    const size_t row_components = params.windows.size() - (params.column ? 1 : 0);
    for (size_t k = 0; k < row_components; ++k) {
        window *= params.windows[k];
    }
    return window;
}

IndexedRowsTable indexed_rows_table(const IndexedRowsParams& params, const Tensor& table) {
    const auto& shape = table.logical_shape();
    IndexedRowsTable t;
    t.B = shape[0];
    t.N = shape[1];
    t.R = shape[-1];
    t.S = 1;
    for (int d = 2; d < static_cast<int>(shape.rank()) - 1; ++d) {
        t.S *= shape[d];
    }
    t.trailing.assign(shape.cbegin() + 2, shape.cend());
    t.column_window = params.column ? params.windows.back() : 0;
    return t;
}

tt::tt_metal::Shape indexed_rows_gathered_shape(
    const IndexedRowsParams& params, const Tensor& table, const Tensor& indices) {
    const IndexedRowsTable t = indexed_rows_table(params, table);
    const uint32_t positions = indices.logical_shape()[2] * indexed_rows_window(params);
    if (params.column) {
        return tt::tt_metal::Shape({t.B, positions * t.column_window});
    }
    ttsl::SmallVector<uint32_t> dims{t.B, positions};
    dims.insert(dims.end(), t.trailing.begin(), t.trailing.end());
    return tt::tt_metal::Shape(dims);
}

void validate_indexed_rows(const IndexedRowsParams& params, const IndexedRowsInputs& in, bool scatter) {
    const char* name = scatter ? "rows_scatter" : "rows_gather";
    auto check_tensor = [&](const Tensor& tensor, const char* what, uint32_t min_rank) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE && tensor.buffer() != nullptr,
            "{}: {} must be an allocated device tensor",
            name,
            what);
        TT_FATAL(tensor.layout() == Layout::ROW_MAJOR, "{}: {} must be row-major", name, what);
        TT_FATAL(
            tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "{}: {} must be interleaved",
            name,
            what);
        TT_FATAL(
            tensor.logical_shape().rank() >= min_rank,
            "{}: {} must have rank {} or more, got {}",
            name,
            what,
            min_rank,
            tensor.logical_shape());
        TT_FATAL(tensor.device() == in.table.device(), "{}: {} must be on the table's device", name, what);
    };
    check_tensor(in.table, "the table", 3);
    check_tensor(in.indices, "the index tensor", 3);
    TT_FATAL(
        in.indices.logical_shape().rank() == 3, "{}: the index tensor must be [Bi, K, M], got {}", name, in.indices.logical_shape());
    TT_FATAL(
        in.indices.dtype() == DataType::INT32 || in.indices.dtype() == DataType::UINT32,
        "{}: the index tensor must be INT32 or UINT32, got {}",
        name,
        in.indices.dtype());

    const IndexedRowsTable t = indexed_rows_table(params, in.table);
    const auto& index_shape = in.indices.logical_shape();
    const uint32_t Bi = index_shape[0], K = params.sizes.size();
    TT_FATAL(Bi == t.B || Bi == 1, "{}: the index tensor has {} batches, the table {}", name, Bi, t.B);
    TT_FATAL(
        K >= 1 && params.windows.size() == K && index_shape[1] == K,
        "{}: {} sizes and {} windows for an index tensor of {} components",
        name,
        K,
        params.windows.size(),
        index_shape[1]);
    // With a column component alone (K == 1), every batch is one row.
    const uint32_t row_components = K - (params.column ? 1 : 0);
    uint64_t rows = 1;
    for (uint32_t k = 0; k < K; ++k) {
        TT_FATAL(
            params.windows[k] >= 1 && params.windows[k] <= params.sizes[k],
            "{}: window {} of component {} must be in [1, {}]",
            name,
            params.windows[k],
            k,
            params.sizes[k]);
        if (k < row_components) {
            rows *= params.sizes[k];
        }
    }
    TT_FATAL(rows == t.N, "{}: the row components span {} rows, the table has {}", name, rows, t.N);
    if (params.column) {
        TT_FATAL(t.S == 1, "{}: a column window needs rows of one page, the table is {}", name, in.table.logical_shape());
        TT_FATAL(
            params.sizes.back() == t.R,
            "{}: the column component spans {} elements, the rows have {}",
            name,
            params.sizes.back(),
            t.R);
    }
    TT_FATAL(
        params.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{}: the output must be interleaved",
        name);

    if (scatter) {
        TT_FATAL(in.source.has_value(), "rows_scatter: a source tensor is required");
        check_tensor(*in.source, "the source tensor", 2);
        TT_FATAL(
            in.source->dtype() == in.table.dtype(),
            "rows_scatter: the source ({}) must have the table's type ({})",
            in.source->dtype(),
            in.table.dtype());
        const tt::tt_metal::Shape want = indexed_rows_gathered_shape(params, in.table, in.indices);
        TT_FATAL(
            in.source->logical_shape() == want,
            "rows_scatter: the source must be {}, got {}",
            want,
            in.source->logical_shape());
        if (params.reduce != RowsReduce::None) {
            const DataType dtype = in.table.dtype();
            TT_FATAL(
                dtype == DataType::FLOAT32 || dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT16 ||
                    dtype == DataType::INT32 || dtype == DataType::UINT32 || dtype == DataType::UINT16 ||
                    dtype == DataType::UINT8 || dtype == DataType::INT8,
                "rows_scatter: cannot reduce {} rows",
                dtype);
        }
    } else {
        TT_FATAL(!in.source.has_value(), "rows_gather takes no source tensor");
    }
}

RowsGatherOperation::program_factory_t RowsGatherOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RowsGatherProgramFactory{};
}

void RowsGatherOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    validate_indexed_rows(params, in, /*scatter=*/false);
}

RowsGatherOperation::spec_return_value_t RowsGatherOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return {tt::tt_metal::TensorSpec(
        indexed_rows_gathered_shape(params, in.table, in.indices),
        tt::tt_metal::TensorLayout(
            in.table.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), params.output_mem_config))};
}

RowsGatherOperation::tensor_return_value_t RowsGatherOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return {ttnn::create_device_tensor(compute_output_specs(params, in)[0], in.table.device())};
}

RowsScatterOperation::program_factory_t RowsScatterOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RowsScatterProgramFactory{};
}

void RowsScatterOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    validate_indexed_rows(params, in, /*scatter=*/true);
}

RowsScatterOperation::spec_return_value_t RowsScatterOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return {tt::tt_metal::TensorSpec(
        in.table.logical_shape(),
        tt::tt_metal::TensorLayout(
            in.table.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), params.output_mem_config))};
}

RowsScatterOperation::tensor_return_value_t RowsScatterOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    return {ttnn::create_device_tensor(compute_output_specs(params, in)[0], in.table.device())};
}

}  // namespace ttnn::experimental::prim
