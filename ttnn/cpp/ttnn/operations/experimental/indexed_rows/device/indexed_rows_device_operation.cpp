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
    for (uint32_t w : params.windows) {
        window *= w;
    }
    return window;
}

void validate_indexed_rows(const IndexedRowsParams& params, const IndexedRowsInputs& in, bool scatter) {
    const char* name = scatter ? "rows_scatter" : "rows_gather";
    auto check_tensor = [&](const Tensor& tensor, const char* what) {
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
        TT_FATAL(tensor.logical_shape().rank() == 3, "{}: {} must have rank 3, got {}", name, what, tensor.logical_shape());
        TT_FATAL(tensor.device() == in.table.device(), "{}: {} must be on the table's device", name, what);
    };
    check_tensor(in.table, "the table");
    check_tensor(in.indices, "the index tensor");
    TT_FATAL(
        in.indices.dtype() == DataType::INT32 || in.indices.dtype() == DataType::UINT32,
        "{}: the index tensor must be INT32 or UINT32, got {}",
        name,
        in.indices.dtype());

    const auto& table_shape = in.table.logical_shape();
    const auto& index_shape = in.indices.logical_shape();
    const uint32_t B = table_shape[0], N = table_shape[1];
    const uint32_t Bi = index_shape[0], M = index_shape[1], K = index_shape[2];
    TT_FATAL(Bi == B || Bi == 1, "{}: the index tensor has {} batches, the table {}", name, Bi, B);
    TT_FATAL(
        params.sizes.size() == K && params.windows.size() == K,
        "{}: {} index components, but {} sizes and {} windows",
        name,
        K,
        params.sizes.size(),
        params.windows.size());
    TT_FATAL(K >= 1, "{}: the index tensor needs at least one component", name);
    uint64_t rows = 1;
    for (uint32_t k = 0; k < K; ++k) {
        TT_FATAL(
            params.windows[k] >= 1 && params.windows[k] <= params.sizes[k],
            "{}: window {} of component {} must be in [1, {}]",
            name,
            params.windows[k],
            k,
            params.sizes[k]);
        rows *= params.sizes[k];
    }
    TT_FATAL(rows == N, "{}: the component sizes span {} rows, the table has {}", name, rows, N);
    TT_FATAL(
        params.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{}: the output must be interleaved",
        name);

    if (scatter) {
        TT_FATAL(in.source.has_value(), "rows_scatter: a source tensor is required");
        check_tensor(*in.source, "the source tensor");
        TT_FATAL(
            in.source->dtype() == in.table.dtype(),
            "rows_scatter: the source ({}) must have the table's type ({})",
            in.source->dtype(),
            in.table.dtype());
        const auto& source_shape = in.source->logical_shape();
        const uint32_t W = indexed_rows_window(params);
        TT_FATAL(
            source_shape[0] == B && source_shape[1] == M * W && source_shape[2] == table_shape[2],
            "rows_scatter: the source must be [{}, {}, {}], got {}",
            B,
            M * W,
            table_shape[2],
            source_shape);
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
    const auto& table_shape = in.table.logical_shape();
    const auto& index_shape = in.indices.logical_shape();
    tt::tt_metal::Shape shape({table_shape[0], index_shape[1] * indexed_rows_window(params), table_shape[2]});
    return {tt::tt_metal::TensorSpec(
        shape,
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
