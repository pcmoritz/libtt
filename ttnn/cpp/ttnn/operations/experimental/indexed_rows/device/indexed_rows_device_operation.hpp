// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/operations/experimental/indexed_rows/device/indexed_rows_program_factory.hpp"

namespace ttnn::experimental::prim {

struct RowsGatherOperation {
    using operation_attributes_t = IndexedRowsParams;
    using tensor_args_t = IndexedRowsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<RowsGatherProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

struct RowsScatterOperation {
    using operation_attributes_t = IndexedRowsParams;
    using tensor_args_t = IndexedRowsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<RowsScatterProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Shared by both operations' validation.
void validate_indexed_rows(const IndexedRowsParams&, const IndexedRowsInputs&, bool scatter);
uint32_t indexed_rows_window(const IndexedRowsParams&);

}  // namespace ttnn::experimental::prim
