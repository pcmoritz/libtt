// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <variant>
#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/dense_rows/dense_rows.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DenseRowsParams {
    DenseRowsDirection direction = DenseRowsDirection::Pack;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct DenseRowsInputs {
    Tensor input;
};

struct DenseRowsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const DenseRowsParams&, const DenseRowsInputs&, std::vector<Tensor>&);
};

struct DenseRowsOperation {
    using operation_attributes_t = DenseRowsParams;
    using tensor_args_t = DenseRowsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<DenseRowsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim
