// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/operations/experimental/matmul_fp32/device/matmul_fp32_program_factory.hpp"

namespace ttnn::experimental::prim {

struct MatmulFp32Operation {
    using operation_attributes_t = MatmulFp32Params;
    using tensor_args_t = MatmulFp32Inputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MatmulFp32ProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim
