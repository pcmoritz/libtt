// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <variant>
#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/masked_reduce/masked_reduce.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MaskedReduceParams {
    MaskedCompare compare = MaskedCompare::GT;
    MaskedMode mode = MaskedMode::Count;
    // Cores that share a tile row.
    uint32_t cores_per_row = 0;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::FLOAT32;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct MaskedReduceInputs {
    Tensor input;
    Tensor threshold;
};

// The result, shaped like the threshold: each tile row's cores send their row
// sums to the row's first core, which adds them up (kernels/).
struct MaskedReduceProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const MaskedReduceParams&, const MaskedReduceInputs&, std::vector<Tensor>&);
};

struct MaskedReduceOperation {
    using operation_attributes_t = MaskedReduceParams;
    using tensor_args_t = MaskedReduceInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<MaskedReduceProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Tile rows of the input over all its leading dimensions.
uint32_t masked_reduce_tile_rows(const Tensor& input);

// Whether the threshold is [..., H] rather than [..., H, 1].
bool masked_reduce_row_form(const Tensor& input, const Tensor& threshold);

// Whether the threshold [..., H] reduces each [N, 32] matrix of a
// [..., H, N, 32] input, and how many such matrices it has.
bool masked_reduce_chunked(const Tensor& input, const Tensor& threshold);
uint32_t masked_reduce_matrices(const Tensor& input);

// Columns of the partials a tile row's master core gathers from cores_per_row cores.
uint32_t masked_reduce_partial_columns(uint32_t cores_per_row);

}  // namespace ttnn::experimental::prim
