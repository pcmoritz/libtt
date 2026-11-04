// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quantize_rows/quantize_rows.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct QuantizeRowsParams {
    tt::tt_metal::DataType dtype = tt::tt_metal::DataType::FP8_E4M3;
    float min_scale = 0.0f;
    // At most this many cores share the row; 0 picks by the row stride.
    uint32_t max_cores = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct QuantizeRowsInputs {
    Tensor input;
};

// How the row's tiles spread over the cores. The reader packs the data rows of
// `pack` input tiles into one dense tile: input tile jj's rows 0 .. stride - 1
// become dense rows jj * stride .. jj * stride + stride - 1. The stride is the
// row count rounded up to 4, 8, 16 or 32, so that every chunk the kernels move
// is a whole number of 64-byte DRAM words and every SFPU band of four dense
// rows holds rows of one input tile.
struct QuantizeRowsPlan {
    uint32_t rows = 0;    // M
    uint32_t tiles = 0;   // input tiles in the row
    uint32_t stride = 0;  // 4, 8, 16 or 32
    uint32_t pack = 0;    // 32 / stride
    uint32_t dense = 0;   // dense tiles
    uint32_t width = 0;   // the cores, a width x height rectangle
    uint32_t height = 0;
    uint32_t cores() const { return width * height; }
    // Each core's row maxima go to every core: rows 0 .. stride - 1 of an FP32
    // tile's left faces, in `slot_bytes` at slot (core) of the gather buffer.
    uint32_t slot_bytes() const { return stride <= 16 ? stride * 64 : 2048; }
    uint32_t gather_tiles() const { return (cores() * slot_bytes() + 4095) / 4096; }
};

QuantizeRowsPlan quantize_rows_plan(const Tensor& input, uint32_t max_cores, uint32_t grid_x, uint32_t grid_y);

struct QuantizeRowsProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const QuantizeRowsParams&, const QuantizeRowsInputs&, std::vector<Tensor>&);
};

struct QuantizeRowsOperation {
    using operation_attributes_t = QuantizeRowsParams;
    using tensor_args_t = QuantizeRowsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<QuantizeRowsProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim
