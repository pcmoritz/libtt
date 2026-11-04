// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/quantize_rows/device/quantize_rows_device_operation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;
using namespace tt::constants;

QuantizeRowsPlan quantize_rows_plan(const Tensor& input, uint32_t max_cores, uint32_t grid_x, uint32_t grid_y) {
    QuantizeRowsPlan plan;
    plan.rows = input.logical_shape()[-2];
    plan.tiles = input.padded_shape()[-1] / TILE_WIDTH;
    plan.stride = plan.rows <= 4 ? 4 : plan.rows <= 8 ? 8 : plan.rows <= 16 ? 16 : 32;
    plan.pack = TILE_HEIGHT / plan.stride;
    plan.dense = (plan.tiles + plan.pack - 1) / plan.pack;
    // Every core sends its maxima to every core, stride * 64 bytes each, so past
    // a point more cores cost more than the work they share (gtest timings:
    // 1x12288 13.4 us on 32 cores, 10.8 on 48; 16x4096 14.5 on 32, 17.1 on 48).
    if (max_cores == 0) {
        max_cores = plan.stride <= 8 ? 48 : 32;
    }
    // The largest rectangle of at most min(dense, max_cores) cores, widest
    // first, so that every core gets a dense tile.
    const uint32_t limit = std::max(1u, std::min(plan.dense, max_cores));
    plan.width = 1;
    plan.height = 1;
    for (uint32_t h = 1; h <= grid_y; ++h) {
        const uint32_t w = std::min(grid_x, limit / h);
        if (w == 0) {
            break;
        }
        if (w * h > plan.width * plan.height) {
            plan.width = w;
            plan.height = h;
        }
    }
    return plan;
}

QuantizeRowsOperation::program_factory_t QuantizeRowsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return QuantizeRowsProgramFactory{};
}

void QuantizeRowsOperation::validate_on_program_cache_miss(const operation_attributes_t& params, const tensor_args_t& in) {
    const auto& input = in.input;
    TT_FATAL(
        input.storage_type() == ttnn::StorageType::DEVICE && input.buffer() != nullptr,
        "quantize_rows: the input must be an allocated device tensor");
    TT_FATAL(input.layout() == Layout::TILE, "quantize_rows: the input must be tiled");
    TT_FATAL(
        input.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "quantize_rows: the input must be interleaved");
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "quantize_rows: the input must be BF16 or FP32");
    const auto& tile = input.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == TILE_HEIGHT && tile.get_width() == TILE_WIDTH, "quantize_rows: the input needs 32x32 tiles");
    const auto& shape = input.logical_shape();
    const auto& padded = input.padded_shape();
    TT_FATAL(shape.rank() >= 2, "quantize_rows: the input must be at least 2-D");
    for (int d = 0; d < static_cast<int>(shape.rank()) - 2; ++d) {
        TT_FATAL(shape[d] == 1, "quantize_rows: the input must be one tile row (dimension {} is {})", d, shape[d]);
    }
    TT_FATAL(
        shape[-2] >= 1 && padded[-2] == TILE_HEIGHT,
        "quantize_rows: at most {} rows, got {}",
        TILE_HEIGHT,
        shape[-2]);
    TT_FATAL(
        shape[-1] >= TILE_WIDTH && shape[-1] == padded[-1],
        "quantize_rows: the row length {} must be a multiple of {}",
        shape[-1],
        TILE_WIDTH);
    TT_FATAL(
        params.dtype == DataType::FP8_E4M3 || params.dtype == DataType::FP8_E5M2,
        "quantize_rows: the result must be FP8_E4M3 or FP8_E5M2");
    TT_FATAL(
        std::isfinite(params.min_scale) && params.min_scale >= 0.0f,
        "quantize_rows: min_scale must be finite and not negative");
}

QuantizeRowsOperation::spec_return_value_t QuantizeRowsOperation::compute_output_specs(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const auto& shape = in.input.logical_shape();
    ttsl::SmallVector<uint32_t> scale_dims(shape.cbegin(), shape.cend());
    scale_dims.back() = 1;
    auto layout = [&](DataType dtype) {
        return tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE), params.output_mem_config);
    };
    return {
        tt::tt_metal::TensorSpec(shape, layout(params.dtype)),
        tt::tt_metal::TensorSpec(ttnn::Shape(scale_dims), layout(DataType::FLOAT32))};
}

QuantizeRowsOperation::tensor_return_value_t QuantizeRowsOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& in) {
    const auto specs = compute_output_specs(params, in);
    return {create_device_tensor(specs[0], in.input.device()), create_device_tensor(specs[1], in.input.device())};
}

}  // namespace ttnn::experimental::prim
