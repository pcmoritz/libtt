// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/matmul_fp32/device/matmul_fp32_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

MatmulFp32Operation::program_factory_t MatmulFp32Operation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return MatmulFp32ProgramFactory{};
}

void MatmulFp32Operation::validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& in) {
    for (const auto& [tensor, name] : {std::pair<const Tensor*, const char*>{&in.a, "a"}, {&in.b, "b"}}) {
        TT_FATAL(
            tensor->storage_type() == ttnn::StorageType::DEVICE && tensor->buffer() != nullptr,
            "matmul_fp32: {} must be an allocated device tensor",
            name);
        TT_FATAL(tensor->dtype() == DataType::FLOAT32, "matmul_fp32: {} must be FLOAT32, got {}", name, tensor->dtype());
        TT_FATAL(tensor->layout() == Layout::TILE, "matmul_fp32: {} must be in TILE layout", name);
        TT_FATAL(
            tensor->memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "matmul_fp32: {} must be interleaved",
            name);
        TT_FATAL(tensor->logical_shape().rank() >= 2, "matmul_fp32: {} must have rank 2 or more", name);
    }
    TT_FATAL(in.a.device() == in.b.device(), "matmul_fp32: a and b must be on the same device");
    TT_FATAL(
        in.a.device()->arch() == tt::ARCH::BLACKHOLE,
        "matmul_fp32 is only implemented for Blackhole, got {}",
        in.a.device()->arch());
    TT_FATAL(
        attrs.output_mem_config.memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "matmul_fp32: the output must be interleaved");

    const auto& a = in.a.padded_shape();
    const auto& b = in.b.padded_shape();
    TT_FATAL(
        in.a.logical_shape()[-1] == in.b.logical_shape()[-2],
        "matmul_fp32: contraction dimensions differ: a {} and b {}",
        in.a.logical_shape(),
        in.b.logical_shape());
    TT_FATAL(a[-1] == b[-2], "matmul_fp32: padded contraction dimensions differ: a {} and b {}", a, b);
    uint32_t b_batch = 1;
    for (int i = 0; i < static_cast<int>(b.rank()) - 2; ++i) {
        b_batch *= b[i];
    }
    if (b_batch != 1) {
        bool same_batch = a.rank() == b.rank();
        for (int i = 0; same_batch && i < static_cast<int>(a.rank()) - 2; ++i) {
            same_batch = a[i] == b[i];
        }
        TT_FATAL(same_batch, "matmul_fp32: b must have the batch dimensions of a, or none: a {} and b {}", a, b);
    }
}

MatmulFp32Operation::spec_return_value_t MatmulFp32Operation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tt::tt_metal::Shape shape = in.a.logical_shape();
    shape[-1] = in.b.logical_shape()[-1];
    return {tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(DataType::FLOAT32, tt::tt_metal::PageConfig(Layout::TILE), attrs.output_mem_config))};
}

MatmulFp32Operation::tensor_return_value_t MatmulFp32Operation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {ttnn::create_device_tensor(compute_output_specs(attrs, in)[0], in.a.device())};
}

}  // namespace ttnn::experimental::prim
