// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The logits' and bias' tiles (one tile row each) and the valid-row count.
void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_valid_addr = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_logits = get_named_compile_time_arg_val("cb_logits");
    constexpr uint32_t cb_bias = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t cb_num_valid = get_named_compile_time_arg_val("cb_num_valid");
    constexpr uint32_t Et = get_named_compile_time_arg_val("Et");
    constexpr bool has_num_valid = get_named_compile_time_arg_val("has_num_valid") != 0;
    constexpr auto logits_args = TensorAccessorArgs<0>();
    constexpr auto bias_args = TensorAccessorArgs<logits_args.next_compile_time_args_offset()>();
    constexpr auto num_valid_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    const uint32_t logits_tile = get_tile_size(cb_logits);
    const uint32_t bias_tile = get_tile_size(cb_bias);
    const auto logits = TensorAccessor(logits_args, logits_addr, logits_tile);
    const auto bias = TensorAccessor(bias_args, bias_addr, bias_tile);
    cb_reserve_back(cb_logits, Et);
    cb_reserve_back(cb_bias, Et);
    for (uint32_t e = 0; e < Et; ++e) {
        noc_async_read(logits.get_noc_addr(e), get_write_ptr(cb_logits) + e * logits_tile, logits_tile);
        noc_async_read(bias.get_noc_addr(e), get_write_ptr(cb_bias) + e * bias_tile, bias_tile);
    }
    if constexpr (has_num_valid) {
        const uint32_t tile = get_tile_size(cb_num_valid);
        const auto num_valid = TensorAccessor(num_valid_args, num_valid_addr, tile);
        cb_reserve_back(cb_num_valid, 1);
        noc_async_read(num_valid.get_noc_addr(0), get_write_ptr(cb_num_valid), tile);
    }
    noc_async_read_barrier();
    cb_push_back(cb_logits, Et);
    cb_push_back(cb_bias, Et);
    if constexpr (has_num_valid) {
        cb_push_back(cb_num_valid, 1);
    }
}
