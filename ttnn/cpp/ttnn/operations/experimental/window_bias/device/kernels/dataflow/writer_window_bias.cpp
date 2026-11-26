// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Each token's bias tile `key_tile` ([T, 1, heads, W]: page t * window_tiles +
// key_tile) and, on the sink core, its sink tile ([T, 1, heads, 32]: page t).
void kernel_main() {
    const uint32_t key_tile = get_arg_val<uint32_t>(0);
    const bool sinks = get_arg_val<uint32_t>(1) != 0;
    const uint32_t bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t sink_addr = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_bias = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t cb_sink_out = get_named_compile_time_arg_val("cb_sink_out");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t window_tiles = get_named_compile_time_arg_val("window_tiles");
    constexpr bool has_sink = get_named_compile_time_arg_val("has_sink") != 0;
    constexpr auto bias_args = TensorAccessorArgs<0>();
    constexpr auto sink_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile = 2048;
    const auto bias = TensorAccessor(bias_args, bias_addr, tile);
    const auto sink = TensorAccessor(sink_args, sink_addr, tile);
    for (uint32_t t = 0; t < tokens; ++t) {
        cb_wait_front(cb_bias, 1);
        noc_async_write(get_read_ptr(cb_bias), bias.get_noc_addr(t * window_tiles + key_tile), tile);
        if (has_sink && sinks) {
            cb_wait_front(cb_sink_out, 1);
            noc_async_write(get_read_ptr(cb_sink_out), sink.get_noc_addr(t), tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_bias, 1);
        if (has_sink && sinks) {
            cb_pop_front(cb_sink_out, 1);
        }
    }
}
