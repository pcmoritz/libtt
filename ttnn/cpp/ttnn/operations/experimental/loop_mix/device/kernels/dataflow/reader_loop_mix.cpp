// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// This core's two tile columns of dest and the messages, its rows of the
// down projection (both halves) and its columns of gain and forget, in the
// order the compute kernel takes them.
void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t dest_addr = get_arg_val<uint32_t>(1);
    const uint32_t messages_addr = get_arg_val<uint32_t>(2);
    const uint32_t down_addr = get_arg_val<uint32_t>(3);
    const uint32_t gain_addr = get_arg_val<uint32_t>(4);
    const uint32_t forget_addr = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_dest = get_named_compile_time_arg_val("cb_dest");
    constexpr uint32_t cb_messages = get_named_compile_time_arg_val("cb_messages");
    constexpr uint32_t cb_down = get_named_compile_time_arg_val("cb_down");
    constexpr uint32_t cb_gain = get_named_compile_time_arg_val("cb_gain");
    constexpr uint32_t cb_forget = get_named_compile_time_arg_val("cb_forget");
    constexpr uint32_t Q = get_named_compile_time_arg_val("num_queries");
    constexpr uint32_t Qd = get_named_compile_time_arg_val("num_dest");
    constexpr uint32_t Rt = get_named_compile_time_arg_val("Rt");
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");
    constexpr uint32_t cnt = 2;
    constexpr auto dest_args = TensorAccessorArgs<0>();
    constexpr auto messages_args = TensorAccessorArgs<dest_args.next_compile_time_args_offset()>();
    constexpr auto down_args = TensorAccessorArgs<messages_args.next_compile_time_args_offset()>();
    constexpr auto gain_args = TensorAccessorArgs<down_args.next_compile_time_args_offset()>();
    constexpr auto forget_args = TensorAccessorArgs<gain_args.next_compile_time_args_offset()>();
    const uint32_t f32_tile = get_tile_size(cb_dest);
    const uint32_t bf16_tile = get_tile_size(cb_down);
    const auto dest = TensorAccessor(dest_args, dest_addr, f32_tile);
    const auto messages = TensorAccessor(messages_args, messages_addr, f32_tile);
    const auto down = TensorAccessor(down_args, down_addr, bf16_tile);
    const auto gain = TensorAccessor(gain_args, gain_addr, bf16_tile);
    const auto forget = TensorAccessor(forget_args, forget_addr, bf16_tile);

    cb_reserve_back(cb_dest, Qd * cnt);
    cb_reserve_back(cb_messages, Q * cnt);
    for (uint32_t q = 0; q < Qd; ++q) {
        for (uint32_t c = 0; c < cnt; ++c) {
            noc_async_read(dest.get_noc_addr(q * Wt + start + c), get_write_ptr(cb_dest) + (q * cnt + c) * f32_tile, f32_tile);
        }
    }
    for (uint32_t q = 0; q < Q; ++q) {
        for (uint32_t c = 0; c < cnt; ++c) {
            noc_async_read(
                messages.get_noc_addr(q * Wt + start + c), get_write_ptr(cb_messages) + (q * cnt + c) * f32_tile, f32_tile);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_dest, Qd * cnt);
    cb_push_back(cb_messages, Q * cnt);

    // down [Q, 2W, R]: tile (q * 2 Wt + row, rt), the dest half's rows c, the
    // messages' Wt + c.
    cb_reserve_back(cb_down, Q * 2 * cnt * Rt);
    uint32_t slot = 0;
    for (uint32_t q = 0; q < Q; ++q) {
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t c = 0; c < cnt; ++c) {
                for (uint32_t rt = 0; rt < Rt; ++rt) {
                    const uint32_t row = q * 2 * Wt + half * Wt + start + c;
                    noc_async_read(down.get_noc_addr(row * Rt + rt), get_write_ptr(cb_down) + slot++ * bf16_tile, bf16_tile);
                }
            }
        }
    }
    // gain and forget [Q, R, W]: tile (q * Rt + kt, c).
    cb_reserve_back(cb_gain, Q * Rt * cnt);
    cb_reserve_back(cb_forget, Q * Rt * cnt);
    slot = 0;
    for (uint32_t q = 0; q < Q; ++q) {
        for (uint32_t kt = 0; kt < Rt; ++kt) {
            for (uint32_t c = 0; c < cnt; ++c) {
                const uint32_t page = (q * Rt + kt) * Wt + start + c;
                noc_async_read(gain.get_noc_addr(page), get_write_ptr(cb_gain) + slot * bf16_tile, bf16_tile);
                noc_async_read(forget.get_noc_addr(page), get_write_ptr(cb_forget) + slot * bf16_tile, bf16_tile);
                ++slot;
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_down, Q * 2 * cnt * Rt);
    cb_push_back(cb_gain, Q * Rt * cnt);
    cb_push_back(cb_forget, Q * Rt * cnt);
}
