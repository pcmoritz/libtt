// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// The weights and ids tiles, rows from the valid-row count on set to weight
// 0 and id -1 first. Row t, column i of a tile (t, i < 16) is word t * 16 + i
// of face 0.
void kernel_main() {
    const uint32_t weights_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_num_valid = get_named_compile_time_arg_val("cb_num_valid");
    constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");
    constexpr uint32_t cb_ids = get_named_compile_time_arg_val("cb_ids");
    constexpr uint32_t k = get_named_compile_time_arg_val("k");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr bool has_num_valid = get_named_compile_time_arg_val("has_num_valid") != 0;
    constexpr auto weights_args = TensorAccessorArgs<0>();
    constexpr auto ids_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    const uint32_t tile = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weights_args, weights_addr, tile);
    const auto ids = TensorAccessor(ids_args, ids_addr, tile);
    cb_wait_front(cb_weights, 1);
    cb_wait_front(cb_ids, 1);
    if constexpr (has_num_valid) {
        cb_wait_front(cb_num_valid, 1);
        const int32_t valid = *reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_read_ptr(cb_num_valid));
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_weights));
        volatile tt_l1_ptr uint32_t* id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ids));
        for (uint32_t t = valid < 0 ? 0 : static_cast<uint32_t>(valid); t < tokens; ++t) {
            for (uint32_t i = 0; i < k; ++i) {
                w[t * 16 + i] = 0;
                id[t * 16 + i] = 0xffffffffu;
            }
        }
        cb_pop_front(cb_num_valid, 1);
    }
    noc_async_write(get_read_ptr(cb_weights), weights.get_noc_addr(0), tile);
    noc_async_write(get_read_ptr(cb_ids), ids.get_noc_addr(0), tile);
    noc_async_write_barrier();
    cb_pop_front(cb_weights, 1);
    cb_pop_front(cb_ids, 1);
}
