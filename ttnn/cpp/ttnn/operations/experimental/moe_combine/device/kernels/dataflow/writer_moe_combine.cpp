// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// This core's output tiles: tile column c of the [T, H] output's one tile row.
void kernel_main() {
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t Ht = get_named_compile_time_arg_val("Ht");
    constexpr auto out_args = TensorAccessorArgs<0>();
    const uint32_t tile = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, tile);
    for (uint32_t c = core; c < Ht; c += num_cores) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(c), tile);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
