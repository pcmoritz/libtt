// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rng_bits data movement: reads the state's first page into the state DFB for
// the compute kernel and writes this core's bits tiles, a contiguous range in
// tile order. The first core also writes the new state: the state with w1
// advanced, modulo 2^32.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t state_page_bytes = get_arg(args::state_page_bytes);
    const uint32_t first_tile = get_arg(args::first_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t write_state = get_arg(args::write_state);

    DataflowBuffer state_dfb(dfb::state);
    DataflowBuffer out_dfb(dfb::out);
    const auto state = TensorAccessor(tensor::state);
    const auto bits = TensorAccessor(tensor::bits);
    const auto new_state = TensorAccessor(tensor::new_state);
    Noc noc;

    state_dfb.reserve_back(1);
    noc.async_read(state, state_dfb, state_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    if (write_state) {
        // The new state goes out from the same entry before the compute kernel
        // sees it, with w1 restored after the write.
        volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(state_dfb.get_write_ptr());
        const uint32_t w1 = words[1];
        words[1] = w1 + 1;
        const uint32_t entry_offset = state_dfb.get_write_ptr() - state_dfb.get_read_ptr();
        noc.async_write(state_dfb, new_state, state_page_bytes, {.offset_bytes = entry_offset}, {.page_id = 0});
        noc.async_write_barrier();
        words[1] = w1;
    }
    state_dfb.push_back(1);

    const uint32_t tile_bytes = out_dfb.get_entry_size();
    for (uint32_t tile = first_tile; tile < first_tile + num_tiles; ++tile) {
        out_dfb.wait_front(1);
        noc.async_write(out_dfb, bits, tile_bytes, {.offset_bytes = 0}, {.page_id = tile});
        noc.async_writes_flushed();
        out_dfb.pop_front(1);
    }
    noc.async_write_barrier();
}
