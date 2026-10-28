// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Copies page 0 of a DRAM buffer (64 bytes) to page 0 of another, through L1,
// after reading the source `delay` times more, so that it finishes late.
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_arg_val<uint32_t>(2);
    const uint32_t delay = get_arg_val<uint32_t>(3);
    const InterleavedAddrGen<true> src{.bank_base_address = src_addr, .page_size = 64};
    const InterleavedAddrGen<true> dst{.bank_base_address = dst_addr, .page_size = 64};
    for (uint32_t i = 0; i <= delay; i++) {
        noc_async_read(src.get_noc_addr(0), scratch, 64);
        noc_async_read_barrier();
    }
    noc_async_write(scratch, dst.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
