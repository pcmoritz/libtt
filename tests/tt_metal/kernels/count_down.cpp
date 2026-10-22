// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Writes (counter > 0) into every word of page 0 of the predicate buffer and
// decrements the counter, held in page 0 of its own buffer, when it is
// positive. Both pages are 64 bytes, staged through L1.
void kernel_main() {
    const uint32_t counter_addr = get_arg_val<uint32_t>(0);
    const uint32_t predicate_addr = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_arg_val<uint32_t>(2);
    const InterleavedAddrGen<true> counter{.bank_base_address = counter_addr, .page_size = 64};
    const InterleavedAddrGen<true> predicate{.bank_base_address = predicate_addr, .page_size = 64};
    noc_async_read(counter.get_noc_addr(0), scratch, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t value = words[0];
    const uint32_t positive = value > 0 ? 1 : 0;
    for (uint32_t i = 0; i < 16; i++) {
        words[i] = value - positive;
    }
    noc_async_write(scratch, counter.get_noc_addr(0), 64);
    noc_async_write_barrier();
    for (uint32_t i = 0; i < 16; i++) {
        words[i] = positive;
    }
    noc_async_write(scratch, predicate.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
