// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

namespace {

// Element i of a [N <= 32] int32 vector in TILE layout: row 0, column i.
FORCE_INLINE uint32_t read_slot(uint32_t address, uint32_t i) {
    const uint32_t offset = (i / 16) * 256 + i % 16;
    return reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(address)[offset];
}

}  // namespace

void kernel_main() {
    constexpr uint32_t count = get_compile_time_arg_val(0);
    constexpr uint32_t pages_per_slot = get_compile_time_arg_val(1);
    constexpr uint32_t num_workers = get_compile_time_arg_val(2);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t buffered_pages = get_compile_time_arg_val(4);

    const uint32_t worker = get_arg_val<uint32_t>(0);
    const uint32_t pool_address = get_arg_val<uint32_t>(1);
    const uint32_t sources_address = get_arg_val<uint32_t>(2);
    const uint32_t targets_address = get_arg_val<uint32_t>(3);

    constexpr auto pool_args = TensorAccessorArgs<5>();
    constexpr auto sources_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();
    constexpr auto targets_args = TensorAccessorArgs<sources_args.next_compile_time_args_offset()>();
    const auto pool = TensorAccessor(pool_args, pool_address, page_bytes);
    const auto sources = TensorAccessor(sources_args, sources_address, 4096);
    const auto targets = TensorAccessor(targets_args, targets_address, 4096);

    CircularBuffer data_cb(tt::CBIndex::c_0);
    CircularBuffer slots_cb(tt::CBIndex::c_1);
    Noc noc;

    // This kernel owns both buffers; use them as scratch without queueing.
    data_cb.reserve_back(buffered_pages);
    slots_cb.reserve_back(2);
    const uint32_t source_slots = slots_cb.get_write_ptr();
    const uint32_t target_slots = source_slots + 4096;
    noc.async_read(sources, slots_cb, 4096, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(targets, slots_cb, 4096, {.page_id = 0}, {.offset_bytes = 4096});
    noc.async_read_barrier();

    uint32_t buffer = 0;
    for (uint32_t work = worker; work < count * pages_per_slot; work += num_workers) {
        const uint32_t pair = work / pages_per_slot;
        const uint32_t page = work % pages_per_slot;
        const uint32_t source = read_slot(source_slots, pair);
        const uint32_t target = read_slot(target_slots, pair);
        if (source == 0 || target == 0) {
            continue;
        }
        // The write that last used this buffer must have left it.
        if (buffer == 0) {
            noc.async_write_barrier();
        }
        const uint32_t offset = buffer * page_bytes;
        noc.async_read(
            pool, data_cb, page_bytes, {.page_id = source * pages_per_slot + page}, {.offset_bytes = offset});
        noc.async_read_barrier();
        noc.async_write(
            CoreLocalMem<uint32_t>(data_cb.get_write_ptr() + offset), pool, page_bytes, {.offset_bytes = 0},
            {.page_id = target * pages_per_slot + page});
        buffer = (buffer + 1) % buffered_pages;
    }
    noc.async_write_barrier();
}
