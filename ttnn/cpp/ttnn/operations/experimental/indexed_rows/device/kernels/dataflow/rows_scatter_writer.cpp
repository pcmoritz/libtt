// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rows_scatter writer. Phase A copies this core's table pages (from the reader)
// into the output. Phase B scans the index tensor for the positions whose
// windows land in those pages and writes, or reduces, the source into them: the
// S pages of a row, or, with a column component, a column window of one page.
// Every output page is owned by exactly one core, so duplicates are applied in
// position order and the result is deterministic. Position (b, m) and window
// offset j read source row (b * M + m) * W + j, or source page b at column
// (m * W + j) * w.

#include <array>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/experimental/indexed_rows/device/kernels/indexed_rows_common.hpp"

void kernel_main() {
    using namespace indexed_rows;
    constexpr uint32_t K = get_arg(args::num_components);
    constexpr bool column = get_arg(args::column) != 0;
    constexpr uint32_t KR = column ? K - 1 : K;
    constexpr bool index_unsigned = get_arg(args::index_unsigned) != 0;
    constexpr bool index_shared = get_arg(args::index_shared) != 0;
    constexpr uint32_t W = get_arg(args::window);
    constexpr uint32_t M = get_arg(args::positions);
    constexpr uint32_t N = get_arg(args::rows_per_batch);
    constexpr uint32_t S = get_arg(args::pages_per_row);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t index_bytes = get_arg(args::index_bytes);
    constexpr uint32_t group = get_arg(args::group);
    constexpr uint32_t index_batch = get_arg(args::index_batch);
    constexpr uint32_t elem_kind = get_arg(args::elem_kind);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t reduce = get_arg(args::reduce);
    constexpr uint32_t C = (page_bytes + chunk_bytes - 1) / chunk_bytes;
    constexpr std::array<uint32_t, K> sizes{ROWS_SIZES};
    constexpr std::array<uint32_t, K> windows{ROWS_WINDOWS};
    constexpr std::array<uint32_t, K> strides{ROWS_STRIDES};
    constexpr uint32_t column_bytes = column ? windows[K - 1] * elem_bytes : 0;

    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);

    DataflowBuffer rows(dfb::rows);
    DataflowBuffer index(dfb::index);
    DataflowBuffer scratch(dfb::scratch);
    const auto output = TensorAccessor(tensor::output);
    const auto indices = TensorAccessor(tensor::indices);
    const auto source = TensorAccessor(tensor::source);
    Noc noc;

    // Phase A: the table pages this core owns.
    const uint32_t entry = rows.get_entry_size();
    const uint32_t total = num_pages * C;
    for (uint32_t q = 0; q < total;) {
        const uint32_t count = total - q < group ? total - q : group;
        rows.wait_front(count);
        for (uint32_t i = 0; i < count; ++i, ++q) {
            const uint32_t page = first_page + q / C;
            const uint32_t offset = (q % C) * chunk_bytes;
            const uint32_t bytes = page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
            noc.async_write(rows, output, bytes, {.offset_bytes = i * entry}, {.page_id = page, .offset_bytes = offset});
        }
        noc.async_writes_flushed();
        rows.pop_front(count);
    }
    noc.async_write_barrier();
    if (num_pages == 0) {
        return;
    }

    // Phase B: the updates into those pages. Two scratch entries hold a source
    // chunk and, for a reduction, the current output chunk.
    scratch.reserve_back(2);
    const uint32_t src_addr = scratch.get_write_ptr();
    const uint32_t cur_addr = src_addr + scratch.get_entry_size();
    const uint32_t end_page = first_page + num_pages;
    const uint32_t b_lo = first_page / (N * S);
    const uint32_t b_hi = (end_page - 1) / (N * S);

    // Reads `bytes` of the source at (src_page, src_offset), reduces them into
    // the output at (page, offset) or overwrites it, and waits for the write so
    // a later update to the same page reads it back.
    // Column windows sit at arbitrary offsets: each is read into L1 at its own
    // offset within a 64-byte granule and moved to the destination's.
    auto update = [&](uint32_t src_page, uint32_t src_offset, uint32_t page, uint32_t offset, uint32_t bytes) {
        const uint32_t src_l1 = src_addr + src_offset % kGranule;
        const uint32_t dst_l1 = cur_addr + offset % kGranule;
        noc.async_read(
            PrecomposedUnicastEndpoint{},
            CoreLocalMem<uint32_t>(src_l1),
            bytes,
            {.noc_addr = source.get_noc_addr(src_page, src_offset)},
            {.offset_bytes = 0});
        if constexpr (reduce != kNone) {
            noc.async_read(
                PrecomposedUnicastEndpoint{},
                CoreLocalMem<uint32_t>(dst_l1),
                bytes,
                {.noc_addr = output.get_noc_addr(page, offset)},
                {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        uint32_t from = src_l1;
        if constexpr (reduce != kNone) {
            reduce_rows<elem_kind, reduce>(dst_l1, src_l1, bytes / elem_bytes);
            from = dst_l1;
        } else if (src_offset % kGranule != offset % kGranule) {
            realign(src_addr, src_offset % kGranule, offset % kGranule, bytes);
            from = src_addr + offset % kGranule;
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(from),
            PrecomposedUnicastEndpoint{},
            bytes,
            {.offset_bytes = 0},
            {.noc_addr = output.get_noc_addr(page, offset)});
        noc.async_write_barrier();
    };

    for (uint32_t b = b_lo; b <= b_hi; ++b) {
        for (uint32_t m0 = 0; m0 < M; m0 += index_batch) {
            const uint32_t n = M - m0 < index_batch ? M - m0 : index_batch;
            read_index_pages(noc, index, indices, (index_shared ? 0 : b) * M + m0, n, index_bytes);
            index.wait_front(n);
            uint32_t addr = index.get_read_ptr();
            for (uint32_t i = 0; i < n; ++i, addr += index.get_entry_size()) {
                const Coords<K> coords =
                    decode<K, index_unsigned>(reinterpret_cast<volatile tt_l1_ptr const uint32_t*>(addr), sizes, windows);
                if (coords.out_of_bounds) {
                    continue;  // StableHLO drops out-of-bounds updates
                }
                const uint32_t m = m0 + i;
                for (uint32_t j = 0; j < W; ++j) {
                    const uint32_t row = b * N + window_row<KR, K>(coords, j, windows, strides);
                    if constexpr (column) {
                        if (row >= first_page && row < end_page) {
                            update(b, (m * W + j) * column_bytes, row,
                                   static_cast<uint32_t>(coords.c[K - 1]) * elem_bytes, column_bytes);
                        }
                    } else {
                        for (uint32_t s = 0; s < S; ++s) {
                            const uint32_t page = row * S + s;
                            if (page < first_page || page >= end_page) {
                                continue;
                            }
                            const uint32_t src_page = ((b * M + m) * W + j) * S + s;
                            for (uint32_t offset = 0; offset < page_bytes; offset += chunk_bytes) {
                                const uint32_t bytes =
                                    page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
                                update(src_page, offset, page, offset, bytes);
                            }
                        }
                    }
                }
            }
            index.pop_front(n);
        }
    }
}
