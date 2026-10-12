// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rows_scatter writer. Phase A takes this core's table pages from the reader:
// when they fit L1 (`resident`) it keeps them, otherwise it copies them to the
// output as they stream by. Phase B streams the index tensor (and, with a
// column component, the source) in chunks and applies the updates that land in
// the owned pages: into the L1 copies, written out once at the end, or as DRAM
// read-modify-writes. Every output page is owned by exactly one core, so
// duplicates apply in position order and the result is deterministic. Position
// (b, m) and window offset j read source row (b * M + m) * W + j, or source page
// b at column (m * W + j) * w.

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
    constexpr bool resident = get_arg(args::resident) != 0;
    constexpr uint32_t W = get_arg(args::window);
    constexpr uint32_t M = get_arg(args::positions);
    constexpr uint32_t N = get_arg(args::rows_per_batch);
    constexpr uint32_t S = get_arg(args::pages_per_row);
    constexpr uint32_t page_bytes = get_arg(args::page_bytes);
    constexpr uint32_t chunk_bytes = get_arg(args::chunk_bytes);
    constexpr uint32_t index_stride = get_arg(args::index_stride);  // words between components
    constexpr uint32_t index_batch = get_arg(args::index_batch);  // positions per chunk
    constexpr uint32_t group = get_arg(args::group);
    constexpr uint32_t elem_kind = get_arg(args::elem_kind);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t reduce = get_arg(args::reduce);
    constexpr uint32_t C = (page_bytes + chunk_bytes - 1) / chunk_bytes;
    constexpr std::array<uint32_t, K> sizes{ROWS_SIZES};
    constexpr std::array<uint32_t, K> windows{ROWS_WINDOWS};
    constexpr std::array<uint32_t, K> strides{ROWS_STRIDES};
    constexpr uint32_t column_bytes = column ? windows[K - 1] * elem_bytes : 0;
    constexpr uint32_t source_bytes = column ? W * column_bytes : 0;  // per position

    const uint32_t first_page = get_arg(args::first_page);
    const uint32_t num_pages = get_arg(args::num_pages);

    DataflowBuffer rows(dfb::rows);
    DataflowBuffer index(dfb::index);
    DataflowBuffer stream(dfb::stream);
    DataflowBuffer scratch(dfb::scratch);
    const auto output = TensorAccessor(tensor::output);
    const auto indices = TensorAccessor(tensor::indices);
    const auto source = TensorAccessor(tensor::source);
    Noc noc;

    const uint32_t entry = rows.get_entry_size();
    if (num_pages == 0) {
        return;
    }

    // Phase A: the table pages this core owns.
    uint32_t pages_l1 = 0;  // resident copies, one entry per page
    if constexpr (resident) {
        rows.wait_front(num_pages);
        pages_l1 = rows.get_read_ptr();
    } else {
        const uint32_t total = num_pages * C;
        for (uint32_t q = 0; q < total;) {
            const uint32_t count = total - q < group ? total - q : group;
            rows.wait_front(count);
            for (uint32_t i = 0; i < count; ++i, ++q) {
                const uint32_t page = first_page + q / C;
                const uint32_t offset = (q % C) * chunk_bytes;
                const uint32_t bytes = page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
                noc.async_write(
                    rows, output, bytes, {.offset_bytes = i * entry}, {.page_id = page, .offset_bytes = offset});
            }
            noc.async_writes_flushed();
            rows.pop_front(count);
        }
        noc.async_write_barrier();
    }

    // Phase B. Two scratch entries hold a source chunk (row mode) or a realigned
    // column window, and the current output chunk of a DRAM read-modify-write.
    scratch.reserve_back(2);
    const uint32_t src_addr = scratch.get_write_ptr();
    const uint32_t cur_addr = src_addr + scratch.get_entry_size();
    const uint32_t end_page = first_page + num_pages;
    const uint32_t b_lo = first_page / (N * S);
    const uint32_t b_hi = (end_page - 1) / (N * S);

    // Applies `bytes` of source, in L1 at `src_l1`, to the output at
    // (page, offset): into the resident copy, or by DRAM read-modify-write.
    auto apply_l1 = [&](uint32_t src_l1, uint32_t page, uint32_t offset, uint32_t bytes) {
        if constexpr (resident) {
            const uint32_t dst = pages_l1 + (page - first_page) * entry + offset;
            if constexpr (reduce != kNone) {
                reduce_rows<elem_kind, reduce>(dst, src_l1, bytes / elem_bytes);
            } else {
                copy_l1(dst, src_l1, bytes);
            }
            return;
        }
        const uint32_t dst_l1 = cur_addr + offset % kGranule;
        uint32_t from = src_l1;
        if constexpr (reduce != kNone) {
            noc.async_read(
                PrecomposedUnicastEndpoint{},
                CoreLocalMem<uint32_t>(dst_l1),
                bytes,
                {.noc_addr = output.get_noc_addr(page, offset)},
                {.offset_bytes = 0});
            noc.async_read_barrier();
            reduce_rows<elem_kind, reduce>(dst_l1, src_l1, bytes / elem_bytes);
            from = dst_l1;
        } else if (src_l1 % kGranule != offset % kGranule) {
            // A NoC write keeps the offset within a 64-byte granule.
            copy_l1(dst_l1, src_l1, bytes);
            from = dst_l1;
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(from),
            PrecomposedUnicastEndpoint{},
            bytes,
            {.offset_bytes = 0},
            {.noc_addr = output.get_noc_addr(page, offset)});
        // A later update to the same page reads it back.
        noc.async_write_barrier();
    };

    for (uint32_t b = b_lo; b <= b_hi; ++b) {
        for (uint32_t m0 = 0; m0 < M; m0 += index_batch) {
            const uint32_t n = M - m0 < index_batch ? M - m0 : index_batch;
            const uint32_t index_l1 =
                stream_index(noc, index, indices, K, index_stride * sizeof(uint32_t), index_shared ? 0 : b, m0, n);
            uint32_t source_l1 = 0;
            if constexpr (column) {
                source_l1 = stream_chunk(noc, stream, source, b, m0 * source_bytes, n * source_bytes);
            }
            for (uint32_t i = 0; i < n; ++i) {
                const auto* position = reinterpret_cast<const uint32_t*>(index_l1) + i;
                if constexpr (column && W == 1) {
                    // Most positions belong to other cores: decide on the row
                    // components alone before decoding the column.
                    const Coords<KR> rows_only = decode<KR, index_unsigned>(position, index_stride, sizes, windows);
                    const uint32_t row = b * N + window_row<KR>(rows_only, 0, windows, strides);
                    if (rows_only.out_of_bounds || row - first_page >= num_pages) {
                        continue;
                    }
                }
                const Coords<K> coords = decode<K, index_unsigned>(position, index_stride, sizes, windows);
                if (coords.out_of_bounds) {
                    continue;  // StableHLO drops out-of-bounds updates
                }
                const uint32_t m = m0 + i;
                for (uint32_t j = 0; j < W; ++j) {
                    const uint32_t row = b * N + window_row<KR, K>(coords, j, windows, strides);
                    if constexpr (column) {
                        if (row < first_page || row >= end_page) {
                            continue;
                        }
                        apply_l1(
                            source_l1 + (i * W + j) * column_bytes, row,
                            static_cast<uint32_t>(coords.c[K - 1]) * elem_bytes, column_bytes);
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
                                // An overwrite of a resident page is read straight into it.
                                const bool direct = resident && reduce == kNone;
                                const uint32_t into =
                                    direct ? pages_l1 + (page - first_page) * entry + offset : src_addr;
                                noc.async_read(
                                    PrecomposedUnicastEndpoint{},
                                    CoreLocalMem<uint32_t>(into),
                                    bytes,
                                    {.noc_addr = source.get_noc_addr(src_page, offset)},
                                    {.offset_bytes = 0});
                                noc.async_read_barrier();
                                if (!direct) {
                                    apply_l1(src_addr, page, offset, bytes);
                                }
                            }
                        }
                    }
                }
            }
            if constexpr (column) {
                stream.pop_front(1);
            }
            index.pop_front(1);
        }
    }

    // Resident pages go out once.
    if constexpr (resident) {
        for (uint32_t p = 0; p < num_pages; ++p) {
            noc.async_write(
                rows, output, page_bytes, {.offset_bytes = p * entry}, {.page_id = first_page + p, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        rows.pop_front(num_pages);
    }
}
