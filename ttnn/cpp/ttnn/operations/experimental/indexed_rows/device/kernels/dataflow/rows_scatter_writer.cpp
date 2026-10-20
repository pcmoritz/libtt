// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// rows_scatter writer. The reader streams this core's table pages through the
// rows DFB in chunks; the writer takes them in batches that fill the buffer,
// and for each batch streams the index tensor (and, with a column component,
// the source) in chunks, applies the updates that land in the batch to its
// L1 copy, and writes the batch out once. Every output page is owned by one
// core, so duplicates apply in position order and the result is deterministic.
// Position (b, m) and window offset j read source row (b * M + m) * W + j, or
// source page b at column (m * W + j) * w.

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
    constexpr uint32_t index_stride = get_arg(args::index_stride);  // words between components
    constexpr uint32_t index_batch = get_arg(args::index_batch);    // positions per chunk
    constexpr uint32_t batch = get_arg(args::batch);                // rows DFB entries
    constexpr uint32_t elem_kind = get_arg(args::elem_kind);
    constexpr uint32_t elem_bytes = get_arg(args::elem_bytes);
    constexpr uint32_t reduce = get_arg(args::reduce);
    constexpr uint32_t C = (page_bytes + chunk_bytes - 1) / chunk_bytes;
    constexpr std::array<uint32_t, K> sizes{ROWS_SIZES};
    constexpr std::array<uint32_t, K> windows{ROWS_WINDOWS};
    constexpr std::array<uint32_t, K> strides{ROWS_STRIDES};
    constexpr uint32_t column_bytes = column ? windows[K - 1] * elem_bytes : 0;
    constexpr uint32_t source_bytes = column ? W * column_bytes : 0;  // per position
    // A column window wider than a piece streams through the stream DFB a piece
    // at a time (the program factory sizes the entry to kStreamPiece).
    constexpr uint32_t kStreamPiece = 64 * 1024;
    constexpr bool stream_window = column && column_bytes > kStreamPiece;

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
    const uint32_t b_lo = first_page / (N * S);
    const uint32_t b_hi = num_pages == 0 ? 0 : (first_page + num_pages - 1) / (N * S);
    // Row-mode reductions read the source row chunk here before combining it.
    scratch.reserve_back(1);
    const uint32_t src_addr = scratch.get_write_ptr();

    const uint32_t total = num_pages * C;  // chunks, in page order
    for (uint32_t q0 = 0; q0 < total; q0 += batch) {
        const uint32_t count = total - q0 < batch ? total - q0 : batch;
        rows.wait_front(count);
        const uint32_t batch_l1 = rows.get_read_ptr();
        // The pages with a chunk in this batch.
        const uint32_t batch_page_lo = first_page + q0 / C;
        const uint32_t batch_page_hi = first_page + (q0 + count + C - 1) / C;

        // The L1 address of (page, offset) when its chunk is in this batch.
        auto in_batch = [&](uint32_t page, uint32_t offset, uint32_t& l1) {
            const uint32_t q = (page - first_page) * C + offset / chunk_bytes;
            if (page < first_page || page >= first_page + num_pages || q < q0 || q >= q0 + count) {
                return false;
            }
            l1 = batch_l1 + (q - q0) * entry + offset % chunk_bytes;
            return true;
        };
        // Applies `bytes` of source at `src_l1` to the output copy at `dst`.
        auto apply = [&](uint32_t dst, uint32_t src_l1, uint32_t bytes) {
            if constexpr (reduce != kNone) {
                reduce_rows<elem_kind, reduce>(dst, src_l1, bytes / elem_bytes);
            } else {
                copy_l1(dst, src_l1, bytes);
            }
        };

        for (uint32_t b = b_lo; b <= b_hi; ++b) {
            for (uint32_t m0 = 0; m0 < M; m0 += index_batch) {
                const uint32_t n = M - m0 < index_batch ? M - m0 : index_batch;
                const uint32_t index_l1 =
                    stream_index(noc, index, indices, K, index_stride * sizeof(uint32_t), index_shared ? 0 : b, m0, n);
                uint32_t source_l1 = 0;
                if constexpr (column && !stream_window) {
                    source_l1 = stream_chunk(noc, stream, source, b, m0 * source_bytes, n * source_bytes);
                }
                for (uint32_t i = 0; i < n; ++i) {
                    const auto* position = reinterpret_cast<const uint32_t*>(index_l1) + i;
                    if constexpr (column && W == 1) {
                        // Most positions belong to other cores: decide on the
                        // row components alone before decoding the column.
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
                        uint32_t dst;
                        if constexpr (column) {
                            // A window may straddle chunks (and so batches):
                            // each chunk takes its part. A streamed window has
                            // [piece_lo, piece_hi) at piece_l1, and only pieces
                            // that land in this batch are read.
                            uint32_t offset = static_cast<uint32_t>(coords.c[K - 1]) * elem_bytes;
                            const uint32_t src = source_l1 + (i * W + j) * column_bytes;
                            uint32_t piece_lo = 0;
                            uint32_t piece_hi = 0;
                            uint32_t piece_l1 = 0;
                            for (uint32_t w = 0; w < column_bytes;) {
                                const uint32_t room = chunk_bytes - offset % chunk_bytes;
                                const uint32_t bytes = column_bytes - w < room ? column_bytes - w : room;
                                if (in_batch(row, offset, dst)) {
                                    if constexpr (stream_window) {
                                        if (w < piece_lo || w + bytes > piece_hi) {
                                            if (piece_hi != 0) {
                                                stream.pop_front(1);
                                            }
                                            piece_lo = w;
                                            piece_hi =
                                                column_bytes - w < kStreamPiece ? column_bytes : w + kStreamPiece;
                                            piece_l1 = stream_chunk(
                                                noc, stream, source, b, (m * W + j) * column_bytes + w, piece_hi - w);
                                        }
                                        apply(dst, piece_l1 + (w - piece_lo), bytes);
                                    } else {
                                        apply(dst, src + w, bytes);
                                    }
                                }
                                offset += bytes;
                                w += bytes;
                            }
                            if constexpr (stream_window) {
                                if (piece_hi != 0) {
                                    stream.pop_front(1);
                                }
                            }
                            continue;
                        }
                        // Only the row's pages in this batch; an overwrite reads them straight
                        // into the copy and waits once per position (the pages of one position are
                        // distinct, and positions apply in order).
                        const uint32_t page_lo = batch_page_lo > row * S ? batch_page_lo : row * S;
                        const uint32_t page_hi = batch_page_hi < (row + 1) * S ? batch_page_hi : (row + 1) * S;
                        for (uint32_t page = page_lo; page < page_hi; ++page) {
                            const uint32_t s = page - row * S;
                            const uint32_t src_page = ((b * M + m) * W + j) * S + s;
                            for (uint32_t offset = 0; offset < page_bytes; offset += chunk_bytes) {
                                if (!in_batch(page, offset, dst)) {
                                    continue;
                                }
                                const uint32_t bytes =
                                    page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
                                // An overwrite is read straight into the copy.
                                const uint32_t into = reduce == kNone ? dst : src_addr;
                                noc.async_read(
                                    PrecomposedUnicastEndpoint{},
                                    CoreLocalMem<uint32_t>(into),
                                    bytes,
                                    {.noc_addr = source.get_noc_addr(src_page, offset)},
                                    {.offset_bytes = 0});
                                if constexpr (reduce != kNone) {
                                    noc.async_read_barrier();
                                    apply(dst, src_addr, bytes);
                                }
                            }
                        }
                    }
                    if constexpr (reduce == kNone) {
                        noc.async_read_barrier();
                    }
                }
                if constexpr (column && !stream_window) {
                    stream.pop_front(1);
                }
                index.pop_front(1);
            }
        }

        // The batch goes out once.
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t q = q0 + i;
            const uint32_t page = first_page + q / C;
            const uint32_t offset = (q % C) * chunk_bytes;
            const uint32_t bytes = page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
            noc.async_write(rows, output, bytes, {.offset_bytes = i * entry}, {.page_id = page, .offset_bytes = offset});
        }
        noc.async_write_barrier();
        rows.pop_front(count);
    }
}
