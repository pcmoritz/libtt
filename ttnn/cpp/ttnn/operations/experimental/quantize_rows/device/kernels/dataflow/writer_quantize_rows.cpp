// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Writes this core's dense float8 tiles back to the row's tiles: dense rows
// jj * stride .. jj * stride + stride - 1 are rows 0 .. stride - 1 of input
// tile jj, one chunk per left/right face pair (whole tiles for a stride of
// 32). The tiles' other rows are padding and keep whatever they held. The
// first core also writes the scale tile.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

namespace {
// Float8 tiles: one byte per element.
constexpr uint32_t face_rows = 16;
constexpr uint32_t row_bytes = 16;
constexpr uint32_t face_bytes = 256;
constexpr uint32_t tile_bytes = 4 * face_bytes;
constexpr uint32_t scale_tile_bytes = 4096;
}  // namespace

void kernel_main() {
    const uint32_t first_dense = get_arg(args::first_dense);
    const uint32_t num_dense = get_arg(args::num_dense);
    const uint32_t last_count = get_arg(args::last_count);
    const uint32_t writes_scale = get_arg(args::writes_scale);
    constexpr uint32_t stride = get_arg(args::stride);
    constexpr uint32_t pack = get_arg(args::pack);
    constexpr uint32_t chunk_tiles = get_arg(args::chunk_tiles);

    DataflowBuffer out_dfb(dfb::out);
    DataflowBuffer scale_dfb(dfb::scale);
    const auto q = TensorAccessor(tensor::q);
    const auto scale = TensorAccessor(tensor::scale);
    Noc noc;

    // The compute kernel pushes its dense tiles chunk_tiles at a time.
    for (uint32_t first = 0; first < num_dense; first += chunk_tiles) {
        const uint32_t n = num_dense - first < chunk_tiles ? num_dense - first : chunk_tiles;
        out_dfb.wait_front(n);
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t d = first + j;
            const uint32_t count = d + 1 == num_dense ? last_count : pack;
            for (uint32_t jj = 0; jj < count; ++jj) {
                const uint32_t page = (first_dense + d) * pack + jj;
                if constexpr (pack == 1) {
                    noc.async_write(out_dfb, q, tile_bytes, {.offset_bytes = j * tile_bytes}, {.page_id = page});
                } else {
                    const uint32_t r0 = jj * stride;
                    const uint32_t face = r0 < face_rows ? 0 : 2;
                    const uint32_t at = j * tile_bytes + face * face_bytes + (r0 % face_rows) * row_bytes;
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc.async_write(
                            out_dfb,
                            q,
                            stride * row_bytes,
                            {.offset_bytes = at + f * face_bytes},
                            {.page_id = page, .offset_bytes = f * face_bytes});
                    }
                }
            }
        }
        noc.async_write_barrier();
        out_dfb.pop_front(n);
    }
    if (writes_scale) {
        scale_dfb.wait_front(1);
        noc.async_write(scale_dfb, scale, scale_tile_bytes, {.offset_bytes = 0}, {.page_id = 0});
        noc.async_write_barrier();
        scale_dfb.pop_front(1);
    }
}
