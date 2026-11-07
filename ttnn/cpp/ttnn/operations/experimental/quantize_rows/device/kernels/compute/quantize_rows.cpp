// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Everything happens on the SFPU in FP32 DST (quantize_rows_sfpu.h).
//
// Phase 1, this core's row maxima: dense band g (rows 4g .. 4g + 3) of a dense
// tile holds rows (4g .. 4g + 3) % stride of one input tile, so the dense
// tiles fold into one accumulator tile, which goes to the reader.
//
// Phase 2: the divisor from every core's maxima in the gather buffer; the
// dense tiles are then divided by it and converted to float8, chunk_tiles at
// a time. DST holds the divisor and three dense tiles, and the divisor is
// made again for every chunk. The first core also packs the divisor tile as
// the scale: row m holds scale[m] in every column.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/quantize_rows/device/kernels/compute/quantize_rows_sfpu.h"

namespace {
// Phase 1: the accumulator and a dense tile. Phase 2: the divisor, then a
// gather tile and the fp8_max tile while making it, then up to three dense
// tiles.
constexpr uint32_t dst_acc = 0, dst_x = 1;
constexpr uint32_t dst_d = 0, dst_t = 1, dst_c = 2, dst_x0 = 1;
}  // namespace

void kernel_main() {
    const uint32_t num_dense = get_arg(args::num_dense);
    const uint32_t last_count = get_arg(args::last_count);
    const uint32_t writes_scale = get_arg(args::writes_scale);
    constexpr uint32_t stride = get_arg(args::stride);
    constexpr uint32_t rows = get_arg(args::rows);
    constexpr uint32_t pack = get_arg(args::pack);
    constexpr uint32_t cores = get_arg(args::cores);
    constexpr uint32_t gather_tiles = get_arg(args::gather_tiles);
    constexpr uint32_t chunk_tiles = get_arg(args::chunk_tiles);
    constexpr uint32_t e5m2 = get_arg(args::e5m2);
    constexpr uint32_t fp8_max_bits = get_arg(args::fp8_max_bits);
    constexpr uint32_t min_scale_bits = get_arg(args::min_scale_bits);
    constexpr uint32_t groups = stride / 4;
    static_assert(chunk_tiles <= 3, "DST holds the divisor and three dense tiles");
    constexpr uint32_t total_bands = cores * groups;

    DataflowBuffer in_dfb(dfb::in);
    DataflowBuffer part_dfb(dfb::part);
    DataflowBuffer gather_dfb(dfb::gather);
    DataflowBuffer out_dfb(dfb::out);
    DataflowBuffer scale_dfb(dfb::scale);

    compute_kernel_hw_startup(dfb::in, dfb::in, dfb::part);
    copy_init(dfb::in);
    in_dfb.wait_front(num_dense);

    // Phase 1: this core's row maxima, for the reader to share.
    tile_regs_acquire();
    qr::sfpu_init();
    MATH((qr::zero_acc<groups, dst_acc>()));
    for (uint32_t i = 0; i < num_dense; ++i) {
        copy_tile(dfb::in, i, dst_x);
        const uint32_t bands = (i + 1 == num_dense ? last_count : pack) * stride / 4;
        MATH((qr::fold_absmax<groups, dst_x, dst_acc>(bands)));
    }
    MATH((qr::finish_partial<groups, dst_acc>()));
    tile_regs_commit();
    part_dfb.reserve_back(1);
    tile_regs_wait();
    pack_tile(dst_acc, dfb::part);
    tile_regs_release();
    part_dfb.push_back(1);

    // Phase 2: the divisor from every core's maxima, then the quotients.
    gather_dfb.wait_front(gather_tiles);
    for (uint32_t first = 0; first < num_dense; first += chunk_tiles) {
        const uint32_t n = num_dense - first < chunk_tiles ? num_dense - first : chunk_tiles;
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::in, dfb::gather);
        qr::make_divisor<groups, dst_d, dst_t, dst_c>(
            dfb::gather, gather_tiles, total_bands, fp8_max_bits, min_scale_bits);
        reconfig_data_format_srca(dfb::gather, dfb::in);
        qr::quantize_tiles<e5m2 != 0, dst_d, dst_x0, 32, rows>(dfb::in, first, n);
        tile_regs_commit();

        tile_regs_wait();
        if (first == 0 && writes_scale) {
            scale_dfb.reserve_back(1);
            pack_reconfig_data_format(dfb::scale);
            pack_tile(dst_d, dfb::scale);
            scale_dfb.push_back(1);
        }
        out_dfb.reserve_back(n);
        pack_reconfig_data_format(dfb::out);
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(dst_x0 + j, dfb::out);
        }
        out_dfb.push_back(n);
        tile_regs_release();
    }
    gather_dfb.pop_front(gather_tiles);
    in_dfb.pop_front(num_dense);
}
