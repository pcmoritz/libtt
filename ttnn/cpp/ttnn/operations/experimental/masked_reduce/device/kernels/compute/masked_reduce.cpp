// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Everything happens on the SFPU in FP32, so the compare is exact for FP32
// inputs (the FPU's adds are not): the threshold tile, pre-broadcast across its
// columns by the reader, is loaded into DST once; then for every input tile,
// d = x - t, mask = d <cmp> 0 as 1.0 / 0.0, and the accumulator tile gains the
// mask (Count) or where(mask, x, 0) (Sum; a select, not a multiply, so a masked
// infinity adds nothing). The whole run accumulates in one DST tile
// within one register acquire; the accumulator is packed once, as this core's
// partial. Summing a partial tile's columns, over the row's cores, gives the
// row's result.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t compare_gt = 0, compare_ge = 1, compare_lt = 2, compare_le = 3;
constexpr uint32_t mode_count = 0, mode_sum = 1;
constexpr uint32_t dst_acc = 0, dst_t = 1, dst_x = 2, dst_mask = 3, dst_valid = 4;

template <uint32_t compare>
inline void mask_init() {
    if constexpr (compare == compare_gt) {
        gtz_tile_init();
    } else if constexpr (compare == compare_ge) {
        gez_tile_init();
    } else if constexpr (compare == compare_lt) {
        ltz_tile_init();
    } else {
        lez_tile_init();
    }
}

template <uint32_t compare>
inline void mask_tile(uint32_t dst) {
    if constexpr (compare == compare_gt) {
        gtz_tile(dst);
    } else if constexpr (compare == compare_ge) {
        gez_tile(dst);
    } else if constexpr (compare == compare_lt) {
        ltz_tile(dst);
    } else {
        lez_tile(dst);
    }
}
}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t compare = get_arg(args::compare);
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr uint32_t last_valid = get_arg(args::last_valid);
    const uint32_t has_last = get_arg(args::has_last);

    DataflowBuffer in_dfb(dfb::in);
    DataflowBuffer t_dfb(dfb::t);
    DataflowBuffer v_dfb(dfb::v);
    DataflowBuffer out_dfb(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::t, dfb::out);
    copy_init(dfb::in);
    sub_binary_tile_init();
    mask_init<compare>();
    add_binary_tile_init();
    if constexpr (last_valid != 0) {
        mul_binary_tile_init();
    }
    if constexpr (mode == mode_sum) {
        where_tile_init();
    }

    tile_regs_acquire();
    t_dfb.wait_front(1);
    copy_tile(dfb::t, 0, dst_t);
    t_dfb.pop_front(1);
    // A width that is not whole tiles: the last tile's padding columns must
    // not count, so their mask is multiplied by the validity tile.
    const bool masks_last = last_valid != 0 && has_last;
    if (masks_last) {
        v_dfb.wait_front(1);
        copy_tile(dfb::v, 0, dst_valid);
        v_dfb.pop_front(1);
    }
    for (uint32_t i = 0; i < num_tiles; ++i) {
        in_dfb.wait_front(1);
        copy_tile(dfb::in, 0, dst_x);
        in_dfb.pop_front(1);
        // The first tile's mask starts the accumulator.
        const uint32_t dst = i == 0 ? dst_acc : dst_mask;
        sub_binary_tile(dst_x, dst_t, dst);
        mask_tile<compare>(dst);
        if (masks_last && i + 1 == num_tiles) {
            mul_binary_tile(dst, dst_valid, dst);
        }
        if constexpr (mode == mode_sum) {
            // DST holds FP32; where mask is nonzero take x, else the mask's 0.
            where_tile<DataFormat::Float32>(dst, dst_x, dst, dst);
        }
        if (i != 0) {
            add_binary_tile(dst_acc, dst_mask, dst_acc);
        }
    }
    tile_regs_commit();

    out_dfb.reserve_back(1);
    tile_regs_wait();
    pack_tile(dst_acc, dfb::out);
    tile_regs_release();
    out_dfb.push_back(1);
}
