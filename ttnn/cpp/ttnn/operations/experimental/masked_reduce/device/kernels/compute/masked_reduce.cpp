// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Everything happens on the SFPU in FP32, so the compare is exact for FP32
// inputs (the FPU's adds are not). The threshold tile, whose column 0 holds
// each dense row's threshold, is loaded into DST once and spread across its
// columns; then for every dense tile, d = x - t, mask = d <cmp> 0 as 1.0 / 0.0,
// and the accumulator tile gains the mask (Count) or where(mask, x, 0) (Sum; a
// select, not a multiply, so a masked infinity adds nothing). The run's last
// dense tile may hold fewer input tiles, and the row's last input tile padding
// columns: those lanes of its mask are zeroed. The whole run accumulates in one
// DST tile within one register acquire; at the end each row of the
// accumulator is summed into its column 16, and the tile is packed once for
// the writer. A tile row's master core then adds up the partial tiles the
// row's cores sent (their row sums in 16-byte slots, see the writer), sums
// each row of the total into its column 16 and packs it for the writer.
//
// The modes of a chunked input's [N, 32] matrices combine their values
// differently: Total adds every element, Max keeps the largest and ArgFirst
// the smallest index of an element that compares true (as a float; indices
// stay below 2^24), with tile padding filled with 0, -inf and 2^24. The
// accumulator, the rows and the partials combine the same way.
//
// SFPU lane layout (fp32 DST): vector d (0..7) of a face holds, in lane l
// (0..31), the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2; the tile-id register holds 2 * l.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef TRISC_MATH
#include "sfpi.h"

namespace mr {

// Row within its 4-row band (l / 8) and twice the column within its 8-column
// group (2 * (l % 8)) of each lane.
sfpi_inline sfpi::vInt lane_row() { return sfpi::vInt(sfpi::vConstTileId) >> 4; }
sfpi_inline sfpi::vInt lane_col2() { return sfpi::vInt(sfpi::vConstTileId) & 14; }

// How values combine: added, or the larger or the smaller kept.
constexpr uint32_t combine_add = 0, combine_max = 1, combine_min = 2;

template <uint32_t op>
sfpi_inline sfpi::vFloat combine(sfpi::vFloat a, sfpi::vFloat b) {
    if constexpr (op == combine_add) {
        return a + b;
    } else if constexpr (op == combine_max) {
        v_if(b > a) { a = b; }
        v_endif;
        return a;
    } else {
        v_if(b < a) { a = b; }
        v_endif;
        return a;
    }
}

// Every lane of each 8-lane group gets the group's combination.
template <uint32_t op>
sfpi_inline sfpi::vFloat combine_lanes8(sfpi::vFloat v) {
    v = combine<op>(v, sfpi::vFloat(sfpi::subvec_shflror1(v)));
    sfpi::vFloat r = sfpi::vFloat(sfpi::subvec_shflror1(v));
    v = combine<op>(v, sfpi::vFloat(sfpi::subvec_shflror1(r)));
    r = sfpi::vFloat(sfpi::subvec_shflror1(v));
    r = sfpi::vFloat(sfpi::subvec_shflror1(r));
    r = sfpi::vFloat(sfpi::subvec_shflror1(r));
    v = combine<op>(v, sfpi::vFloat(sfpi::subvec_shflror1(r)));
    return v;
}
sfpi_inline sfpi::vFloat sum_lanes8(sfpi::vFloat v) { return combine_lanes8<combine_add>(v); }

// Column 0 of band `d` (vector 2d, the left face's even columns, lane 8 * i
// holding band row i's column 0) spread over the band's 8-lane groups.
sfpi_inline sfpi::vFloat band_column0(uint32_t d) {
    sfpi::vFloat v = sfpi::dst_reg[2 * d];
    v_if(lane_col2() != 0) { v = 0.0f; }
    v_endif;
    return sum_lanes8(v);
}

// Both vectors (even and odd columns) of band `d` of the current face set to v.
sfpi_inline void store_band(uint32_t d, sfpi::vFloat v) {
    sfpi::dst_reg[2 * d] = v;
    sfpi::dst_reg[2 * d + 1] = v;
}

// Column 0 of DST tile `dst` spread across all 32 columns of each row. SFPU
// vectors cannot be spilled, so the four bands are four named registers.
inline void spread_column0(uint32_t dst) {
    _llk_math_eltwise_sfpu_start_(dst);
    for (uint32_t pair = 0; pair < 2; ++pair) {
        sfpi::vFloat b0 = band_column0(0);
        sfpi::vFloat b1 = band_column0(1);
        sfpi::vFloat b2 = band_column0(2);
        sfpi::vFloat b3 = band_column0(3);
        store_band(0, b0);
        store_band(1, b1);
        store_band(2, b2);
        store_band(3, b3);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();  // the right face, same rows
        store_band(0, b0);
        store_band(1, b1);
        store_band(2, b2);
        store_band(3, b3);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}

// Fill the tile's lanes of rows from `rows_valid` on, and of columns from
// `cols_valid` on in rows row_lo .. row_hi - 1, with 0 (or -inf for max).
template <uint32_t op = combine_add>
inline void zero_invalid(uint32_t dst, uint32_t rows_valid, uint32_t row_lo, uint32_t row_hi, uint32_t cols_valid) {
    const sfpi::vFloat fill = op == combine_max ? sfpi::vFloat(-__builtin_inff()) : sfpi::vFloat(0.0f);
    _llk_math_eltwise_sfpu_start_(dst);
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t d = 0; d < 8; ++d) {
            const sfpi::vInt row = lane_row() + static_cast<int>(16 * (face >> 1) + 4 * (d >> 1));
            const sfpi::vInt col = lane_col2() + static_cast<int>(16 * (face & 1) + (d & 1));
            v_if(row >= static_cast<int>(rows_valid)) { sfpi::dst_reg[0] = fill; }
            v_endif;
            v_if(row >= static_cast<int>(row_lo)) {
                v_if(row < static_cast<int>(row_hi)) {
                    v_if(col >= static_cast<int>(cols_valid)) { sfpi::dst_reg[0] = fill; }
                    v_endif;
                }
                v_endif;
            }
            v_endif;
            sfpi::dst_reg++;
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}

// The combination of band `d`'s two vectors (even and odd columns) of the
// current face.
template <uint32_t op>
sfpi_inline sfpi::vFloat band_combine(uint32_t d) {
    return combine<op>(sfpi::vFloat(sfpi::dst_reg[2 * d]), sfpi::vFloat(sfpi::dst_reg[2 * d + 1]));
}

// Each row's 32 columns combined into column 16 (the right face's column 0).
template <uint32_t op>
inline void combine_rows(uint32_t dst) {
    _llk_math_eltwise_sfpu_start_(dst);
    for (uint32_t pair = 0; pair < 2; ++pair) {
        sfpi::vFloat b0 = band_combine<op>(0);
        sfpi::vFloat b1 = band_combine<op>(1);
        sfpi::vFloat b2 = band_combine<op>(2);
        sfpi::vFloat b3 = band_combine<op>(3);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();  // the right face, same rows
        b0 = combine<op>(b0, band_combine<op>(0));
        b1 = combine<op>(b1, band_combine<op>(1));
        b2 = combine<op>(b2, band_combine<op>(2));
        b3 = combine<op>(b3, band_combine<op>(3));
        sfpi::dst_reg[0] = combine_lanes8<op>(b0);
        sfpi::dst_reg[2] = combine_lanes8<op>(b1);
        sfpi::dst_reg[4] = combine_lanes8<op>(b2);
        sfpi::dst_reg[6] = combine_lanes8<op>(b3);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}

// The mask tile's 1.0 lanes replaced by their element's index in a chunked
// matrix (base + 32 * row + column, base the tile's first), its 0.0 lanes by
// 2^24.
inline void arg_index(uint32_t dst, int base) {
    _llk_math_eltwise_sfpu_start_(dst);
    for (uint32_t face = 0; face < 4; ++face) {
        for (uint32_t d = 0; d < 8; ++d) {
            const sfpi::vInt row = lane_row() + static_cast<int>(16 * (face >> 1) + 4 * (d >> 1));
            const sfpi::vInt col = lane_col2() + static_cast<int>(16 * (face & 1) + (d & 1));
            const sfpi::vInt index = (row << 5) + col + base;
            const sfpi::vFloat value = sfpi::vFloat(__builtin_rvtt_sfpcast(index.get(), 0));
            sfpi::vFloat out = sfpi::vFloat(16777216.0f);
            v_if(sfpi::vFloat(sfpi::dst_reg[0]) != 0.0f) { out = value; }
            v_endif;
            sfpi::dst_reg[0] = out;
            sfpi::dst_reg++;
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace mr
#endif

namespace {
constexpr uint32_t compare_gt = 0, compare_ge = 1, compare_lt = 2, compare_le = 3;
constexpr uint32_t mode_count = 0, mode_sum = 1, mode_total = 2, mode_max = 3, mode_arg_first = 4;
constexpr uint32_t dst_acc = 0, dst_t = 1, dst_x = 2, dst_mask = 3;

// How a mode's values combine (see mr::combine).
constexpr uint32_t combine_of(uint32_t mode) {
    return mode == mode_max ? 1 : mode == mode_arg_first ? 2 : 0;
}

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

inline void custom_sfpu_init() {
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
}

// The SFPU set-up of the mode's per-tile ops, again after a custom function.
template <uint32_t compare, uint32_t mode>
inline void ops_init() {
    if constexpr (mode == mode_count || mode == mode_sum || mode == mode_arg_first) {
        sub_binary_tile_init();
        mask_init<compare>();
    }
    if constexpr (mode == mode_sum) {
        where_tile_init();
    }
    if constexpr (mode == mode_max) {
        binary_max_tile_init();
    } else if constexpr (mode == mode_arg_first) {
        binary_min_tile_init();
    } else {
        add_binary_tile_init();
    }
}

// acc = acc (+, max or min) other.
template <uint32_t mode>
inline void combine_tiles(uint32_t acc, uint32_t other) {
    if constexpr (mode == mode_max) {
        binary_max_tile(acc, other, acc);
    } else if constexpr (mode == mode_arg_first) {
        binary_min_tile(acc, other, acc);
    } else {
        add_binary_tile(acc, other, acc);
    }
}
}  // namespace

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t rows = get_arg(args::rows);
    const uint32_t stride = get_arg(args::stride);
    const uint32_t pack = get_arg(args::pack);
    const uint32_t has_last = get_arg(args::has_last);
    // Data rows of the run's last whole tile: fewer only at a chunked
    // matrix's end.
    const uint32_t last_rows = get_arg(args::last_rows);
    // The run's first tile in its chunked matrix (ArgFirst numbers elements).
    const uint32_t first_in_row = get_arg(args::first_in_row);
    const uint32_t is_master = get_arg(args::is_master);
    constexpr uint32_t compare = get_arg(args::compare);
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr uint32_t last_valid = get_arg(args::last_valid);
    constexpr uint32_t partial_tiles = get_arg(args::partial_tiles);
    constexpr uint32_t op = combine_of(mode);
    // Total and Max take every element: no threshold, no mask.
    constexpr bool masked = mode != mode_total && mode != mode_max;

    DataflowBuffer in_dfb(dfb::in);
    DataflowBuffer t_dfb(dfb::t);
    DataflowBuffer out_dfb(dfb::out);
    DataflowBuffer part_dfb(dfb::part);

    compute_kernel_hw_startup(dfb::in, dfb::t, dfb::out);
    copy_init(dfb::in);

    const uint32_t dense_tiles = (num_tiles + pack - 1) / pack;
    // Input tiles in the last dense tile, and the rows of the row's last tile.
    const uint32_t in_last = num_tiles - (dense_tiles - 1) * pack;
    const uint32_t rows_valid = pack == 1 ? (has_last ? last_rows : 32) : in_last * stride;
    const uint32_t last_lo = (in_last - 1) * stride;
    const uint32_t last_hi = last_lo + rows;
    const uint32_t cols_valid = has_last && last_valid != 0 ? last_valid : 32;
    const bool trims_last = rows_valid != 32 || cols_valid != 32;

    tile_regs_acquire();
    t_dfb.wait_front(1);
    if constexpr (masked) {
        copy_tile(dfb::t, 0, dst_t);
        custom_sfpu_init();
        MATH((mr::spread_column0(dst_t)));
    }
    t_dfb.pop_front(1);

    ops_init<compare, mode>();
    for (uint32_t i = 0; i < dense_tiles; ++i) {
        in_dfb.wait_front(1);
        const bool last = trims_last && i + 1 == dense_tiles;
        if constexpr (!masked) {
            // The tile itself, padding filled with the combination's identity.
            const uint32_t dst = i == 0 ? dst_acc : dst_x;
            copy_tile(dfb::in, 0, dst);
            in_dfb.pop_front(1);
            if (last) {
                custom_sfpu_init();
                MATH((mr::zero_invalid<op>(dst, rows_valid, last_lo, last_hi, cols_valid)));
                ops_init<compare, mode>();
            }
            if (i != 0) {
                combine_tiles<mode>(dst_acc, dst_x);
            }
            continue;
        }
        copy_tile(dfb::in, 0, dst_x);
        in_dfb.pop_front(1);
        // The first tile's mask starts the accumulator.
        const uint32_t dst = i == 0 ? dst_acc : dst_mask;
        sub_binary_tile(dst_x, dst_t, dst);
        mask_tile<compare>(dst);
        if (last || mode == mode_arg_first) {
            custom_sfpu_init();
            if (last) {
                MATH((mr::zero_invalid(dst, rows_valid, last_lo, last_hi, cols_valid)));
            }
            if constexpr (mode == mode_arg_first) {
                MATH((mr::arg_index(dst, static_cast<int>((first_in_row + i) * 1024))));
            }
            ops_init<compare, mode>();
        }
        if constexpr (mode == mode_sum) {
            // DST holds FP32; where mask is nonzero take x, else the mask's 0.
            where_tile<DataFormat::Float32>(dst, dst_x, dst, dst);
        }
        if (i != 0) {
            combine_tiles<mode>(dst_acc, dst_mask);
        }
    }
    custom_sfpu_init();
    MATH((mr::combine_rows<op>(dst_acc)));
    tile_regs_commit();

    out_dfb.reserve_back(1);
    tile_regs_wait();
    pack_tile(dst_acc, dfb::out);
    tile_regs_release();
    out_dfb.push_back(1);
    if (!is_master) {
        return;
    }

    // The row's partials, FP32 tiles combined exactly on the SFPU.
    part_dfb.wait_front(partial_tiles);
    tile_regs_acquire();
    // The partials are FP32 whatever the input: switch the unpacker over.
    reconfig_data_format_srca(dfb::in, dfb::part);
    copy_init(dfb::part);
    copy_tile(dfb::part, 0, dst_acc);
    ops_init<compare, mode>();
    for (uint32_t i = 1; i < partial_tiles; ++i) {
        copy_tile(dfb::part, i, dst_x);
        combine_tiles<mode>(dst_acc, dst_x);
    }
    part_dfb.pop_front(partial_tiles);
    custom_sfpu_init();
    MATH((mr::combine_rows<op>(dst_acc)));
    tile_regs_commit();

    out_dfb.reserve_back(1);
    tile_regs_wait();
    pack_tile(dst_acc, dfb::out);
    tile_regs_release();
    out_dfb.push_back(1);
}
