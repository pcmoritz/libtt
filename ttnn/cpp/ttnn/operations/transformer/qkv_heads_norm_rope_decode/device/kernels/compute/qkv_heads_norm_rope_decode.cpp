// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"

// For every tile row of stacked q and k heads, one head per row:
// out = rotary(x / sqrt(mean(x^2) + eps) * weight), with a weight row per
// head and out = y * cos + rotate_half(y) * sin, where rotate_half(y) =
// [-y2, y1] for the halves y1, y2 of a head (whole tiles) and cos and sin are
// row 0 for every row. Row tiles below q_row_tiles go to the q output, the
// rest to the k stage.
void kernel_main() {
    constexpr uint32_t cb_heads = get_named_compile_time_arg_val("cb_heads");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t cb_cos = get_named_compile_time_arg_val("cb_cos");
    constexpr uint32_t cb_sin = get_named_compile_time_arg_val("cb_sin");
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_squares = get_named_compile_time_arg_val("cb_squares");
    constexpr uint32_t cb_rstd = get_named_compile_time_arg_val("cb_rstd");
    constexpr uint32_t cb_normed = get_named_compile_time_arg_val("cb_normed");
    constexpr uint32_t cb_weighted = get_named_compile_time_arg_val("cb_weighted");
    constexpr uint32_t cb_cos_part = get_named_compile_time_arg_val("cb_cos_part");
    constexpr uint32_t cb_sin_part = get_named_compile_time_arg_val("cb_sin_part");
    constexpr uint32_t cb_q_out = get_named_compile_time_arg_val("cb_q_out");
    constexpr uint32_t cb_k_stage = get_named_compile_time_arg_val("cb_k_stage");
    constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");
    constexpr uint32_t row_tiles = get_named_compile_time_arg_val("row_tiles");
    constexpr uint32_t q_row_tiles = get_named_compile_time_arg_val("q_row_tiles");
    constexpr uint32_t eps = get_named_compile_time_arg_val("eps");
    constexpr uint32_t inv_head_dim = get_named_compile_time_arg_val("inv_head_dim");
    constexpr uint32_t half = head_tiles / 2;

    // One tile op per DST acquire for tiles j of a head row, packed as tile j
    // of `cb`.
    auto each_tile = [&](uint32_t cb, auto op) {
        cb_reserve_back(cb, head_tiles);
        pack_reconfig_data_format(cb);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            tile_regs_acquire();
            op(j);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb, j);
            tile_regs_release();
        }
        cb_push_back(cb, head_tiles);
    };

    compute_kernel_hw_startup(cb_heads, cb_heads, cb_squares);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_cos, head_tiles);
    cb_wait_front(cb_sin, head_tiles);
    cb_wait_front(cb_heads, row_tiles * head_tiles);
    cb_wait_front(cb_weight, row_tiles * head_tiles);
    for (uint32_t r = 0; r < row_tiles; ++r) {
        const uint32_t base = r * head_tiles;

        reconfig_data_format(cb_heads, cb_heads);
        mul_tiles_init(cb_heads, cb_heads);
        each_tile(cb_squares, [&](uint32_t j) { mul_tiles(cb_heads, cb_heads, base + j, base + j, 0); });

        // rstd = rsqrt(sum / head_dim + eps), one per row, down column 0. A
        // row reduce unpacks the scaler first; set the formats in that order.
        cb_wait_front(cb_squares, head_tiles);
        cb_reserve_back(cb_rstd, 1);
        reconfig_data_format(cb_scaler, cb_squares);
        pack_reconfig_data_format(cb_rstd);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_squares, cb_scaler, cb_rstd);
        tile_regs_acquire();
        for (uint32_t j = 0; j < head_tiles; ++j) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_squares, cb_scaler, j, 0, 0);
        }
        reduce_uninit(cb_squares);
        // The SFPU's column mode covers faces 0 and 2, two rows per
        // iteration: all 32 rows.
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
            (APPROX, ckernel::sfpu::MUL, 8, DST_ACCUM_MODE), 0, VectorMode::C, inv_head_dim));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
            (APPROX, ckernel::sfpu::ADD, 8, DST_ACCUM_MODE), 0, VectorMode::C, eps));
        rsqrt_tile_init();
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 8, DST_ACCUM_MODE, false, false), 0,
            VectorMode::C));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rstd, 0);
        tile_regs_release();
        cb_push_back(cb_rstd, 1);
        cb_pop_front(cb_squares, head_tiles);

        // normed = x * rstd (one per row)
        cb_wait_front(cb_rstd, 1);
        reconfig_data_format(cb_heads, cb_rstd);
        mul_bcast_cols_init_short(cb_heads, cb_rstd);
        each_tile(cb_normed, [&](uint32_t j) { mul_tiles_bcast_cols(cb_heads, cb_rstd, base + j, 0, 0); });
        cb_pop_front(cb_rstd, 1);

        // weighted = normed * weight (a row per head)
        cb_wait_front(cb_normed, head_tiles);
        reconfig_data_format(cb_normed, cb_weight);
        mul_tiles_init(cb_normed, cb_weight);
        each_tile(cb_weighted, [&](uint32_t j) { mul_tiles(cb_normed, cb_weight, j, base + j, 0); });
        cb_pop_front(cb_normed, head_tiles);

        // cos_part = weighted * cos, sin_part = weighted[the other half] * sin
        cb_wait_front(cb_weighted, head_tiles);
        reconfig_data_format(cb_weighted, cb_cos);
        mul_bcast_rows_init_short(cb_weighted, cb_cos);
        each_tile(cb_cos_part, [&](uint32_t j) { mul_tiles_bcast_rows(cb_weighted, cb_cos, j, j, 0); });
        reconfig_data_format(cb_weighted, cb_sin);
        mul_bcast_rows_init_short(cb_weighted, cb_sin);
        each_tile(cb_sin_part, [&](uint32_t j) {
            mul_tiles_bcast_rows(cb_weighted, cb_sin, (j + half) % head_tiles, j, 0);
        });
        cb_pop_front(cb_weighted, head_tiles);

        // out = cos_part - sin_part in the first half, + in the second.
        cb_wait_front(cb_cos_part, head_tiles);
        cb_wait_front(cb_sin_part, head_tiles);
        reconfig_data_format(cb_cos_part, cb_sin_part);
        each_tile(r < q_row_tiles ? cb_q_out : cb_k_stage, [&](uint32_t j) {
            if (j < half) {
                sub_tiles_init(cb_cos_part, cb_sin_part);
                sub_tiles(cb_cos_part, cb_sin_part, j, j, 0);
            } else {
                add_tiles_init(cb_cos_part, cb_sin_part);
                add_tiles(cb_cos_part, cb_sin_part, j, j, 0);
            }
        });
        cb_pop_front(cb_cos_part, head_tiles);
        cb_pop_front(cb_sin_part, head_tiles);
    }
    cb_pop_front(cb_heads, row_tiles * head_tiles);
    cb_pop_front(cb_weight, row_tiles * head_tiles);
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_cos, head_tiles);
    cb_pop_front(cb_sin, head_tiles);
}
