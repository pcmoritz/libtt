// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// FP32 matrix multiply on the vector unit: C = A * B for 32x32 tiles in Dst
// ============================================================================
//
// The matrix unit reads FP32 matmul operands as TF32, so its products carry
// about 10 bits of mantissa. The vector unit's SFPMAD is an FP32 multiply-add
// with a single rounding, which gives FP32 results.
//
// Dst layout (Dst32b, as addressed by SFPLOAD / SFPSTORE). A tile is 64
// address rows: faces 0..3 (rows 0-15 x cols 0-15, rows 0-15 x cols 16-31,
// rows 16-31 x cols 0-15, rows 16-31 x cols 16-31) of 16 rows each. One
// SFPLOAD at address `addr` reads rows (addr & ~3) .. +3 of the face and its
// even columns, or its odd columns if addr & 2, into the 32 lanes:
//
//     lane 8 * i + j  <-  row (addr & ~3) + i, face column 2 * j + ((addr >> 1) & 1)
//
// so a register holds a 4x8 block (a "slot"); the 32 slots of a tile are at
// even addresses 0..62. Sub-row i has the 8 tile columns col(g, j) = 16 * (g >> 1)
// + 2 * j + (g & 1) of column group g; groups 0..3 together cover a row.
//
// Cross-lane movement is what costs on the vector unit, so the product is
// formed from operands that need little of it:
//
//   X_g = row k of B for column group g, the same in all four sub-rows
//         (lane j: B[k][col(g, j)]). SFPTRANSP brings sub-row k % 4 of the
//         slot holding it to sub-row 0, and SFPCONFIG broadcasts that to all
//         sub-rows of LREG[11 + g].
//   R   = column k of A in all 32 lanes (lane 8 * h + j: A[col(h, j)][k]):
//         SFPTRANSP of the four slots of row k of A^T.
//
// With R rotated right by s lanes within each group of 8 (SFPSHFT2), the
// lanewise product X_g * R is A[col(h, j - s)][k] * B[k][col(g, j)], a term
// of C[col(h, j - s)][col(g, j)]. The 32 combinations of g and s = 0..7
// cover all of C, so each k is 32 SFPMADs on 32 accumulator registers
// acc(g, s), kept in a Dst tile in this skewed order. matmul_fp32_finish
// then gathers the natural C tile from it.
//
// LREG11 holds -1.0 for SFPI-compiled code; it is restored before returning.

constexpr std::uint32_t MATMUL_FP32_TILE_ROWS = 64;  // Dst address rows per 32x32 tile
constexpr std::uint32_t MATMUL_FP32_REPLAY_START = 0;
constexpr std::uint32_t MATMUL_FP32_REPLAY_LEN = 5;

// Dst address (relative to the tile) of the slot holding tile row `row` and tile column `col`.
inline std::uint32_t _matmul_fp32_slot_(std::uint32_t row, std::uint32_t col) {
    return 16 * (2 * (row >> 4) + (col >> 4)) + (row & 12) + ((col & 1) << 1);
}

// First tile column of column group g.
inline constexpr std::uint32_t _matmul_fp32_group_col_(std::uint32_t g) { return 16 * (g >> 1) + (g & 1); }

// Address of accumulator acc(g, s) in the accumulator tile.
inline constexpr std::uint32_t _matmul_fp32_acc_(std::uint32_t g, std::uint32_t s) { return 2 * (8 * g + s); }

inline void _init_matmul_fp32_() {
    _init_sfpu_config_reg();
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
    // Steps the Dst address counter to the next accumulator column s.
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6);

    // Load macro g = load acc(g, s) into LREG g, then (at the MAD and store
    // sub-units, in the same cycles as the following loads) LREG g = X_g * R +
    // LREG g and store it back: one accumulator per cycle.
    //
    // t  | Load     | MAD                      | Store
    // -- | -------- | ------------------------ | --------
    //  0 | [g] acc  |                          |
    //  1 |          | [g] = X_g * R + [g]      |
    //  3 |          |                          | [g] acc
    //
    // InstructionTemplate[g]: SFPMAD(VA = LREG[11 + g], VB = R); the macro
    // replaces VC and VD with the loaded register. (VD = 12 + g writes the
    // template instead of executing.)
    TTI_SFPMAD(p_sfpu::LREG11, p_sfpu::LREG4, 0, 12, 0);
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG4, 0, 13, 0);
    TTI_SFPMAD(p_sfpu::LREG13, p_sfpu::LREG4, 0, 14, 0);
    TTI_SFPMAD(p_sfpu::LREG14, p_sfpu::LREG4, 0, 15, 0);
    for (std::uint32_t g = 0; g < 4; ++g) {
        // Per sub-unit: bits 0-2 select the instruction (template 0..3 = 4..7,
        // SFPSTORE = 3), bits 3-5 the delay in cycles.
        const std::uint32_t mad_bits = (0 << 3) | (4 + g);
        const std::uint32_t store_bits = (2 << 3) | 3;
        TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, mad_bits << 8);
        TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, store_bits << 8);
        TT_SFPCONFIG(0, 4 + g, 0);
    }
    // Misc: stores use the load's format (all four macros); delays count cycles.
    TTI_SFPCONFIG(0x0f0, 8, 1);
}

// One k: X_g = row k of B into LREG[11 + g] for the four column groups, R =
// column k of A into LREG4, then the eight rotations of R. Everything is a
// compile-time constant, so the RISC-V issues prebuilt instructions.
template <std::uint32_t DST_AT, std::uint32_t DST_B, std::uint32_t K>
inline void _matmul_fp32_k_() {
    constexpr auto FP32 = InstrModLoadStore::FP32;
    constexpr std::uint32_t at = DST_AT * MATMUL_FP32_TILE_ROWS;
    constexpr std::uint32_t b = DST_B * MATMUL_FP32_TILE_ROWS;
    constexpr std::uint32_t sub_row = K & 3;
    // Address of the slot of tile row K in column group g.
    constexpr std::uint32_t row = 32 * (K >> 4) + (K & 12);
    constexpr std::uint32_t group[4] = {0, 2, 16, 18};
    // The previous k's replays left the Dst address counter at acc(0, 8).
    math::clear_dst_reg_addr();

    // X_g; SFPTRANSP clobbers LREG0..7, so before R.
#define MATMUL_FP32_X(g)                                                   \
    TTI_SFPLOAD(p_sfpu::LREG0, FP32, ADDR_MOD_7, b + row + group[g]);      \
    if constexpr (sub_row != 0) {                                          \
        TTI_SFPTRANSP(0, 0, 0, 0);                                         \
        TTI_SFPMOV(0, sub_row, p_sfpu::LREG0, 0);                          \
    }                                                                      \
    TTI_SFPCONFIG(0, p_sfpu::LREG11 + (g), 0);
    MATMUL_FP32_X(0)
    MATMUL_FP32_X(1)
    MATMUL_FP32_X(2)
    MATMUL_FP32_X(3)
#undef MATMUL_FP32_X

    // R: sub-row h of LREG[K % 4] after the transpose is group h of row K of A^T.
    TTI_SFPLOAD(p_sfpu::LREG0, FP32, ADDR_MOD_7, at + row + group[0]);
    TTI_SFPLOAD(p_sfpu::LREG1, FP32, ADDR_MOD_7, at + row + group[1]);
    TTI_SFPLOAD(p_sfpu::LREG2, FP32, ADDR_MOD_7, at + row + group[2]);
    TTI_SFPLOAD(p_sfpu::LREG3, FP32, ADDR_MOD_7, at + row + group[3]);
    TTI_SFPTRANSP(0, 0, 0, 0);
    TTI_SFPMOV(0, sub_row, p_sfpu::LREG4, 0);

    for (std::uint32_t s = 0; s < 8; ++s) {
        lltt::replay(MATMUL_FP32_REPLAY_START, MATMUL_FP32_REPLAY_LEN);
    }
    // Let the last scheduled MAD and stores finish before LREG0..7 change.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

template <std::uint32_t DST_AT, std::uint32_t DST_B, std::uint32_t... K>
inline void _matmul_fp32_all_k_(std::integer_sequence<std::uint32_t, K...>) {
    (_matmul_fp32_k_<DST_AT, DST_B, K>(), ...);
}

// acc (Dst tile DST_ACC, skewed) (+)= A * B, with A^T in Dst tile DST_AT and
// B in DST_B. The tile indices are template parameters so that every address
// is a compile-time constant; the run-time arguments repeat them.
template <bool ACCUMULATE, std::uint32_t DST_AT, std::uint32_t DST_B, std::uint32_t DST_ACC>
inline void _calculate_matmul_fp32_(std::uint32_t, std::uint32_t, std::uint32_t) {
    constexpr auto FP32 = InstrModLoadStore::FP32;
    constexpr std::uint32_t acc = DST_ACC * MATMUL_FP32_TILE_ROWS;
    math::clear_dst_reg_addr();

    if constexpr (!ACCUMULATE) {
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        for (std::uint32_t n = 0; n < 32; ++n) {
            TT_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, acc + 2 * n);
        }
    }

    // One rotation s: acc(g, s) += X_g * R for the four groups (load macros,
    // see _init_matmul_fp32_), then rotate R. The last load moves the Dst
    // address counter on to acc(g, s + 1).
    lltt::record(MATMUL_FP32_REPLAY_START, MATMUL_FP32_REPLAY_LEN);
    TTI_SFPLOADMACRO((0 << 2) | p_sfpu::LREG0, FP32, ADDR_MOD_7, acc + _matmul_fp32_acc_(0, 0));
    TTI_SFPLOADMACRO((1 << 2) | p_sfpu::LREG1, FP32, ADDR_MOD_7, acc + _matmul_fp32_acc_(1, 0));
    TTI_SFPLOADMACRO((2 << 2) | p_sfpu::LREG2, FP32, ADDR_MOD_7, acc + _matmul_fp32_acc_(2, 0));
    TTI_SFPLOADMACRO((3 << 2) | p_sfpu::LREG3, FP32, ADDR_MOD_6, acc + _matmul_fp32_acc_(3, 0));
    // R is read by the last MAD in this cycle, and written two cycles later.
    TTI_SFPSHFT2(0, p_sfpu::LREG4, p_sfpu::LREG4, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);

    _matmul_fp32_all_k_<DST_AT, DST_B>(std::make_integer_sequence<std::uint32_t, 32>{});
    math::clear_dst_reg_addr();

    // Give SFPI-compiled code its -1.0 back.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0xbf80);
    TTI_SFPCONFIG(0, p_sfpu::LREG11, 0);
}

// LREG0 = all ones in lane column `column` of each 8-lane group, zero elsewhere:
// ((((2 * lane - 2 * column) << 28) >> 28) - 1) >> 31 (arithmetic), where LTILEID holds 2 * lane.
inline void _matmul_fp32_lane_column_mask_(std::uint32_t column) {
    constexpr std::uint32_t SHFT_IMM = 1;        // SFPSHFT_MOD1_ARG_IMM
    constexpr std::uint32_t SHFT_IMM_ARITH = 3;  // SFPSHFT_MOD1_ARG_IMM | SFPSHFT_MOD1_ARITHMETIC
    constexpr std::uint32_t IADD_IMM = sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE;
    TT_SFPIADD((0u - 2 * column) & 0xfff, p_sfpu::LTILEID, p_sfpu::LREG0, IADD_IMM);
    TTI_SFPSHFT(28, 0, p_sfpu::LREG0, SHFT_IMM);
    TTI_SFPSHFT((0u - 28) & 0xfff, 0, p_sfpu::LREG0, SHFT_IMM);
    TTI_SFPIADD(0xfff /* -1 */, p_sfpu::LREG0, p_sfpu::LREG0, IADD_IMM);
    TTI_SFPSHFT((0u - 31) & 0xfff, 0, p_sfpu::LREG0, SHFT_IMM_ARITH);
}

// The natural C tile (Dst tile dst_out) from the skewed accumulators (dst_acc),
// using dst_scratch for lane masks. All moves are bitwise, so they are exact.
//
// acc(g, s) holds C[col(h, j - s)][col(g, j)] in lane (h, j). For t = 0..7,
// G_t gathers lane j from acc(g, (j - t) mod 8), so its sub-row h is row
// col(h, t) of C: rows 2t, 2t + 1, 16 + 2t, 17 + 2t. Two SFPTRANSPs and a
// sub-row merge turn G_{2v} and G_{2v+1} into the slots of rows 4v..4v+3 and
// 16+4v..19+4v.
inline void _matmul_fp32_finish_(std::uint32_t dst_acc, std::uint32_t dst_out, std::uint32_t dst_scratch) {
    constexpr auto FP32 = InstrModLoadStore::FP32;
    const std::uint32_t acc = dst_acc * MATMUL_FP32_TILE_ROWS;
    const std::uint32_t out = dst_out * MATMUL_FP32_TILE_ROWS;
    const std::uint32_t scratch = dst_scratch * MATMUL_FP32_TILE_ROWS;
    math::clear_dst_reg_addr();

    // Scratch slots 0..7: lane column masks; slot 8: all ones in sub-rows 2 and 3.
    for (std::uint32_t column = 0; column < 8; ++column) {
        _matmul_fp32_lane_column_mask_(column);
        TT_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, scratch + 2 * column);
    }
    TTI_SFPIADD((0u - 32) & 0xfff, p_sfpu::LTILEID, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT((0u - 31) & 0xfff, 0, p_sfpu::LREG0, 3 /* ARG_IMM | ARITHMETIC */);
    TTI_SFPNOT(0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TT_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, scratch + 16);

    // G_t of group g into register `dst`, with LREG2 and LREG3 as scratch.
    auto gather = [&](std::uint32_t g, std::uint32_t t, std::uint32_t dst) {
        TT_SFPMOV(0, p_sfpu::LCONST_0, dst, 0);
        for (std::uint32_t s = 0; s < 8; ++s) {
            TT_SFPLOAD(p_sfpu::LREG2, FP32, ADDR_MOD_7, acc + _matmul_fp32_acc_(g, s));
            TT_SFPLOAD(p_sfpu::LREG3, FP32, ADDR_MOD_7, scratch + 2 * ((s + t) & 7));
            TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG2, 0);
            TT_SFPOR(0, p_sfpu::LREG2, dst, 0);
        }
    };

#pragma GCC unroll 0
    for (std::uint32_t g = 0; g < 4; ++g) {
#pragma GCC unroll 0
        for (std::uint32_t v = 0; v < 4; ++v) {
            gather(g, 2 * v, p_sfpu::LREG0);      // (a0, a1, a2, a3): rows 4v, 4v + 1, 16 + 4v, 17 + 4v
            gather(g, 2 * v + 1, p_sfpu::LREG1);  // (b0, b1, b2, b3): rows 4v + 2, 4v + 3, 18 + 4v, 19 + 4v
            // (a, b, a, b) transposed: LREG i = (a_i, b_i, a_i, b_i).
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
            TTI_SFPTRANSP(0, 0, 0, 0);
            // Again, as (L0, L1, L0, L1) and (L2, L3, L2, L3):
            // LREG0 = (a0, a1, a0, a1), LREG1 = (b0, b1, b0, b1), LREG4 = (a2, a3, a2, a3), LREG5 = (b2, b3, b2, b3).
            TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG4, 0);
            TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG5, 0);
            TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG6, 0);
            TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG7, 0);
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
            TTI_SFPTRANSP(0, 0, 0, 0);
            // Rows 4v..4v+3 = LREG0 with sub-rows 2 and 3 from LREG1; rows 16+4v.. likewise from LREG4, LREG5.
            TT_SFPLOAD(p_sfpu::LREG2, FP32, ADDR_MOD_7, scratch + 16);
            TTI_SFPXOR(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
            TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
            TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPXOR(0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
            TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG5, 0);
            TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG4, 0);
            TT_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, out + _matmul_fp32_slot_(4 * v, _matmul_fp32_group_col_(g)));
            TT_SFPSTORE(
                p_sfpu::LREG4, FP32, ADDR_MOD_7, out + _matmul_fp32_slot_(16 + 4 * v, _matmul_fp32_group_col_(g)));
        }
    }
    math::clear_dst_reg_addr();
}

}  // namespace sfpu
}  // namespace ckernel
