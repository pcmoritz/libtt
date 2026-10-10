// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel {
namespace sfpu {

// ============================================================================
// FP32 matrix multiply on the vector unit: C += A * B for 32x32 tiles in Dst
// ============================================================================
//
// The matrix unit reads FP32 operands as TF32, so its products carry about 10
// bits of mantissa. The vector unit's SFPMAD is an FP32 multiply-add with a
// single rounding, which gives FP32 results at much lower throughput.
//
// Dst layout (Dst32b, as addressed by SFPLOAD / SFPSTORE). A tile is 64
// address rows: faces 0..3 (rows 0-15 x cols 0-15, rows 0-15 x cols 16-31,
// rows 16-31 x cols 0-15, rows 16-31 x cols 16-31) of 16 rows each. One
// SFPLOAD at address `addr` reads rows (addr & ~3) .. +3 of the face and its
// even columns, or its odd columns if addr & 2, into the 32 lanes:
//
//     lane 8 * i + j  <-  row (addr & ~3) + i, face column 2 * j + ((addr >> 1) & 1)
//
// so a register holds a 4x8 block of the tile (a "slot"). The 8 lanes j of a
// sub-row i are tile columns 16 * h + 2 * j + p for half h and parity p; the
// four (h, p) combinations are the column groups g = 2 * h + p of a row.
//
// For each k, C[r][c] += A[r][k] * B[k][c] on all 32 slots of C needs:
//
//   Y_g: row k of B for column group g, the same in all four sub-rows. Load
//        the slot of B holding row k; SFPTRANSP moves sub-row s of LREG0 to
//        sub-row 0 of LREG s, which SFPCONFIG then broadcasts to all four
//        sub-rows of LREG[11 + g]. The four registers serve all row bands.
//   X:   column k of A for the four rows of a band, the same in all 8 lanes
//        of a sub-row. Load the slot of A holding column k, keep only its
//        lane column with a bitwise AND, and spread it with three
//        rotate-and-OR steps within each 8-lane sub-vector. Bitwise AND and
//        OR (unlike multiplying by a 0/1 mask) keep infinities and NaNs exact.
//
// then C_slot = X * Y_g + C_slot with SFPMAD for the 4 column groups.
//
// LREG11 holds -1.0 for SFPI-compiled code; it is restored before returning.

constexpr std::uint32_t MATMUL_FP32_TILE_ROWS = 64;  // Dst address rows per 32x32 tile

// Dst address (relative to the tile) of the slot holding tile row `row` and tile column `col`.
inline std::uint32_t _matmul_fp32_slot_(std::uint32_t row, std::uint32_t col) {
    return 16 * (2 * (row >> 4) + (col >> 4)) + (row & 12) + ((col & 1) << 1);
}

// Representative tile column of column group g.
inline constexpr std::uint32_t _matmul_fp32_group_col_(std::uint32_t g) { return 16 * (g >> 1) + (g & 1); }

constexpr std::uint32_t MATMUL_FP32_X = p_sfpu::LREG4;     // broadcast column of A
constexpr std::uint32_t MATMUL_FP32_TMP = p_sfpu::LREG5;   // rotation scratch
constexpr std::uint32_t MATMUL_FP32_MASK = p_sfpu::LREG7;  // all ones in the lane column of k, else zero

inline void _init_matmul_fp32_() {
    _init_sfpu_config_reg();
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}

// Broadcast row k of the B tile at Dst address `b` into LREG[11 + g] for column groups g = 0..3.
inline void _matmul_fp32_broadcast_b_row_(std::uint32_t b, std::uint32_t k) {
    constexpr auto FP32 = InstrModLoadStore::FP32;
    const std::uint32_t sub_row = k & 3;
    for (std::uint32_t g = 0; g < 4; ++g) {
        TT_SFPLOAD(p_sfpu::LREG0, FP32, ADDR_MOD_7, b + _matmul_fp32_slot_(k, _matmul_fp32_group_col_(g)));
        if (sub_row != 0) {
            TTI_SFPTRANSP(0, 0, 0, 0);
            TT_SFPMOV(0, sub_row, p_sfpu::LREG0, 0);
        }
        TT_SFPCONFIG(0, p_sfpu::LREG11 + g, 0);
    }
}

// MATMUL_FP32_MASK = all ones in lanes of 8-lane column (k % 16) / 2, zero elsewhere:
// ((((2 * lane - 2 * column) << 28) >> 28) - 1) >> 31 (arithmetic), where LTILEID holds 2 * lane.
inline void _matmul_fp32_build_mask_(std::uint32_t k) {
    constexpr std::uint32_t SHFT_IMM = 1;         // SFPSHFT_MOD1_ARG_IMM
    constexpr std::uint32_t SHFT_IMM_ARITH = 3;   // SFPSHFT_MOD1_ARG_IMM | SFPSHFT_MOD1_ARITHMETIC
    constexpr std::uint32_t IADD_IMM = sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE;
    const std::uint32_t column = (k & 15) >> 1;
    TT_SFPIADD((0u - 2 * column) & 0xfff, p_sfpu::LTILEID, MATMUL_FP32_MASK, IADD_IMM);
    TTI_SFPSHFT(28, 0, MATMUL_FP32_MASK, SHFT_IMM);
    TTI_SFPSHFT((0u - 28) & 0xfff, 0, MATMUL_FP32_MASK, SHFT_IMM);
    TTI_SFPIADD(0xfff /* -1 */, MATMUL_FP32_MASK, MATMUL_FP32_MASK, IADD_IMM);
    TTI_SFPSHFT((0u - 31) & 0xfff, 0, MATMUL_FP32_MASK, SHFT_IMM_ARITH);
}

// MATMUL_FP32_X = A[rows of the slot at `a_slot`][k], spread over the 8 lanes of each sub-row.
inline void _matmul_fp32_broadcast_a_col_(std::uint32_t a_slot) {
    constexpr auto FP32 = InstrModLoadStore::FP32;
    constexpr std::uint32_t ROR1 = sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1;
    TT_SFPLOAD(MATMUL_FP32_X, FP32, ADDR_MOD_7, a_slot);
    TTI_SFPAND(0, MATMUL_FP32_MASK, MATMUL_FP32_X, 0);
    // 1 -> 2 lanes
    TTI_SFPSHFT2(0, MATMUL_FP32_X, MATMUL_FP32_TMP, ROR1);
    TTI_SFPOR(0, MATMUL_FP32_TMP, MATMUL_FP32_X, 0);
    // 2 -> 4 lanes
    TTI_SFPSHFT2(0, MATMUL_FP32_X, MATMUL_FP32_TMP, ROR1);
    TTI_SFPSHFT2(0, MATMUL_FP32_TMP, MATMUL_FP32_TMP, ROR1);
    TTI_SFPOR(0, MATMUL_FP32_TMP, MATMUL_FP32_X, 0);
    // 4 -> 8 lanes
    TTI_SFPSHFT2(0, MATMUL_FP32_X, MATMUL_FP32_TMP, ROR1);
    TTI_SFPSHFT2(0, MATMUL_FP32_TMP, MATMUL_FP32_TMP, ROR1);
    TTI_SFPSHFT2(0, MATMUL_FP32_TMP, MATMUL_FP32_TMP, ROR1);
    TTI_SFPSHFT2(0, MATMUL_FP32_TMP, MATMUL_FP32_TMP, ROR1);
    TTI_SFPOR(0, MATMUL_FP32_TMP, MATMUL_FP32_X, 0);
}

// C (Dst tile dst_c) = A (dst_a) * B (dst_b), plus the previous C if ACCUMULATE. FP32 Dst only.
template <bool ACCUMULATE>
inline void _calculate_matmul_fp32_(std::uint32_t dst_a, std::uint32_t dst_b, std::uint32_t dst_c) {
    constexpr auto FP32 = InstrModLoadStore::FP32;
    const std::uint32_t a = dst_a * MATMUL_FP32_TILE_ROWS;
    const std::uint32_t b = dst_b * MATMUL_FP32_TILE_ROWS;
    const std::uint32_t c = dst_c * MATMUL_FP32_TILE_ROWS;

#pragma GCC unroll 0
    for (std::uint32_t k = 0; k < 32; ++k) {
        // SFPTRANSP clobbers LREG0..7, so broadcast B before building anything else.
        _matmul_fp32_broadcast_b_row_(b, k);
        _matmul_fp32_build_mask_(k);
        const bool first = !ACCUMULATE && k == 0;
#pragma GCC unroll 0
        for (std::uint32_t row = 0; row < 32; row += 4) {
            _matmul_fp32_broadcast_a_col_(a + _matmul_fp32_slot_(row, k));
            const std::uint32_t c0 = c + _matmul_fp32_slot_(row, _matmul_fp32_group_col_(0));
            const std::uint32_t c1 = c + _matmul_fp32_slot_(row, _matmul_fp32_group_col_(1));
            const std::uint32_t c2 = c + _matmul_fp32_slot_(row, _matmul_fp32_group_col_(2));
            const std::uint32_t c3 = c + _matmul_fp32_slot_(row, _matmul_fp32_group_col_(3));
            if (first) {
                TTI_SFPMUL(MATMUL_FP32_X, p_sfpu::LREG11, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
                TTI_SFPMUL(MATMUL_FP32_X, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
                TTI_SFPMUL(MATMUL_FP32_X, p_sfpu::LREG13, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
                TTI_SFPMUL(MATMUL_FP32_X, p_sfpu::LREG14, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
            } else {
                TT_SFPLOAD(p_sfpu::LREG0, FP32, ADDR_MOD_7, c0);
                TT_SFPLOAD(p_sfpu::LREG1, FP32, ADDR_MOD_7, c1);
                TT_SFPLOAD(p_sfpu::LREG2, FP32, ADDR_MOD_7, c2);
                TT_SFPLOAD(p_sfpu::LREG3, FP32, ADDR_MOD_7, c3);
                TTI_SFPMAD(MATMUL_FP32_X, p_sfpu::LREG11, p_sfpu::LREG0, p_sfpu::LREG0, 0);
                TTI_SFPMAD(MATMUL_FP32_X, p_sfpu::LREG12, p_sfpu::LREG1, p_sfpu::LREG1, 0);
                TTI_SFPMAD(MATMUL_FP32_X, p_sfpu::LREG13, p_sfpu::LREG2, p_sfpu::LREG2, 0);
                TTI_SFPMAD(MATMUL_FP32_X, p_sfpu::LREG14, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            }
            TT_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, c0);
            TT_SFPSTORE(p_sfpu::LREG1, FP32, ADDR_MOD_7, c1);
            TT_SFPSTORE(p_sfpu::LREG2, FP32, ADDR_MOD_7, c2);
            TT_SFPSTORE(p_sfpu::LREG3, FP32, ADDR_MOD_7, c3);
        }
    }

    // Give SFPI-compiled code its -1.0 back.
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0xbf80);
    TTI_SFPCONFIG(0, p_sfpu::LREG11, 0);
}

}  // namespace sfpu
}  // namespace ckernel
