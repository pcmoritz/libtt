// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// For each output tile, in FP32 DST: tile 0 gets its rows' rstd, from its
// group's input tiles copied one by one into tile 1; then tile 1 holds the
// input tile, tile 2 the weight broadcast down the rows and tile 3 the gate,
// and output = input * rstd * weight * silu(gate).
//
// SFPU lane layout (fp32 DST): vector d (0..7) of a face holds, in lane l
// (0..31), the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2: a band of four rows is the even and odd vectors of a
// left and a right face, and a row is one 8-lane subvector of each.
#include <cstdint>
#include <type_traits>
#include <utility>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "sfpi.h"

namespace {

template <typename F, uint32_t... I>
sfpi_inline void static_for_impl(F&& f, std::integer_sequence<uint32_t, I...>) {
    (f(std::integral_constant<uint32_t, I>{}), ...);
}

// f(0) .. f(N - 1), with compile-time arguments.
template <uint32_t N, typename F>
sfpi_inline void static_for(F&& f) {
    static_for_impl(f, std::make_integer_sequence<uint32_t, N>{});
}

// SFPU vector `parity` (even or odd columns) of band b (rows 4b .. 4b + 3) of
// face f of DST tile t.
constexpr uint32_t vec(uint32_t t, uint32_t f, uint32_t b, uint32_t parity) { return t * 32 + f * 8 + b * 2 + parity; }

// Adds the squares of band b of tile x to the band's partial row sums, the
// even vector of its left face in tile acc (each lane sums its columns).
template <uint32_t x, uint32_t acc, uint32_t b>
sfpi_inline void add_band_squares(bool first) {
    constexpr uint32_t f = b < 4 ? 0 : 2;
    const sfpi::vFloat v0 = sfpi::dst_reg[vec(x, f, b % 4, 0)];
    const sfpi::vFloat v1 = sfpi::dst_reg[vec(x, f, b % 4, 1)];
    const sfpi::vFloat v2 = sfpi::dst_reg[vec(x, f + 1, b % 4, 0)];
    const sfpi::vFloat v3 = sfpi::dst_reg[vec(x, f + 1, b % 4, 1)];
    sfpi::vFloat sum = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    if (!first) {
        sum = sum + sfpi::vFloat(sfpi::dst_reg[vec(acc, f, b % 4, 0)]);
    }
    sfpi::dst_reg[vec(acc, f, b % 4, 0)] = sum;
}

template <uint32_t x, uint32_t acc>
inline void add_squares(bool first) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<8>([&](auto b) { add_band_squares<x, acc, decltype(b)::value>(first); });
    _llk_math_eltwise_sfpu_done_();
}

sfpi_inline sfpi::vFloat rotate(sfpi::vFloat v) { return sfpi::vFloat(sfpi::subvec_shflror1(v)); }

// Band b of tile acc: each row's sum over its eight lanes, times inv_width
// plus eps, in every element of the row.
template <uint32_t acc, uint32_t b>
sfpi_inline void finish_band(uint32_t inv_width_bits, uint32_t eps_bits) {
    constexpr uint32_t f = b < 4 ? 0 : 2;
    sfpi::vFloat sum = sfpi::dst_reg[vec(acc, f, b % 4, 0)];
    sum = sum + rotate(sum);
    sum = sum + rotate(rotate(sum));
    sum = sum + rotate(rotate(rotate(rotate(sum))));
    const sfpi::vFloat mean = sum * sfpi::as<sfpi::vFloat>(sfpi::vInt(inv_width_bits)) +
                              sfpi::as<sfpi::vFloat>(sfpi::vInt(eps_bits));
    sfpi::dst_reg[vec(acc, f, b % 4, 0)] = mean;
    sfpi::dst_reg[vec(acc, f, b % 4, 1)] = mean;
    sfpi::dst_reg[vec(acc, f + 1, b % 4, 0)] = mean;
    sfpi::dst_reg[vec(acc, f + 1, b % 4, 1)] = mean;
}

template <uint32_t acc>
inline void finish_mean(uint32_t inv_width_bits, uint32_t eps_bits) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<8>([&](auto b) { finish_band<acc, decltype(b)::value>(inv_width_bits, eps_bits); });
    _llk_math_eltwise_sfpu_done_();
}

// Row 0 of face f's `parity` vector of tile t (a subvector transpose of four
// copies puts it in every row of the band) in every band of faces f and
// f + 2.
template <uint32_t t, uint32_t f, uint32_t parity>
sfpi_inline void broadcast_row0() {
    sfpi::vFloat a = sfpi::dst_reg[vec(t, f, 0, parity)];
    sfpi::vFloat b = a;
    sfpi::vFloat c = a;
    sfpi::vFloat d = a;
    sfpi::subvec_transp(a, b, c, d);
    // (SFPU vectors cannot be spilled, so no lambda captures `a`.)
    sfpi::dst_reg[vec(t, f, 0, parity)] = a;
    sfpi::dst_reg[vec(t, f, 1, parity)] = a;
    sfpi::dst_reg[vec(t, f, 2, parity)] = a;
    sfpi::dst_reg[vec(t, f, 3, parity)] = a;
    sfpi::dst_reg[vec(t, f + 2, 0, parity)] = a;
    sfpi::dst_reg[vec(t, f + 2, 1, parity)] = a;
    sfpi::dst_reg[vec(t, f + 2, 2, parity)] = a;
    sfpi::dst_reg[vec(t, f + 2, 3, parity)] = a;
}

// Row 0 of tile t in every row.
template <uint32_t t>
inline void broadcast_rows() {
    _llk_math_eltwise_sfpu_start_(0);
    broadcast_row0<t, 0, 0>();
    broadcast_row0<t, 0, 1>();
    broadcast_row0<t, 1, 0>();
    broadcast_row0<t, 1, 1>();
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace
#endif

namespace {
inline void sfpu_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }
}  // namespace

void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_input = get_named_compile_time_arg_val("cb_input");
    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("Nt");
    constexpr uint32_t Gt = get_named_compile_time_arg_val("Gt");
    constexpr uint32_t inv_width = get_named_compile_time_arg_val("inv_width");
    constexpr uint32_t eps = get_named_compile_time_arg_val("eps");
    constexpr uint32_t dst_rstd = 0, dst_x = 1, dst_weight = 2, dst_gate = 3;

    compute_kernel_hw_startup(cb_input, cb_input, cb_output);
    for (uint32_t unit = start; unit < start + count; ++unit) {
        const uint32_t column = unit % Nt % Gt;
        cb_wait_front(cb_input, Gt);
        cb_wait_front(cb_gate, 1);
        cb_wait_front(cb_weight, 1);
        tile_regs_acquire();

        // rstd = rsqrt(mean(x^2) + eps) of each row, in every element.
        reconfig_data_format_srca(cb_input);
        copy_init(cb_input);
        sfpu_init();
        for (uint32_t k = 0; k < Gt; ++k) {
            copy_tile(cb_input, k, dst_x);
            MATH((add_squares<dst_x, dst_rstd>(k == 0)));
        }
        MATH((finish_mean<dst_rstd>(inv_width, eps)));
        rsqrt_tile_init();
        rsqrt_tile(dst_rstd);

        // The operands of the output tile.
        copy_init(cb_input);
        copy_tile(cb_input, column, dst_x);
        reconfig_data_format_srca(cb_weight);
        copy_init(cb_weight);
        copy_tile(cb_weight, 0, dst_weight);
        reconfig_data_format_srca(cb_gate);
        copy_init(cb_gate);
        copy_tile(cb_gate, 0, dst_gate);
        sfpu_init();
        MATH((broadcast_rows<dst_weight>()));

        // ((x * rstd) * weight) * silu(gate), in PyTorch's order.
        silu_tile_init();
        silu_tile(dst_gate);
        mul_binary_tile_init();
        mul_binary_tile(dst_x, dst_rstd, dst_x);
        mul_binary_tile(dst_x, dst_weight, dst_x);
        mul_binary_tile(dst_x, dst_gate, dst_x);
        tile_regs_commit();

        cb_reserve_back(cb_output, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_output);
        pack_tile(dst_x, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, 1);
        cb_pop_front(cb_input, Gt);
        cb_pop_front(cb_gate, 1);
        cb_pop_front(cb_weight, 1);
    }
}
