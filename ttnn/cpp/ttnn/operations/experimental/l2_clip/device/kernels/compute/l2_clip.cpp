// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// For each output tile, in FP32 DST: tile 0 gets its rows' (or, for groups
// of 16, its rows' left and right faces') sums of squares, from its group's
// scaled input tiles copied one by one into tile 1 (the scale's row
// broadcast down tile 2); tile 4 keeps the output tile's own scaled input,
// tile 6 the row scale spread across each row, tile 3 the gate; then
// output = v * min(1, threshold * rsqrt(sum + eps^2)) * row_scale * sigmoid(gate).
//
// SFPU lane layout (fp32 DST): vector d (0..7) of a face holds, in lane l
// (0..31), the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2: rows 4B .. 4B + 3 of the tile ("band" B, 0..7) are
// vectors 2 (B % 4) and the next of faces 2 (B / 4) (columns 0..15) and the
// next (columns 16..31), a row an 8-lane subvector of each.
#include <cstdint>
#include <type_traits>
#include <utility>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

namespace {

template <typename F, uint32_t... I>
inline __attribute__((always_inline)) void static_for_impl(F&& f, std::integer_sequence<uint32_t, I...>) {
    (f(std::integral_constant<uint32_t, I>{}), ...);
}

template <uint32_t N, typename F>
inline __attribute__((always_inline)) void static_for(F&& f) {
    static_for_impl(f, std::make_integer_sequence<uint32_t, N>{});
}

}  // namespace

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"

namespace {

using sfpi::dst_reg;
using sfpi::vFloat;
using sfpi::vInt;
using sfpi::vUInt;

constexpr uint32_t vec(uint32_t t, uint32_t f, uint32_t b, uint32_t parity) { return t * 32 + f * 8 + b * 2 + parity; }
// Band B's vector of tile t: left (columns 0..15) or right face, even or odd columns.
constexpr uint32_t band(uint32_t t, uint32_t B, uint32_t right, uint32_t parity) {
    return vec(t, 2 * (B / 4) + right, B % 4, parity);
}

sfpi_inline vInt lane_col2() { return vInt(sfpi::vConstTileId) & 14; }
sfpi_inline vFloat rotate(vFloat v) { return vFloat(sfpi::subvec_shflror1(v)); }

// Every lane of each 8-lane subvector gets the subvector's sum, exactly.
sfpi_inline vFloat sum_lanes8(vFloat v) {
    v += rotate(v);
    vFloat r = rotate(v);
    v += rotate(r);
    r = rotate(v);
    r = rotate(r);
    r = rotate(r);
    v += rotate(r);
    return v;
}

// 1 / sqrt(x) for positive finite x (the SQRT_23-bits algorithm of
// ckernel_sfpu_sqrt.h, with its reciprocal refinement).
sfpi_inline vFloat rsqrt_positive(vFloat x) {
    vInt i = sfpi::as<vInt>(sfpi::as<vUInt>(x) >> 1);
    vFloat y = sfpi::as<vFloat>(vInt(0x5f1110a0) - i);
    vFloat xy = x * y;
    vFloat c = -y * xy;
    y = y * (2.2825186f + c * (2.2533049f + c));
    xy = x * y;
    vFloat one_minus_xyy = 1.0f - y * xy;
    return one_minus_xyy * sfpi::addexp(y, -1) + y;
}

// a into band B.. of tile t's (right, parity) vectors.
template <uint32_t t, uint32_t right, uint32_t parity, uint32_t B>
sfpi_inline void store_bands(vFloat a) {
    dst_reg[band(t, B, right, parity)] = a;
    if constexpr (B + 1 < 8) {
        store_bands<t, right, parity, B + 1>(a);
    }
}

// Row 0 of tile t broadcast down all its rows.
template <uint32_t t>
inline void broadcast_rows() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        constexpr uint32_t right = x / 2, parity = x % 2;
        vFloat a = dst_reg[band(t, 0, right, parity)];
        vFloat b = a;
        vFloat c = a;
        vFloat d = a;
        sfpi::subvec_transp(a, b, c, d);
        store_bands<t, right, parity, 0>(a);
    });
    _llk_math_eltwise_sfpu_done_();
}

// Column 0 of tile t spread across each row: band B's left even vector
// (index band(t, B, 0, 0)) gets each row's value in every lane.
template <uint32_t t>
inline void spread_column0() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<8>([](auto BB) {
        constexpr uint32_t B = decltype(BB)::value;
        vFloat v = dst_reg[band(t, B, 0, 0)];
        v_if(lane_col2() != 0) { v = 0.0f; }
        v_endif;
        dst_reg[band(t, B, 0, 0)] = sum_lanes8(v);
    });
    _llk_math_eltwise_sfpu_done_();
}

// Tile 1 times the broadcast scale of tile 2; kept in tile 4 when `keep`.
template <bool scaled>
inline void scale_and_keep(bool keep) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<32>([&](auto ii) {
        constexpr uint32_t i = decltype(ii)::value;
        vFloat v = dst_reg[32 + i];
        if constexpr (scaled) {
            v = v * vFloat(dst_reg[2 * 32 + i]);
            dst_reg[32 + i] = v;
        }
        if (keep) {
            dst_reg[4 * 32 + i] = v;
        }
    });
    _llk_math_eltwise_sfpu_done_();
}

// Each lane's partial sums of squares of tile 1 into tile 0: band B's at
// vector B (whole rows) or B (left face) and 8 + B (right face).
template <bool half_tile_groups>
inline void add_squares(bool first) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<8>([&](auto BB) {
        constexpr uint32_t B = decltype(BB)::value;
        const vFloat l0 = dst_reg[band(1, B, 0, 0)];
        const vFloat l1 = dst_reg[band(1, B, 0, 1)];
        const vFloat r0 = dst_reg[band(1, B, 1, 0)];
        const vFloat r1 = dst_reg[band(1, B, 1, 1)];
        if constexpr (half_tile_groups) {
            vFloat left = l0 * l0 + l1 * l1;
            vFloat right = r0 * r0 + r1 * r1;
            if (!first) {
                left += vFloat(dst_reg[B]);
                right += vFloat(dst_reg[8 + B]);
            }
            dst_reg[B] = left;
            dst_reg[8 + B] = right;
        } else {
            vFloat sum = l0 * l0 + l1 * l1 + r0 * r0 + r1 * r1;
            if (!first) {
                sum += vFloat(dst_reg[B]);
            }
            dst_reg[B] = sum;
        }
    });
    _llk_math_eltwise_sfpu_done_();
}

// min(1, threshold * rsqrt(sum + eps^2)) of a row's lane partial sums.
sfpi_inline vFloat clip_factor(vFloat partial, uint32_t threshold_bits, uint32_t eps_bits) {
    vFloat total = sum_lanes8(partial);
    vFloat f = sfpi::as<vFloat>(vInt(threshold_bits)) * rsqrt_positive(total + sfpi::as<vFloat>(vInt(eps_bits)));
    v_if(f > 1.0f) { f = 1.0f; }
    v_endif;
    return f;
}

// Tile 4 (the kept scaled input) times its rows' clip factors (with `clip`)
// and row scales (with `row_scaled`).
template <bool half_tile_groups, bool row_scaled, bool clip>
inline void apply_clip(uint32_t threshold_bits, uint32_t eps_bits) {
    if constexpr (!row_scaled && !clip) {
        return;
    }
    _llk_math_eltwise_sfpu_start_(0);
    static_for<8>([&](auto BB) {
        constexpr uint32_t B = decltype(BB)::value;
        vFloat left = 1.0f;
        vFloat right = 1.0f;
        if constexpr (clip) {
            left = clip_factor(dst_reg[B], threshold_bits, eps_bits);
            right = left;
            if constexpr (half_tile_groups) {
                right = clip_factor(dst_reg[8 + B], threshold_bits, eps_bits);
            }
        }
        if constexpr (row_scaled) {
            const vFloat s = dst_reg[band(6, B, 0, 0)];
            left = left * s;
            right = right * s;
        }
        dst_reg[band(4, B, 0, 0)] = vFloat(dst_reg[band(4, B, 0, 0)]) * left;
        dst_reg[band(4, B, 0, 1)] = vFloat(dst_reg[band(4, B, 0, 1)]) * left;
        dst_reg[band(4, B, 1, 0)] = vFloat(dst_reg[band(4, B, 1, 0)]) * right;
        dst_reg[band(4, B, 1, 1)] = vFloat(dst_reg[band(4, B, 1, 1)]) * right;
    });
    _llk_math_eltwise_sfpu_done_();
}

sfpi_inline vFloat recip_positive(vFloat x) {
    vFloat y = sfpi::approx_recip(x);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

// Tile 4 times sigmoid(tile 3), every element, in FP32.
inline void apply_gate() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<32>([](auto ii) {
        constexpr uint32_t i = decltype(ii)::value;
        vFloat g = dst_reg[3 * 32 + i];
        v_if(g < -80.0f) { g = -80.0f; }
        v_endif;
        const vFloat s = recip_positive(1.0f + ckernel::sfpu::_sfpu_exp_fp32_accurate_(-g));
        dst_reg[4 * 32 + i] = vFloat(dst_reg[4 * 32 + i]) * s;
    });
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace
#endif

namespace {

inline void sfpu_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

inline void copy_from(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

}  // namespace

void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_input = get_named_compile_time_arg_val("cb_input");
    constexpr uint32_t cb_scale = get_named_compile_time_arg_val("cb_scale");
    constexpr uint32_t cb_row_scale = get_named_compile_time_arg_val("cb_row_scale");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("Nt");
    constexpr uint32_t Gt = get_named_compile_time_arg_val("Gt");
    constexpr bool half_tile_groups = get_named_compile_time_arg_val("half_tile_groups") != 0;
    constexpr bool has_scale = get_named_compile_time_arg_val("has_scale") != 0;
    constexpr bool has_row_scale = get_named_compile_time_arg_val("has_row_scale") != 0;
    constexpr uint32_t threshold = get_named_compile_time_arg_val("threshold");
    constexpr uint32_t eps_squared = get_named_compile_time_arg_val("eps_squared");
    constexpr uint32_t cb_gate = get_named_compile_time_arg_val("cb_gate");
    constexpr bool has_gate = get_named_compile_time_arg_val("has_gate") != 0;
    constexpr bool clip = get_named_compile_time_arg_val("clip") != 0;

    compute_kernel_hw_startup(cb_input, cb_input, cb_output);
    for (uint32_t unit = start; unit < start + count; ++unit) {
        const uint32_t own = unit % Nt % Gt;
        cb_wait_front(cb_input, Gt);
        if constexpr (has_scale) {
            cb_wait_front(cb_scale, Gt);
        }
        if constexpr (has_row_scale) {
            cb_wait_front(cb_row_scale, 1);
        }
        if constexpr (has_gate) {
            cb_wait_front(cb_gate, 1);
        }
        tile_regs_acquire();
        if constexpr (has_row_scale) {
            copy_from(cb_row_scale, 0, 6);
            sfpu_init();
            MATH((spread_column0<6>()));
        }
        for (uint32_t k = 0; k < Gt; ++k) {
            copy_from(cb_input, k, 1);
            if constexpr (has_scale) {
                copy_from(cb_scale, k, 2);
                sfpu_init();
                MATH((broadcast_rows<2>()));
            } else {
                sfpu_init();
            }
            MATH((scale_and_keep<has_scale>(k == own)));
            if constexpr (clip) {
                MATH((add_squares<half_tile_groups>(k == 0)));
            }
        }
        MATH((apply_clip<half_tile_groups, has_row_scale, clip>(threshold, eps_squared)));
        if constexpr (has_gate) {
            copy_from(cb_gate, 0, 3);
            sfpu_init();
            MATH((apply_gate()));
        }
        tile_regs_commit();

        cb_reserve_back(cb_output, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_output);
        pack_tile(4, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, 1);
        cb_pop_front(cb_input, Gt);
        if constexpr (has_scale) {
            cb_pop_front(cb_scale, Gt);
        }
        if constexpr (has_row_scale) {
            cb_pop_front(cb_row_scale, 1);
        }
        if constexpr (has_gate) {
            cb_pop_front(cb_gate, 1);
        }
    }
}
