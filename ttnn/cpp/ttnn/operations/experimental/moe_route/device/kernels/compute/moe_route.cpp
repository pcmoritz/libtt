// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// A router's selection (moe_route.hpp) for the tokens in rows 0..3, all in
// FP32 on the SFPU.
//
// SFPU lane layout (fp32 DST): vector d (0..7) of a face holds, in lane l
// (0..31), the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2. Band 0's four vectors (positions x: face 0 even, face
// 0 odd, face 1 even, face 1 odd) hold the tokens, token t the 8-lane
// subvector t; in logits tile e, lane p of position x is expert
// 32 e + 16 (x / 2) + 2 p + x % 2.
//
// DST tiles: 0 a logits tile, 1 its bias tile, 2 the weights, 3 the ids,
// 6 every expert's score and 7 its key (score + bias), position j = 4 e + x
// at vector j.
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

constexpr uint32_t vec(uint32_t t, uint32_t f, uint32_t b, uint32_t parity) { return t * 32 + f * 8 + b * 2 + parity; }
constexpr uint32_t pos(uint32_t t, uint32_t x) { return vec(t, x / 2, 0, x % 2); }
constexpr uint32_t score(uint32_t j) { return 6 * 32 + j; }
constexpr uint32_t key(uint32_t j) { return 7 * 32 + j; }
// The column of lane 0 of position j = 4 e + x (lanes add 2 p).
constexpr int column(uint32_t j) { return static_cast<int>(32 * (j / 4) + 16 * (j % 4 / 2) + j % 2); }
constexpr float kNeg = -1e30f;

sfpi_inline vInt lane_col2() { return vInt(sfpi::vConstTileId) & 14; }

template <typename V>
sfpi_inline V rotate(V v) {
    return V(sfpi::subvec_shflror1(v));
}

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

sfpi_inline vFloat vmax(vFloat a, vFloat b) {
    v_if(a < b) { a = b; }
    v_endif;
    return a;
}

sfpi_inline vInt vmin(vInt a, vInt b) {
    v_if(b < a) { a = b; }
    v_endif;
    return a;
}

// Every lane of each subvector: the subvector's maximum (minimum).
sfpi_inline vFloat max_lanes8(vFloat v) {
    v = vmax(v, rotate(v));
    vFloat r = rotate(v);
    v = vmax(v, rotate(r));
    r = rotate(v);
    r = rotate(r);
    r = rotate(r);
    v = vmax(v, rotate(r));
    return v;
}

sfpi_inline vInt min_lanes8(vInt v) {
    v = vmin(v, rotate(v));
    vInt r = rotate(v);
    v = vmin(v, rotate(r));
    r = rotate(v);
    r = rotate(r);
    r = rotate(r);
    v = vmin(v, rotate(r));
    return v;
}

sfpi_inline vFloat recip_positive(vFloat x) {
    vFloat y = sfpi::approx_recip(x);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

// Logits tile e (DST 0) and its bias tile (DST 1, row 0): scores and keys.
template <uint32_t e>
inline void prepare() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        vFloat b = dst_reg[pos(1, x)];
        vFloat b1 = b;
        vFloat b2 = b;
        vFloat b3 = b;
        sfpi::subvec_transp(b, b1, b2, b3);
        vFloat l = dst_reg[pos(0, x)];
        v_if(l < -80.0f) { l = -80.0f; }
        v_endif;
        const vFloat s = recip_positive(1.0f + ckernel::sfpu::_sfpu_exp_fp32_accurate_(-l));
        dst_reg[score(4 * e + x)] = s;
        dst_reg[key(4 * e + x)] = s + b;
    });
    _llk_math_eltwise_sfpu_done_();
}

template <uint32_t J, uint32_t j>
sfpi_inline vFloat max_keys(vFloat m) {
    if constexpr (j == J) {
        return m;
    } else {
        return max_keys<J, j + 1>(vmax(m, dst_reg[key(j)]));
    }
}

// The lowest column whose key is m, in each lane.
template <uint32_t J, uint32_t j>
sfpi_inline vInt first_match(vFloat m, vInt best) {
    if constexpr (j == J) {
        return best;
    } else {
        vFloat k = dst_reg[key(j)];
        v_if(k == m) {
            vInt c = lane_col2() + column(j);
            v_if(c < best) { best = c; }
            v_endif;
        }
        v_endif;
        return first_match<J, j + 1>(m, best);
    }
}

// The selected expert's score (in its lane, 0 elsewhere); its key masked.
template <uint32_t J, uint32_t j>
sfpi_inline vFloat take(vInt idx, vFloat selected) {
    if constexpr (j == J) {
        return selected;
    } else {
        v_if(lane_col2() + column(j) == idx) {
            selected = dst_reg[score(j)];
            dst_reg[key(j)] = kNeg;
        }
        v_endif;
        return take<J, j + 1>(idx, selected);
    }
}

// Selection i: column i of the weights (DST 2) and ids (DST 3) of each
// token, and the running sum of the selected scores (DST 2 vector 16).
template <uint32_t J, uint32_t i>
inline void select_one() {
    _llk_math_eltwise_sfpu_start_(0);
    const vFloat m = max_lanes8(max_keys<J, 1>(dst_reg[key(0)]));
    const vInt idx = min_lanes8(first_match<J, 0>(m, vInt(1 << 20)));
    const vFloat selected = sum_lanes8(take<J, 0>(idx, vFloat(0.0f)));
    if constexpr (i == 0) {
        dst_reg[2 * 32 + 16] = selected;
    } else {
        dst_reg[2 * 32 + 16] = vFloat(dst_reg[2 * 32 + 16]) + selected;
    }
    constexpr int lane2 = static_cast<int>(2 * (i / 2));
    vFloat w = dst_reg[pos(2, i % 2)];
    // Integer loads and stores: an FP32 one would flush the ids' bits as denormals.
    vInt id = dst_reg[pos(3, i % 2)];
    v_if(lane_col2() == lane2) {
        w = selected;
        id = idx;
    }
    v_endif;
    dst_reg[pos(2, i % 2)] = w;
    dst_reg[pos(3, i % 2)] = id;
    _llk_math_eltwise_sfpu_done_();
}

inline void clear_outputs() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<32>([](auto ii) {
        constexpr uint32_t i = decltype(ii)::value;
        dst_reg[2 * 32 + i] = vFloat(0.0f);
        dst_reg[3 * 32 + i] = vInt(0);
    });
    _llk_math_eltwise_sfpu_done_();
}

// The weights normalized and scaled; the running sum's vector cleared.
inline void normalize(uint32_t scale_bits) {
    _llk_math_eltwise_sfpu_start_(0);
    const vFloat inv = recip_positive(dst_reg[2 * 32 + 16]) * sfpi::as<vFloat>(vInt(scale_bits));
    dst_reg[pos(2, 0)] = vFloat(dst_reg[pos(2, 0)]) * inv;
    dst_reg[pos(2, 1)] = vFloat(dst_reg[pos(2, 1)]) * inv;
    dst_reg[2 * 32 + 16] = vFloat(0.0f);
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
    constexpr uint32_t cb_logits = get_named_compile_time_arg_val("cb_logits");
    constexpr uint32_t cb_bias = get_named_compile_time_arg_val("cb_bias");
    constexpr uint32_t cb_weights = get_named_compile_time_arg_val("cb_weights");
    constexpr uint32_t cb_ids = get_named_compile_time_arg_val("cb_ids");
    constexpr uint32_t Et = get_named_compile_time_arg_val("Et");
    constexpr uint32_t K = get_named_compile_time_arg_val("k");
    constexpr uint32_t scale = get_named_compile_time_arg_val("scale");

    compute_kernel_hw_startup(cb_logits, cb_logits, cb_weights);
    cb_wait_front(cb_logits, Et);
    cb_wait_front(cb_bias, Et);
    tile_regs_acquire();
    static_for<Et>([](auto ee) {
        constexpr uint32_t e = decltype(ee)::value;
        copy_from(cb_logits, e, 0);
        copy_from(cb_bias, e, 1);
        sfpu_init();
        MATH((prepare<e>()));
    });
    MATH((clear_outputs()));
    static_for<K>([](auto ii) { MATH((select_one<4 * Et, decltype(ii)::value>())); });
    MATH((normalize(scale)));
    tile_regs_commit();
    cb_pop_front(cb_logits, Et);
    cb_pop_front(cb_bias, Et);
    cb_reserve_back(cb_weights, 1);
    cb_reserve_back(cb_ids, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_weights);
    pack_tile(2, cb_weights);
    pack_reconfig_data_format(cb_ids);
    pack_tile(3, cb_ids);
    tile_regs_release();
    cb_push_back(cb_weights, 1);
    cb_push_back(cb_ids, 1);
}
