// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Block attention residuals for up to four tokens (attn_res.hpp), on this
// core's `count` (1 or 2) tile columns, all in FP32 on the SFPU.
//
// SFPU lane layout (fp32 DST): vector d (0..7) of a face holds, in lane l
// (0..31), the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2. Tokens are rows 0..3, band 0: in a band-0 vector,
// token t is the 8-lane subvector t, and lane p of it holds columns 2p
// (even vector) or 2p + 1 (odd vector) of the face. Band 0 of a tile is
// four vectors, "positions" x: face 0 even, face 0 odd, face 1 even, face 1
// odd.
//
// Exchange rows carry one value per column and token. The first exchange's
// columns: 8k + j for query k's product with source j (k < 3), 24 + j for
// source j's sum of squares; so in a row's vectors, lanes p < 4 ("group A")
// and p >= 4 ("group B") hold four sources each (j = 2 (p % 4) + parity):
// group A of face 0 query 0, group B of face 0 query 1, group A of face 1
// query 2, group B of face 1 the sums of squares. The second exchange's
// columns are the queries' sums of squares of their mixtures, column q.
//
// DST tiles (all eight, FP32, full sync):
//   phase 1: 0 partial sums, 1 a source tile, 2 the addend tile, 3 and 7 the
//            query tiles and their rows broadcast down the band, 4-5 the
//            sources' band-0 vectors kept for phase 2, 6 the exchange row.
//   phase 2: 0 a gather tile, 3 each query and source's weight in
//            every lane of its token's subvector, 4-5 the kept sources, 6
//            the mixtures, 7 working vectors and the second exchange row.
//   phase 3: 0 a gather tile, 3 each query's rstd, 4 a weight tile, 6
//            the mixtures.
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

// f(0) .. f(N - 1), with compile-time arguments.
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
// Band-0 position x of tile t.
constexpr uint32_t pos(uint32_t t, uint32_t x) { return vec(t, x / 2, 0, x % 2); }

// Phase 1.
constexpr uint32_t acc(uint32_t j, uint32_t k) { return j * 4 + k; }
constexpr uint32_t query_tile(uint32_t c) { return c == 0 ? 3 : 7; }
constexpr uint32_t query_row(uint32_t c, uint32_t k, uint32_t x) { return query_tile(c) * 32 + 16 + 4 * k + x; }
template <uint32_t CNT>
constexpr uint32_t kept(uint32_t j, uint32_t c, uint32_t x) {
    return 4 * 32 + (j * CNT + c) * 4 + x;
}
// Phase 2: slot s of a gather tile in DST tile 0.
constexpr uint32_t gathered(uint32_t s, uint32_t x) { return vec(0, 2 * (s / 4) + x / 2, s % 4, x % 2); }
constexpr uint32_t stat(uint32_t x) { return 7 * 32 + 16 + x; }
constexpr uint32_t prob(uint32_t x) { return 7 * 32 + 20 + x; }
constexpr uint32_t rfull(uint32_t parity) { return 7 * 32 + 24 + parity; }
constexpr uint32_t mix_squares(uint32_t q) { return 7 * 32 + 28 + q; }
constexpr uint32_t weight_of(uint32_t q, uint32_t j) { return 3 * 32 + q * 8 + j; }
constexpr uint32_t stage(uint32_t u, uint32_t x) { return vec(6, 2 * (u / 4) + x / 2, u % 4, x % 2); }
// Phase 3.
constexpr uint32_t rstd_of(uint32_t q) { return 3 * 32 + q; }
constexpr uint32_t weight_row(uint32_t q, uint32_t x) { return 4 * 32 + 16 + 4 * q + x; }

constexpr float kNeg = -1e30f;

// 2 * (lane % 8): twice the lane's position in its subvector.
sfpi_inline vInt lane_col2() { return vInt(sfpi::vConstTileId) & 14; }
// 16 * (lane / 8): sixteen times the lane's subvector (token).
sfpi_inline vInt lane_row16() { return vInt(sfpi::vConstTileId) & 48; }

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

sfpi_inline vFloat vmax(vFloat a, vFloat b) {
    v_if(a < b) { a = b; }
    v_endif;
    return a;
}

// Every lane of each 8-lane subvector gets the subvector's maximum.
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

// Lanes p < 4 of each subvector get the sum (max) over lanes 0..3, lanes
// p >= 4 over lanes 4..7.
sfpi_inline vFloat sum_groups(vFloat v) {
    vFloat a = v;
    v_if(lane_col2() >= 8) { a = 0.0f; }
    v_endif;
    a = sum_lanes8(a);
    vFloat b = v;
    v_if(lane_col2() < 8) { b = 0.0f; }
    v_endif;
    b = sum_lanes8(b);
    v_if(lane_col2() >= 8) { a = b; }
    v_endif;
    return a;
}

sfpi_inline vFloat max_groups(vFloat v) {
    vFloat a = v;
    v_if(lane_col2() >= 8) { a = kNeg; }
    v_endif;
    a = max_lanes8(a);
    vFloat b = v;
    v_if(lane_col2() < 8) { b = kNeg; }
    v_endif;
    b = max_lanes8(b);
    v_if(lane_col2() >= 8) { a = b; }
    v_endif;
    return a;
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

// 1 / x for positive finite x: an approximation and two Newton steps.
sfpi_inline vFloat recip_positive(vFloat x) {
    vFloat y = sfpi::approx_recip(x);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

// ---------------------------------------------------------------- phase 1

// Query rows 0..3 of tile c's band 0, each broadcast down the band.
template <uint32_t c>
inline void broadcast_queries() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        vFloat a = dst_reg[pos(query_tile(c), x)];
        vFloat b = a;
        vFloat cc = a;
        vFloat d = a;
        sfpi::subvec_transp(a, b, cc, d);
        dst_reg[query_row(c, 0, x)] = a;
        dst_reg[query_row(c, 1, x)] = b;
        dst_reg[query_row(c, 2, x)] = cc;
        dst_reg[query_row(c, 3, x)] = d;
    });
    _llk_math_eltwise_sfpu_done_();
}

// Source j's tile c (in DST tile 1, plus the addend in tile 2): kept, and
// added to the partial sums of squares and query products of each lane.
template <uint32_t Q, uint32_t CNT, uint32_t j, uint32_t c, bool add_addend>
inline void accumulate() {
    _llk_math_eltwise_sfpu_start_(0);
    if constexpr (add_addend) {
        static_for<4>([](auto xx) {
            constexpr uint32_t x = decltype(xx)::value;
            dst_reg[pos(1, x)] = vFloat(dst_reg[pos(1, x)]) + vFloat(dst_reg[pos(2, x)]);
        });
    }
    {
        vFloat v0 = dst_reg[pos(1, 0)];
        vFloat v1 = dst_reg[pos(1, 1)];
        vFloat v2 = dst_reg[pos(1, 2)];
        vFloat v3 = dst_reg[pos(1, 3)];
        dst_reg[kept<CNT>(j, c, 0)] = v0;
        dst_reg[kept<CNT>(j, c, 1)] = v1;
        dst_reg[kept<CNT>(j, c, 2)] = v2;
        dst_reg[kept<CNT>(j, c, 3)] = v3;
        vFloat squares = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        if constexpr (c > 0) {
            squares += vFloat(dst_reg[acc(j, 3)]);
        }
        dst_reg[acc(j, 3)] = squares;
    }
    static_for<Q>([](auto kk) {
        constexpr uint32_t k = decltype(kk)::value;
        vFloat dot = vFloat(dst_reg[pos(1, 0)]) * vFloat(dst_reg[query_row(c, k, 0)]);
        dot += vFloat(dst_reg[pos(1, 1)]) * vFloat(dst_reg[query_row(c, k, 1)]);
        dot += vFloat(dst_reg[pos(1, 2)]) * vFloat(dst_reg[query_row(c, k, 2)]);
        dot += vFloat(dst_reg[pos(1, 3)]) * vFloat(dst_reg[query_row(c, k, 3)]);
        if constexpr (c > 0) {
            dot += vFloat(dst_reg[acc(j, k)]);
        }
        dst_reg[acc(j, k)] = dot;
    });
    _llk_math_eltwise_sfpu_done_();
}

// Rows past the tokens zeroed in the exchange row of tile t.
template <uint32_t t, uint32_t T>
sfpi_inline void zero_padding_rows() {
    if constexpr (T < 4) {
        static_for<4>([](auto xx) {
            constexpr uint32_t x = decltype(xx)::value;
            vFloat out = dst_reg[pos(t, x)];
            v_if(lane_row16() >= static_cast<int>(16 * T)) { out = 0.0f; }
            v_endif;
            dst_reg[pos(t, x)] = out;
        });
    }
}

// The first exchange row (tile 6) from the partial sums.
template <uint32_t N, uint32_t Q, uint32_t T>
inline void make_row1() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) { dst_reg[pos(6, decltype(xx)::value)] = vFloat(0.0f); });
    static_for<N>([](auto jj) {
        static_for<Q + 1>([](auto kk) {
            constexpr uint32_t j = decltype(jj)::value;
            constexpr uint32_t k = decltype(kk)::value == Q ? 3 : decltype(kk)::value;
            constexpr uint32_t column = 8 * k + j;
            constexpr uint32_t x = 2 * (column / 16) + column % 2;
            constexpr int lane2 = static_cast<int>(2 * ((column % 16) / 2));
            vFloat total = sum_lanes8(dst_reg[acc(j, k)]);
            vFloat out = dst_reg[pos(6, x)];
            v_if(lane_col2() == lane2) { out = total; }
            v_endif;
            dst_reg[pos(6, x)] = out;
        });
    });
    zero_padding_rows<6, T>();
    _llk_math_eltwise_sfpu_done_();
}

// ---------------------------------------------------------------- phase 2

// Positions 0 .. P - 1 of gather tile g's slots (in DST tile 0, cores 8 g
// ..), added to stat(x), which the first tile sets.
template <uint32_t K, uint32_t g, uint32_t P>
inline void sum_gather_tile() {
    _llk_math_eltwise_sfpu_start_(0);
    constexpr uint32_t slots = K - 8 * g < 8 ? K - 8 * g : 8;
    static_for<P>([](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        if constexpr (g == 0) {
            dst_reg[stat(x)] = vFloat(dst_reg[gathered(0, x)]);
        } else {
            dst_reg[stat(x)] = vFloat(dst_reg[stat(x)]) + vFloat(dst_reg[gathered(0, x)]);
        }
        static_for<slots - 1>([](auto ss) {
            constexpr uint32_t s = decltype(ss)::value + 1;
            dst_reg[stat(x)] = vFloat(dst_reg[stat(x)]) + vFloat(dst_reg[gathered(s, x)]);
        });
    });
    _llk_math_eltwise_sfpu_done_();
}

// The softmax weights of every query and source, each in every lane of its
// token's subvector (weight_of(q, j)).
template <uint32_t N, uint32_t Q, bool NORMALIZE>
inline void attention_weights(uint32_t inv_width_bits, uint32_t eps_bits) {
    _llk_math_eltwise_sfpu_start_(0);

    // r_j = rsqrt(sum of squares / W + eps) of the sources in group B of
    // face 1, rotated by four lanes into group A too, for each parity.
    static_for<2>([&](auto pp) {
        constexpr uint32_t parity = decltype(pp)::value;
        vFloat r = rsqrt_positive(
            vFloat(dst_reg[stat(2 + parity)]) * sfpi::as<vFloat>(vInt(inv_width_bits)) +
            sfpi::as<vFloat>(vInt(eps_bits)));
        vFloat rotated = rotate(rotate(rotate(rotate(r))));
        v_if(lane_col2() < 8) { r = rotated; }
        v_endif;
        dst_reg[rfull(parity)] = r;
    });

    // Logits, with lanes of no source or query at kNeg.
    static_for<4>([](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        constexpr uint32_t face = x / 2, parity = x % 2;
        constexpr bool group_a_valid = 2 * face < Q;
        constexpr bool group_b_valid = face == 0 && Q >= 2;
        vFloat logit = vFloat(dst_reg[stat(x)]) * vFloat(dst_reg[rfull(parity)]);
        // Source j = 2 (p % 4) + parity exists when 2 (p % 4) < N - parity.
        v_if((lane_col2() & 6) >= static_cast<int>(N - parity)) { logit = kNeg; }
        v_endif;
        if constexpr (!group_a_valid) {
            v_if(lane_col2() < 8) { logit = kNeg; }
            v_endif;
        }
        if constexpr (!group_b_valid) {
            v_if(lane_col2() >= 8) { logit = kNeg; }
            v_endif;
        }
        dst_reg[prob(x)] = logit;
    });

    // Softmax over each group's eight sources: both parities of a face.
    static_for<2>([](auto ff) {
        constexpr uint32_t face = decltype(ff)::value;
        vFloat m = max_groups(vmax(dst_reg[prob(2 * face)], dst_reg[prob(2 * face + 1)]));
        vFloat e0 = ckernel::sfpu::_sfpu_exp_fp32_accurate_(vFloat(dst_reg[prob(2 * face)]) - m);
        vFloat e1 = ckernel::sfpu::_sfpu_exp_fp32_accurate_(vFloat(dst_reg[prob(2 * face + 1)]) - m);
        vFloat inv = recip_positive(sum_groups(e0 + e1));
        if constexpr (NORMALIZE) {
            e0 = e0 * inv * vFloat(dst_reg[rfull(0)]);
            e1 = e1 * inv * vFloat(dst_reg[rfull(1)]);
        } else {
            e0 = e0 * inv;
            e1 = e1 * inv;
        }
        dst_reg[prob(2 * face)] = e0;
        dst_reg[prob(2 * face + 1)] = e1;
    });

    // Each weight in every lane of its subvector.
    static_for<Q>([](auto qq) {
        static_for<N>([](auto jj) {
            constexpr uint32_t q = decltype(qq)::value, j = decltype(jj)::value;
            constexpr uint32_t x = 2 * (q / 2) + j % 2;
            constexpr int lane2 = static_cast<int>(2 * (4 * (q % 2) + j / 2));
            vFloat w = dst_reg[prob(x)];
            v_if(lane_col2() != lane2) { w = 0.0f; }
            v_endif;
            dst_reg[weight_of(q, j)] = sum_lanes8(w);
        });
    });
    _llk_math_eltwise_sfpu_done_();
}

// m plus the weighted sources j.. of query q's mixture at tile c, position x.
template <uint32_t N, uint32_t CNT, uint32_t q, uint32_t c, uint32_t x, uint32_t j>
sfpi_inline vFloat mix_from(vFloat m) {
    if constexpr (j == N) {
        return m;
    } else {
        return mix_from<N, CNT, q, c, x, j + 1>(
            m + vFloat(dst_reg[weight_of(q, j)]) * vFloat(dst_reg[kept<CNT>(j, c, x)]));
    }
}

// The mixtures of tile c's kept sources, into the stage tile; with a weight
// also each lane's partial sums of their squares.
template <uint32_t N, uint32_t Q, uint32_t CNT, uint32_t c, bool SQUARES>
inline void mix() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<Q>([](auto qq) {
        constexpr uint32_t q = decltype(qq)::value;
        constexpr uint32_t u = q * CNT + c;
        static_for<4>([](auto xx) {
            constexpr uint32_t x = decltype(xx)::value;
            vFloat m = mix_from<N, CNT, q, c, x, 1>(
                vFloat(dst_reg[weight_of(q, 0)]) * vFloat(dst_reg[kept<CNT>(0, c, x)]));
            dst_reg[stage(u, x)] = m;
            if constexpr (SQUARES) {
                vFloat squares = m * m;
                if constexpr (c > 0 || x > 0) {
                    squares += vFloat(dst_reg[mix_squares(q)]);
                }
                dst_reg[mix_squares(q)] = squares;
            }
        });
    });
    _llk_math_eltwise_sfpu_done_();
}

// The second exchange row (tile 7): each query's sum of squares in column q.
template <uint32_t Q, uint32_t T>
inline void make_row2() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) { dst_reg[pos(7, decltype(xx)::value)] = vFloat(0.0f); });
    static_for<Q>([](auto qq) {
        constexpr uint32_t q = decltype(qq)::value;
        vFloat total = sum_lanes8(dst_reg[mix_squares(q)]);
        vFloat out = dst_reg[pos(7, 0)];
        v_if(lane_col2() == static_cast<int>(2 * q)) { out = total; }
        v_endif;
        dst_reg[pos(7, 0)] = out;
    });
    zero_padding_rows<7, T>();
    _llk_math_eltwise_sfpu_done_();
}

// ---------------------------------------------------------------- phase 3

// Each query's rstd from the gathered sums of squares, in every lane of
// its token's subvector.
template <uint32_t Q>
inline void mixture_rstd(uint32_t inv_width_bits, uint32_t eps_bits) {
    _llk_math_eltwise_sfpu_start_(0);
    vFloat r = rsqrt_positive(
        vFloat(dst_reg[stat(0)]) * sfpi::as<vFloat>(vInt(inv_width_bits)) + sfpi::as<vFloat>(vInt(eps_bits)));
    dst_reg[stat(0)] = r;
    static_for<Q>([](auto qq) {
        constexpr uint32_t q = decltype(qq)::value;
        vFloat v = dst_reg[stat(0)];
        v_if(lane_col2() != static_cast<int>(2 * q)) { v = 0.0f; }
        v_endif;
        dst_reg[rstd_of(q)] = sum_lanes8(v);
    });
    _llk_math_eltwise_sfpu_done_();
}

// Tile c's mixtures normalized and scaled by the weight rows of DST tile 4.
template <uint32_t Q, uint32_t CNT, uint32_t c>
inline void normalize_tile() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        vFloat a = dst_reg[pos(4, x)];
        vFloat b = a;
        vFloat cc = a;
        vFloat d = a;
        sfpi::subvec_transp(a, b, cc, d);
        dst_reg[weight_row(0, x)] = a;
        dst_reg[weight_row(1, x)] = b;
        dst_reg[weight_row(2, x)] = cc;
    });
    static_for<Q>([](auto qq) {
        constexpr uint32_t q = decltype(qq)::value;
        constexpr uint32_t u = q * CNT + c;
        static_for<4>([](auto xx) {
            constexpr uint32_t x = decltype(xx)::value;
            vFloat y = vFloat(dst_reg[stage(u, x)]) * vFloat(dst_reg[rstd_of(q)]);
            dst_reg[stage(u, x)] = y * vFloat(dst_reg[weight_row(q, x)]);
        });
    });
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace
#endif

namespace {

inline void sfpu_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

// Tile `tile` of `cb` into DST tile `dst`.
inline void copy_from(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

// DST tile `dst` into `cb`'s next reserved tile.
inline void pack_to(uint32_t dst, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
}

template <uint32_t CNT>
void run() {
    constexpr uint32_t cb_sources = get_named_compile_time_arg_val("cb_sources");
    constexpr uint32_t cb_addend = get_named_compile_time_arg_val("cb_addend");
    constexpr uint32_t cb_queries = get_named_compile_time_arg_val("cb_queries");
    constexpr uint32_t cb_weight = get_named_compile_time_arg_val("cb_weight");
    constexpr uint32_t cb_row1 = get_named_compile_time_arg_val("cb_row1");
    constexpr uint32_t cb_stash = get_named_compile_time_arg_val("cb_stash");
    constexpr uint32_t cb_gather1 = get_named_compile_time_arg_val("cb_gather1");
    constexpr uint32_t cb_row2 = get_named_compile_time_arg_val("cb_row2");
    constexpr uint32_t cb_gather2 = get_named_compile_time_arg_val("cb_gather2");
    constexpr uint32_t cb_mix = get_named_compile_time_arg_val("cb_mix");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t N = get_named_compile_time_arg_val("num_sources");
    constexpr uint32_t Q = get_named_compile_time_arg_val("num_queries");
    constexpr uint32_t T = get_named_compile_time_arg_val("tokens");
    constexpr bool has_addend = get_named_compile_time_arg_val("has_addend") != 0;
    constexpr bool has_weight = get_named_compile_time_arg_val("has_weight") != 0;
    constexpr bool normalize = get_named_compile_time_arg_val("normalize") != 0;
    constexpr uint32_t K = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t G = get_named_compile_time_arg_val("gather_tiles");
    constexpr uint32_t inv_width = get_named_compile_time_arg_val("inv_width");
    constexpr uint32_t eps_res = get_named_compile_time_arg_val("eps_res");
    constexpr uint32_t eps_norm = get_named_compile_time_arg_val("eps_norm");

    // Phase 1: this core's partial sums of squares and query products,
    // its exchange row and its sources' band-0 vectors kept for phase 2.
    cb_wait_front(cb_queries, CNT);
    cb_wait_front(cb_sources, N * CNT);
    if constexpr (has_addend) {
        cb_wait_front(cb_addend, CNT);
    }
    tile_regs_acquire();
    copy_from(cb_queries, 0, 3);
    if constexpr (CNT > 1) {
        copy_from(cb_queries, 1, 7);
    }
    sfpu_init();
    MATH((broadcast_queries<0>()));
    if constexpr (CNT > 1) {
        MATH((broadcast_queries<1>()));
    }
    static_for<CNT>([](auto cc) {
        static_for<N>([](auto jj) {
            constexpr uint32_t c = decltype(cc)::value, j = decltype(jj)::value;
            constexpr bool add_addend = has_addend && j == N - 1;
            copy_from(cb_sources, j * CNT + c, 1);
            if constexpr (add_addend) {
                copy_from(cb_addend, c, 2);
            }
            sfpu_init();
            MATH((accumulate<Q, CNT, j, c, add_addend>()));
        });
    });
    MATH((make_row1<N, Q, T>()));
    tile_regs_commit();
    cb_pop_front(cb_queries, CNT);
    cb_pop_front(cb_sources, N * CNT);
    if constexpr (has_addend) {
        cb_pop_front(cb_addend, CNT);
    }
    cb_reserve_back(cb_row1, 1);
    cb_reserve_back(cb_stash, 2);
    tile_regs_wait();
    pack_to(6, cb_row1);
    pack_to(4, cb_stash);
    pack_to(5, cb_stash);
    tile_regs_release();
    cb_push_back(cb_row1, 1);
    cb_push_back(cb_stash, 2);

    // Phase 2: the softmax weights from every core's rows, and the mixtures.
    cb_wait_front(cb_gather1, G);
    cb_wait_front(cb_stash, 2);
    tile_regs_acquire();
    static_for<G>([](auto gg) {
        copy_from(cb_gather1, decltype(gg)::value, 0);
        sfpu_init();
        MATH((sum_gather_tile<K, decltype(gg)::value, 4>()));
    });
    copy_from(cb_stash, 0, 4);
    copy_from(cb_stash, 1, 5);
    sfpu_init();
    MATH((attention_weights<N, Q, normalize>(inv_width, eps_res)));
    static_for<CNT>([](auto cc) { MATH((mix<N, Q, CNT, decltype(cc)::value, has_weight>())); });
    if constexpr (has_weight) {
        MATH((make_row2<Q, T>()));
    }
    tile_regs_commit();
    cb_pop_front(cb_gather1, G);
    cb_pop_front(cb_stash, 2);
    if constexpr (!has_weight) {
        cb_reserve_back(cb_output, 1);
        tile_regs_wait();
        pack_to(6, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, 1);
        return;
    }
    cb_reserve_back(cb_mix, 1);
    cb_reserve_back(cb_row2, 1);
    tile_regs_wait();
    pack_to(6, cb_mix);
    pack_to(7, cb_row2);
    tile_regs_release();
    cb_push_back(cb_mix, 1);
    cb_push_back(cb_row2, 1);

    // Phase 3: the mixtures' RMS norm and weight.
    cb_wait_front(cb_gather2, G);
    cb_wait_front(cb_mix, 1);
    cb_wait_front(cb_weight, CNT);
    tile_regs_acquire();
    static_for<G>([](auto gg) {
        copy_from(cb_gather2, decltype(gg)::value, 0);
        sfpu_init();
        MATH((sum_gather_tile<K, decltype(gg)::value, 1>()));
    });
    copy_from(cb_mix, 0, 6);
    sfpu_init();
    MATH((mixture_rstd<Q>(inv_width, eps_norm)));
    static_for<CNT>([](auto cc) {
        constexpr uint32_t c = decltype(cc)::value;
        copy_from(cb_weight, c, 4);
        sfpu_init();
        MATH((normalize_tile<Q, CNT, c>()));
    });
    tile_regs_commit();
    cb_pop_front(cb_gather2, G);
    cb_pop_front(cb_mix, 1);
    cb_pop_front(cb_weight, CNT);
    cb_reserve_back(cb_output, 1);
    tile_regs_wait();
    pack_to(6, cb_output);
    tile_regs_release();
    cb_push_back(cb_output, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_sources = get_named_compile_time_arg_val("cb_sources");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    compute_kernel_hw_startup(cb_sources, cb_sources, cb_output);
    if (count == 2) {
        run<2>();
    } else {
        run<1>();
    }
}
