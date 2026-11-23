// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
//
// Inkling 2's loop-message controller (loop_mix.hpp) on this core's two tile
// columns. Streams s are the destinations (s < Qd) and then the messages.
//
// SFPU lane layout (fp32 DST): vector d (0..7) of a face holds, in lane l
// (0..31), the face's element at row 4 * (d / 2) + l / 8, column
// 2 * (l % 8) + d % 2. Tokens are rows 0..3, band 0: positions x = face 0
// even, face 0 odd, face 1 even, face 1 odd, token t the 8-lane subvector t.
//
//   A: each stream's sum of squares over this core's columns, per token
//      (exchange row: column s), packed for the first exchange.
//   B: rstd of each stream from every core's rows; this core's columns of
//      each stream normalized and rounded to BF16 (tokens in rows 0..3, the
//      other rows zero); its share of the down projection,
//      [dn | mn] @ down over its rows of K, packed for the second exchange.
//   B3 (first core): the shares summed into hidden, rounded to BF16.
//   C: per destination q, both columns: gate and forget = tanh(bf16(hidden @
//      gain or forget)), out = dest + (gate * message - forget * dest).
#include <cstdint>
#include <type_traits>
#include <utility>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
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
#include "ckernel_sfpu_tanh.h"

namespace {

using sfpi::dst_reg;
using sfpi::vFloat;
using sfpi::vInt;
using sfpi::vUInt;

constexpr uint32_t vec(uint32_t t, uint32_t f, uint32_t b, uint32_t parity) { return t * 32 + f * 8 + b * 2 + parity; }
constexpr uint32_t pos(uint32_t t, uint32_t x) { return vec(t, x / 2, 0, x % 2); }
// Slot s of a gather tile in DST tile t.
constexpr uint32_t slot(uint32_t t, uint32_t s, uint32_t x) { return vec(t, 2 * (s / 4) + x / 2, s % 4, x % 2); }
// Working vectors in DST tile 7 (phases A and B).
constexpr uint32_t squares(uint32_t s) { return 7 * 32 + s; }
constexpr uint32_t stat_row() { return 7 * 32 + 8; }
constexpr uint32_t rstd_of(uint32_t s) { return 7 * 32 + 16 + s; }

sfpi_inline vInt lane_col2() { return vInt(sfpi::vConstTileId) & 14; }
sfpi_inline vInt lane_row16() { return vInt(sfpi::vConstTileId) & 48; }
sfpi_inline vFloat rotate(vFloat v) { return vFloat(sfpi::subvec_shflror1(v)); }

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

// x rounded to BF16 (to nearest, ties to even) and kept in FP32.
sfpi_inline vFloat round_bf16(vFloat x) {
    vUInt bits = sfpi::as<vUInt>(x);
    vUInt lsb = (bits >> 16) & 1;
    bits = bits + 0x7fffU + lsb;
    bits = bits & 0xFFFF0000U;
    return sfpi::as<vFloat>(bits);
}

// Phase A: tile 0 (stream s, this core's column c) into s's partial sums.
template <uint32_t s, bool first>
inline void add_squares() {
    _llk_math_eltwise_sfpu_start_(0);
    vFloat v0 = dst_reg[pos(0, 0)];
    vFloat v1 = dst_reg[pos(0, 1)];
    vFloat v2 = dst_reg[pos(0, 2)];
    vFloat v3 = dst_reg[pos(0, 3)];
    vFloat sum = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    if constexpr (!first) {
        sum += vFloat(dst_reg[squares(s)]);
    }
    dst_reg[squares(s)] = sum;
    _llk_math_eltwise_sfpu_done_();
}

// The exchange row in tile 6: stream s's total in column s (face 0, even
// columns, lane s), rows past the tokens zero.
template <uint32_t S, uint32_t T>
inline void make_row1() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) { dst_reg[pos(6, decltype(xx)::value)] = vFloat(0.0f); });
    static_for<S>([](auto ss) {
        constexpr uint32_t s = decltype(ss)::value;
        vFloat total = sum_lanes8(dst_reg[squares(s)]);
        vFloat out = dst_reg[pos(6, 0)];
        v_if(lane_col2() == static_cast<int>(2 * s)) { out = total; }
        v_endif;
        dst_reg[pos(6, 0)] = out;
    });
    if constexpr (T < 4) {
        vFloat out = dst_reg[pos(6, 0)];
        v_if(lane_row16() >= static_cast<int>(16 * T)) { out = 0.0f; }
        v_endif;
        dst_reg[pos(6, 0)] = out;
    }
    _llk_math_eltwise_sfpu_done_();
}

// Phase B: slots `count` of the gather tile in DST tile 0, face 0 even
// position, added to the stat row (which the first tile sets).
template <bool first>
inline void sum_stat_slots(uint32_t count) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<8>([&](auto ss) {
        constexpr uint32_t s = decltype(ss)::value;
        if (s < count) {
            vFloat v = dst_reg[slot(0, s, 0)];
            if (first && s == 0) {
                dst_reg[stat_row()] = v;
            } else {
                dst_reg[stat_row()] = vFloat(dst_reg[stat_row()]) + v;
            }
        }
    });
    _llk_math_eltwise_sfpu_done_();
}

// Each stream's rstd in every lane of its token's subvector.
template <uint32_t S>
inline void stream_rstd(uint32_t inv_width_bits, uint32_t eps_bits) {
    _llk_math_eltwise_sfpu_start_(0);
    vFloat r = rsqrt_positive(
        vFloat(dst_reg[stat_row()]) * sfpi::as<vFloat>(vInt(inv_width_bits)) + sfpi::as<vFloat>(vInt(eps_bits)));
    dst_reg[stat_row()] = r;
    static_for<S>([](auto ss) {
        constexpr uint32_t s = decltype(ss)::value;
        vFloat v = dst_reg[stat_row()];
        v_if(lane_col2() != static_cast<int>(2 * s)) { v = 0.0f; }
        v_endif;
        dst_reg[rstd_of(s)] = sum_lanes8(v);
    });
    _llk_math_eltwise_sfpu_done_();
}

// DST tile t (stream s's tile) normalized in its token rows, the other rows
// zero; the rstd vectors are in DST tile 7.
template <uint32_t t, uint32_t s>
inline void normalize_tile() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        dst_reg[pos(t, x)] = vFloat(dst_reg[pos(t, x)]) * vFloat(dst_reg[rstd_of(s)]);
    });
    static_for<32>([](auto ii) {
        constexpr uint32_t i = decltype(ii)::value;
        constexpr bool band0 = i == 0 || i == 1 || i == 8 || i == 9;
        if constexpr (!band0) {
            dst_reg[t * 32 + i] = vFloat(0.0f);
        }
    });
    _llk_math_eltwise_sfpu_done_();
}

// Phase B3: slots `count` of the gather tile in DST tile 7 added into the
// token rows of DST tile j (set by the first group).
template <uint32_t j, bool first>
inline void sum_hidden_slots(uint32_t count) {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([&](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        static_for<8>([&](auto ss) {
            constexpr uint32_t s = decltype(ss)::value;
            if (s < count) {
                vFloat v = dst_reg[slot(7, s, x)];
                if (first && s == 0) {
                    dst_reg[pos(j, x)] = v;
                } else {
                    dst_reg[pos(j, x)] = vFloat(dst_reg[pos(j, x)]) + v;
                }
            }
        });
    });
    _llk_math_eltwise_sfpu_done_();
}

// Tile j's other rows zero.
template <uint32_t j>
inline void zero_padding_rows() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<32>([](auto ii) {
        constexpr uint32_t i = decltype(ii)::value;
        constexpr bool band0 = i == 0 || i == 1 || i == 8 || i == 9;
        if constexpr (!band0) {
            dst_reg[j * 32 + i] = vFloat(0.0f);
        }
    });
    _llk_math_eltwise_sfpu_done_();
}

// Phase C, column c: tiles c (gate pre-activation), 2 + c (forget), 4 + c
// (dest), 6 + c (message): tile 4 + c's token rows = d + (tanh(bf16(g)) * m
// - tanh(bf16(f)) * d).
template <uint32_t c>
inline void combine() {
    _llk_math_eltwise_sfpu_start_(0);
    static_for<4>([](auto xx) {
        constexpr uint32_t x = decltype(xx)::value;
        const vFloat g = ckernel::sfpu::_sfpu_tanh_fp32_accurate_(round_bf16(dst_reg[pos(c, x)]));
        const vFloat f = ckernel::sfpu::_sfpu_tanh_fp32_accurate_(round_bf16(dst_reg[pos(2 + c, x)]));
        const vFloat d = dst_reg[pos(4 + c, x)];
        const vFloat m = dst_reg[pos(6 + c, x)];
        dst_reg[pos(4 + c, x)] = d + (g * m - f * d);
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

inline void pack_to(uint32_t dst, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
}

inline void matmul_setup(uint32_t in0, uint32_t in1) {
    // Matmul unpacks in1 to SrcA and in0 to SrcB.
    reconfig_data_format(in1, in0);
    matmul_init(in0, in1);
}

}  // namespace

void kernel_main() {
    const bool first = get_arg_val<uint32_t>(0) != 0;
    constexpr uint32_t cb_dest = get_named_compile_time_arg_val("cb_dest");
    constexpr uint32_t cb_messages = get_named_compile_time_arg_val("cb_messages");
    constexpr uint32_t cb_down = get_named_compile_time_arg_val("cb_down");
    constexpr uint32_t cb_gain = get_named_compile_time_arg_val("cb_gain");
    constexpr uint32_t cb_forget = get_named_compile_time_arg_val("cb_forget");
    constexpr uint32_t cb_row1 = get_named_compile_time_arg_val("cb_row1");
    constexpr uint32_t cb_gather1 = get_named_compile_time_arg_val("cb_gather1");
    constexpr uint32_t cb_normed = get_named_compile_time_arg_val("cb_normed");
    constexpr uint32_t cb_partial = get_named_compile_time_arg_val("cb_partial");
    constexpr uint32_t cb_gather2 = get_named_compile_time_arg_val("cb_gather2");
    constexpr uint32_t cb_hidden = get_named_compile_time_arg_val("cb_hidden");
    constexpr uint32_t cb_output = get_named_compile_time_arg_val("cb_output");
    constexpr uint32_t cb_rstd = get_named_compile_time_arg_val("cb_rstd");
    constexpr uint32_t Q = get_named_compile_time_arg_val("num_queries");
    constexpr uint32_t Qd = get_named_compile_time_arg_val("num_dest");
    constexpr uint32_t Rt = get_named_compile_time_arg_val("Rt");
    constexpr uint32_t T = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t K = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t G1 = get_named_compile_time_arg_val("gather1_tiles");
    constexpr uint32_t inv_width = get_named_compile_time_arg_val("inv_width");
    constexpr uint32_t eps = get_named_compile_time_arg_val("eps");
    constexpr uint32_t cnt = 2;
    constexpr uint32_t S = Qd + Q;
    constexpr uint32_t QR = Q * Rt;
    constexpr uint32_t groups = (K + 7) / 8;
    // A stream's tile c in its CB.
    auto stream_cb = [](uint32_t s) { return s < Qd ? cb_dest : cb_messages; };
    auto stream_tile = [](uint32_t s, uint32_t c) { return (s < Qd ? s : s - Qd) * cnt + c; };

    compute_kernel_hw_startup(cb_dest, cb_dest, cb_row1);

    // Phase A.
    cb_wait_front(cb_dest, Qd * cnt);
    cb_wait_front(cb_messages, Q * cnt);
    tile_regs_acquire();
    static_for<S>([&](auto ss) {
        static_for<cnt>([&](auto cc) {
            constexpr uint32_t s = decltype(ss)::value, c = decltype(cc)::value;
            copy_from(stream_cb(s), stream_tile(s, c), 0);
            sfpu_init();
            MATH((add_squares<s, c == 0>()));
        });
    });
    MATH((make_row1<S, T>()));
    tile_regs_commit();
    cb_reserve_back(cb_row1, 1);
    tile_regs_wait();
    pack_to(6, cb_row1);
    tile_regs_release();
    cb_push_back(cb_row1, 1);

    // Phase B: rstd, the normalized BF16 tiles (stream-major, then column),
    // in two DST rounds of S tiles each (column 0, then 1); the first keeps
    // the rstd vectors (DST tile 7) in cb_rstd for the second.
    cb_wait_front(cb_gather1, G1);
    cb_reserve_back(cb_normed, S * cnt);
    static_for<cnt>([&](auto cc) {
        constexpr uint32_t c = decltype(cc)::value;
        tile_regs_acquire();
        if constexpr (c == 0) {
            static_for<G1>([&](auto gg) {
                constexpr uint32_t g = decltype(gg)::value;
                copy_from(cb_gather1, g, 0);
                sfpu_init();
                MATH((sum_stat_slots<g == 0>(K - 8 * g < 8 ? K - 8 * g : 8)));
            });
            MATH((stream_rstd<S>(inv_width, eps)));
        } else {
            cb_wait_front(cb_rstd, 1);
            copy_from(cb_rstd, 0, 7);
        }
        static_for<S>([&](auto ss) {
            constexpr uint32_t s = decltype(ss)::value;
            copy_from(stream_cb(s), stream_tile(s, c), s);
            sfpu_init();
            MATH((normalize_tile<s, s>()));
        });
        tile_regs_commit();
        tile_regs_wait();
        // Packed in (c, s) order; the matmuls index normed tile c * S + s.
        static_for<S>([](auto ss) { pack_to(decltype(ss)::value, cb_normed); });
        if constexpr (c == 0) {
            cb_reserve_back(cb_rstd, 1);
            pack_to(7, cb_rstd);
        }
        tile_regs_release();
        if constexpr (c == 0) {
            cb_push_back(cb_rstd, 1);
        }
    });
    cb_pop_front(cb_rstd, 1);
    cb_push_back(cb_normed, S * cnt);
    cb_pop_front(cb_gather1, G1);

    // This core's share of the down projection.
    cb_wait_front(cb_normed, S * cnt);
    cb_wait_front(cb_down, Q * 2 * cnt * Rt);
    tile_regs_acquire();
    matmul_setup(cb_normed, cb_down);
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t qd = Qd == 1 ? 0 : q;
        for (uint32_t rt = 0; rt < Rt; ++rt) {
            for (uint32_t c = 0; c < cnt; ++c) {
                // down tile (q, half, c, rt), normed tile (c, stream).
                matmul_tiles(cb_normed, cb_down, c * S + qd, ((q * 2 + 0) * cnt + c) * Rt + rt, q * Rt + rt);
                matmul_tiles(cb_normed, cb_down, c * S + Qd + q, ((q * 2 + 1) * cnt + c) * Rt + rt, q * Rt + rt);
            }
        }
    }
    tile_regs_commit();
    cb_reserve_back(cb_partial, QR);
    tile_regs_wait();
    for (uint32_t j = 0; j < QR; ++j) {
        pack_to(j, cb_partial);
    }
    tile_regs_release();
    cb_push_back(cb_partial, QR);
    cb_pop_front(cb_normed, S * cnt);

    // Phase B3 on the first core: hidden from every core's shares (gather
    // tile (group, j) holds cores 8 group .. of tile j).
    if (first) {
        cb_wait_front(cb_gather2, groups * QR);
        tile_regs_acquire();
        for (uint32_t group = 0; group < groups; ++group) {
            const uint32_t count = K - 8 * group < 8 ? K - 8 * group : 8;
            static_for<QR>([&](auto jj) {
                constexpr uint32_t j = decltype(jj)::value;
                copy_from(cb_gather2, group * QR + j, 7);
                sfpu_init();
                if (group == 0) {
                    MATH((sum_hidden_slots<j, true>(count)));
                } else {
                    MATH((sum_hidden_slots<j, false>(count)));
                }
            });
        }
        static_for<QR>([](auto jj) { MATH((zero_padding_rows<decltype(jj)::value>())); });
        tile_regs_commit();
        cb_reserve_back(cb_hidden, QR);
        tile_regs_wait();
        for (uint32_t j = 0; j < QR; ++j) {
            pack_to(j, cb_hidden);
        }
        tile_regs_release();
        cb_push_back(cb_hidden, QR);
        cb_pop_front(cb_gather2, groups * QR);
    }

    // Phase C.
    cb_wait_front(cb_hidden, QR);
    cb_wait_front(cb_gain, Q * Rt * cnt);
    cb_wait_front(cb_forget, Q * Rt * cnt);
    // Both columns of a destination in one DST round.
    for (uint32_t q = 0; q < Q; ++q) {
        const uint32_t qd = Qd == 1 ? 0 : q;
        tile_regs_acquire();
        matmul_setup(cb_hidden, cb_gain);
        for (uint32_t c = 0; c < cnt; ++c) {
            for (uint32_t kt = 0; kt < Rt; ++kt) {
                matmul_tiles(cb_hidden, cb_gain, q * Rt + kt, (q * Rt + kt) * cnt + c, c);
            }
        }
        matmul_setup(cb_hidden, cb_forget);
        for (uint32_t c = 0; c < cnt; ++c) {
            for (uint32_t kt = 0; kt < Rt; ++kt) {
                matmul_tiles(cb_hidden, cb_forget, q * Rt + kt, (q * Rt + kt) * cnt + c, 2 + c);
            }
        }
        for (uint32_t c = 0; c < cnt; ++c) {
            copy_from(cb_dest, qd * cnt + c, 4 + c);
            copy_from(cb_messages, q * cnt + c, 6 + c);
        }
        sfpu_init();
        MATH((ckernel::sfpu::tanh_init<false, true>()));
        MATH((combine<0>()));
        MATH((combine<1>()));
        tile_regs_commit();
        cb_reserve_back(cb_output, cnt);
        tile_regs_wait();
        pack_to(4, cb_output);
        pack_to(5, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, cnt);
    }
    cb_pop_front(cb_hidden, QR);
    cb_pop_front(cb_gain, Q * Rt * cnt);
    cb_pop_front(cb_forget, Q * Rt * cnt);
    cb_pop_front(cb_down, Q * 2 * cnt * Rt);
    cb_pop_front(cb_dest, Qd * cnt);
    cb_pop_front(cb_messages, Q * cnt);
}
