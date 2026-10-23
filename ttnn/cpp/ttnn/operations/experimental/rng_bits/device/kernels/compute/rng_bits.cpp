// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// rng_bits compute: this core's random tiles, on the vector unit. State is
// in the state DFB; see rng_bits.hpp for the low/high key and counter mapping.
//
// Threefry runs one 32-lane vector at a time, all in registers. Philox keeps
// its four counter words in DST and fuses each multiply and XOR into one
// vector pass, sharing the partial products for the low and high halves.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#ifdef TRISC_MATH
#include "sfpi.h"

namespace rng {

// An SFPU load at DST offset 2 * d (d = 0..7) of a face gives lane l (0..31) the
// element at face row 4 * (d / 2) + l / 8 and column 2 * (l % 8) + d % 2, and
// the tile-id register holds 2 * l. Each lane's offset, in elements of a
// row-major [..., W] tensor, from the element of its vector's first lane:
// (l / 8) * W + 2 * (l % 8).
sfpi_inline sfpi::vUInt lane_offsets(uint32_t W) {
    const sfpi::vUInt tile_id = sfpi::vUInt(sfpi::vInt(sfpi::vConstTileId));
    sfpi::vUInt offset = tile_id & 14u;
    const sfpi::vInt row = sfpi::vInt(tile_id >> 4);
    v_if((row & 1) != 0) { offset += sfpi::vUInt(W); }
    v_endif;
    v_if((row & 2) != 0) { offset += sfpi::vUInt(W << 1); }
    v_endif;
    return offset;
}

// The index of the element of lane 0 of vector d of a face whose first element
// has index `face_index`.
inline uint32_t vector_index(uint32_t face_index, uint32_t W, uint32_t d) {
    return face_index + 4 * (d >> 1) * W + (d & 1);
}

template <int r, typename T>
sfpi_inline T rotate_left(T x) {
    return (x << r) | (x >> (32 - r));
}

// Four Threefry-2x32 rounds with rotations r0..r3.
template <int r0, int r1, int r2, int r3, typename T>
sfpi_inline void threefry_rounds(T& x0, T& x1) {
    x0 += x1;
    x1 = rotate_left<r0>(x1) ^ x0;
    x0 += x1;
    x1 = rotate_left<r1>(x1) ^ x0;
    x0 += x1;
    x1 = rotate_left<r2>(x1) ^ x0;
    x0 += x1;
    x1 = rotate_left<r3>(x1) ^ x0;
}

// Shared by scalar stream-key derivation and vector output generation.
// x0/x1 already include the initial key addition.
template <typename T>
sfpi_inline void threefry(T& x0, T& x1, uint32_t ks0, uint32_t ks1, uint32_t ks2) {
    threefry_rounds<13, 15, 26, 6>(x0, x1);
    x0 += T(ks1);
    x1 += T(ks2 + 1);
    threefry_rounds<17, 29, 16, 24>(x0, x1);
    x0 += T(ks2);
    x1 += T(ks0 + 2);
    threefry_rounds<13, 15, 26, 6>(x0, x1);
    x0 += T(ks0);
    x1 += T(ks1 + 3);
    threefry_rounds<17, 29, 16, 24>(x0, x1);
    x0 += T(ks1);
    x1 += T(ks2 + 4);
    threefry_rounds<13, 15, 26, 6>(x0, x1);
    x0 += T(ks2);
    x1 += T(ks0 + 5);
}

// One tile of Threefry-2x32-20 words into DST tile `dst`: the tile's element
// (0, 0) has index `base`, its rows are W elements apart, and rows from
// `rows_valid` on are padding, which is left as it is. ks is the three-word key schedule.
inline void threefry_tile(
    uint32_t dst, uint32_t base, uint32_t W, uint32_t rows_valid, uint32_t ks0, uint32_t ks1, uint32_t ks2, uint32_t w1) {
    _llk_math_eltwise_sfpu_start_(dst);
    const sfpi::vUInt lanes = lane_offsets(W);
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_row = 16 * (face >> 1);
        const uint32_t face_index = base + face_row * W + 16 * (face & 1);
        for (uint32_t d = 0; d < 8; ++d) {
            if (face_row + 4 * (d >> 1) < rows_valid) {
                // Counter (i, w1) plus the first key words (ks0, ks1).
                sfpi::vUInt x0 = lanes + sfpi::vUInt(vector_index(face_index, W, d) + ks0);
                sfpi::vUInt x1 = sfpi::vUInt(w1 + ks1);
                threefry(x0, x1, ks0, ks1, ks2);
                sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = sfpi::vInt(x0);
            }
            sfpi::dst_reg++;
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}

// The elements' indices into DST tile `dst` (see threefry_tile).
inline void index_tile(uint32_t dst, uint32_t base, uint32_t W) {
    _llk_math_eltwise_sfpu_start_(dst);
    const sfpi::vUInt lanes = lane_offsets(W);
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_index = base + 16 * (face >> 1) * W + 16 * (face & 1);
        for (uint32_t d = 0; d < 8; ++d) {
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() =
                sfpi::vInt(lanes + sfpi::vUInt(vector_index(face_index, W, d)));
            sfpi::dst_reg++;
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}

// Multiply a 32-bit word by a constant, sharing the 23-bit partial products
// used by the existing mul_int32 and mulhi_uint32 kernels. SFPMUL24 returns
// 23-bit halves; middle is the sum of the three terms at bit 23, so its low
// nine bits contribute to lo and the rest to hi.
template <uint32_t multiplier>
sfpi_inline void mulhilo(sfpi::vUInt x, sfpi::vUInt& hi, sfpi::vUInt& lo) {
    const sfpi::vUInt y = sfpi::vUInt(multiplier);
    const sfpi::vUInt xh = x >> 23;
    const sfpi::vUInt yh = sfpi::vUInt(multiplier >> 23);
    sfpi::vUInt middle = sfpi::as<sfpi::vUInt>(sfpi::fractional_mul(x, y, sfpi::FractionalHalf::High));
    middle += sfpi::as<sfpi::vUInt>(sfpi::fractional_mul(xh, y));
    middle += sfpi::as<sfpi::vUInt>(sfpi::fractional_mul(x, yh));
    hi = sfpi::as<sfpi::vUInt>(sfpi::fractional_mul(xh, yh));
    hi += sfpi::as<sfpi::vUInt>(sfpi::fractional_mul(xh, y, sfpi::FractionalHalf::High));
    hi += sfpi::as<sfpi::vUInt>(sfpi::fractional_mul(x, yh, sfpi::FractionalHalf::High));
    hi = (hi << 14) + (middle >> 9);
    lo = sfpi::as<sfpi::vUInt>(sfpi::fractional_mul(x, y)) + (middle << 23);
}

// One half of a Philox round: multiply one counter word and mix its high
// half with another word and the key. Keep the four counter words in DST so
// the multiply's temporaries fit in the eight SFPU registers.
template <uint32_t multiplier, bool last_round = false>
inline void philox_half_round(uint32_t word, uint32_t other, uint32_t key, uint32_t rows_valid) {
    _llk_math_eltwise_sfpu_start_(0);
    for (uint32_t face = 0; face < 4; ++face) {
#pragma GCC unroll 8
        for (uint32_t d = 0; d < 8; ++d) {
            if (16 * (face >> 1) + 4 * (d >> 1) < rows_valid) {
                sfpi::vUInt hi, lo;
                mulhilo<multiplier>(sfpi::vUInt(sfpi::dst_reg[32 * word]), hi, lo);
                if constexpr (!last_round) {
                    sfpi::dst_reg[32 * word].mode<sfpi::DataLayout::I32>() = sfpi::vInt(lo);
                }
                hi ^= sfpi::vUInt(sfpi::dst_reg[32 * other]);
                hi ^= sfpi::vUInt(key);
                sfpi::dst_reg[32 * other].mode<sfpi::DataLayout::I32>() = sfpi::vInt(hi);
            }
            sfpi::dst_reg++;
        }
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}

}  // namespace rng
#endif

void kernel_main() {
    constexpr bool philox = get_arg(args::philox) != 0;
    constexpr uint32_t state_words = get_arg(args::state_words);
    const uint32_t W = get_arg(args::W);
    const uint32_t H = get_arg(args::H);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t Ht = get_arg(args::Ht);
    const uint32_t first_tile = get_arg(args::first_tile);
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer state_dfb(dfb::state);
    DataflowBuffer out_dfb(dfb::out);
    compute_kernel_hw_startup(dfb::state, dfb::out);

    // Low/high pairs occupy the first two columns of the tile's 16-wide first face.
    // The mover restores the original state before push_back; wait_front keeps
    // these reads after its temporary update and the new-state write barrier.
    state_dfb.wait_front(1);
    const uint32_t state_address = state_dfb.get_tile_address(0);
    uint32_t w0 = 0, w1 = 0, w2 = 0, key_high = 0, counter_high = 0, extra_high = 0;
    uint32_t ks0 = 0, ks1 = 0;
    MATH({
        const volatile uint32_t* words = reinterpret_cast<const volatile uint32_t*>(state_address);
        w0 = words[0];
        w1 = words[16];
        w2 = philox && state_words > 2 ? words[32] : 0;
        key_high = words[1];
        counter_high = words[17];
        extra_high = philox && state_words > 2 ? words[33] : 0;
        if constexpr (!philox) {
            // Hash the counter high word under the seed once per core. XORing
            // it into the seed would alias neighboring seeds after 2^32 units.
            ks0 = counter_high + w0;
            ks1 = 0x1BD11BDAu + key_high;
            rng::threefry(ks0, ks1, w0, key_high, w0 ^ key_high ^ 0x1BD11BDAu);
        }
    });

    for (uint32_t tile = first_tile; tile < first_tile + num_tiles; ++tile) {
        // Tile (b, tr, tc) starts at element (b * H + tr * 32) * W + tc * 32.
        const uint32_t tc = tile % Wt;
        const uint32_t tr = (tile / Wt) % Ht;
        const uint32_t b = tile / (Wt * Ht);
        const uint32_t base = (b * H + tr * 32) * W + tc * 32;
        uint32_t out_dst = 0;
        const uint32_t rows_valid = H - tr * 32 < 32 ? H - tr * 32 : 32;

        tile_regs_acquire();
        if constexpr (!philox) {
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
            MATH((rng::threefry_tile(0, base, W, rows_valid, ks0, ks1, ks0 ^ ks1 ^ 0x1BD11BDA, w1)));
        } else {
            // The four counter words, renamed after each round.
            uint32_t x[4] = {0, 1, 2, 3};
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
            MATH((rng::index_tile(x[0], base, W)));
            fill_tile_init();
            fill_tile_int<DataFormat::UInt32>(x[1], w1);
            fill_tile_int<DataFormat::UInt32>(x[2], w2);
            fill_tile_int<DataFormat::UInt32>(x[3], counter_high);
            uint32_t k0 = w0;
            uint32_t k1 = key_high ^ extra_high;
            for (uint32_t round = 0; round < 9; ++round) {
                // (x0, x1, x2, x3) = (hi(x2 * M1) ^ x1 ^ k0, lo(x2 * M1), hi(x0 * M0) ^ x3 ^ k1, lo(x0 * M0))
                MATH((rng::philox_half_round<0xD2511F53>(x[0], x[3], k1, rows_valid)));
                MATH((rng::philox_half_round<0xCD9E8D57>(x[2], x[1], k0, rows_valid)));
                k0 += 0x9E3779B9;
                k1 += 0xBB67AE85;
                const uint32_t low0 = x[0];
                x[0] = x[1];
                x[1] = x[2];
                x[2] = x[3];
                x[3] = low0;
            }
            // Only the first word of the final Philox round is returned.
            MATH((rng::philox_half_round<0xCD9E8D57, true>(x[2], x[1], k0, rows_valid)));
            out_dst = x[1];
        }
        tile_regs_commit();

        out_dfb.reserve_back(1);
        tile_regs_wait();
        pack_tile(out_dst, dfb::out);
        tile_regs_release();
        out_dfb.push_back(1);
    }
    state_dfb.pop_front(1);
}
