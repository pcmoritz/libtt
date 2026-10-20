// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0

// rng_bits compute: this core's tiles of random bits, on the vector unit. The
// state's first page is in the state DFB; the key is (w0, 0), and element i
// (row-major) is the first word of Threefry-2x32-20 of counter (i, w1), or of
// Philox-4x32-10 of counter (i, w1, w2, 0).
//
// Threefry runs one 32-lane vector at a time, all in registers. Philox works on
// whole tiles in DST with the existing 32-bit multiply kernels (mul_int32 for
// the low and mulhi_uint32 for the high half of the product), so it holds its
// four words, the multiplier and both high halves in seven DST tiles.

#include <cstdint>

#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/bitwise.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/mul_int_sfpu.h"
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

template <int r>
sfpi_inline sfpi::vUInt rotate_left(sfpi::vUInt x) {
    return (x << r) | (x >> (32 - r));
}

// Four Threefry-2x32 rounds with rotations r0..r3.
template <int r0, int r1, int r2, int r3>
sfpi_inline void threefry_rounds(sfpi::vUInt& x0, sfpi::vUInt& x1) {
    x0 += x1;
    x1 = rotate_left<r0>(x1) ^ x0;
    x0 += x1;
    x1 = rotate_left<r1>(x1) ^ x0;
    x0 += x1;
    x1 = rotate_left<r2>(x1) ^ x0;
    x0 += x1;
    x1 = rotate_left<r3>(x1) ^ x0;
}

// One tile of Threefry-2x32-20 words into DST tile `dst`: the tile's element
// (0, 0) has index `base`, its rows are W elements apart, and rows from
// `rows_valid` on are padding, which is left as it is. ks = (w0, 0, w0 ^ C240).
template <uint32_t width>
inline void threefry_tile(
    uint32_t dst, uint32_t base, uint32_t W, uint32_t rows_valid, uint32_t ks0, uint32_t ks2, uint32_t w1) {
    _llk_math_eltwise_sfpu_start_(dst);
    const sfpi::vUInt lanes = lane_offsets(W);
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t face_row = 16 * (face >> 1);
        const uint32_t face_index = base + face_row * W + 16 * (face & 1);
        for (uint32_t d = 0; d < 8; ++d) {
            if (face_row + 4 * (d >> 1) < rows_valid) {
                // Counter (i, w1) plus the first key words (ks0, ks1 = 0).
                sfpi::vUInt x0 = lanes + sfpi::vUInt(vector_index(face_index, W, d) + ks0);
                sfpi::vUInt x1 = sfpi::vUInt(w1);
                threefry_rounds<13, 15, 26, 6>(x0, x1);
                // x0 += ks1 (0)
                x1 += sfpi::vUInt(ks2 + 1);
                threefry_rounds<17, 29, 16, 24>(x0, x1);
                x0 += sfpi::vUInt(ks2);
                x1 += sfpi::vUInt(ks0 + 2);
                threefry_rounds<13, 15, 26, 6>(x0, x1);
                x0 += sfpi::vUInt(ks0);
                x1 += sfpi::vUInt(3u);  // ks1 + 3
                threefry_rounds<17, 29, 16, 24>(x0, x1);
                // x0 += ks1 (0)
                x1 += sfpi::vUInt(ks2 + 4);
                threefry_rounds<13, 15, 26, 6>(x0, x1);
                x0 += sfpi::vUInt(ks2);
                if constexpr (width < 32) {
                    x0 &= sfpi::vUInt((1u << width) - 1);
                }
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

}  // namespace rng
#endif

void kernel_main() {
    constexpr bool philox = get_arg(args::philox) != 0;
    constexpr uint32_t width = get_arg(args::width);
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

    // The state's words, read in place by the math thread.
    state_dfb.wait_front(1);
    const uint32_t state_address = state_dfb.get_tile_address(0);
    uint32_t w0 = 0, w1 = 0, w2 = 0;
    MATH({
        const volatile uint32_t* words = reinterpret_cast<const volatile uint32_t*>(state_address);
        w0 = words[0];
        w1 = words[1];
        w2 = philox && state_words > 2 ? words[2] : 0;
    });

    for (uint32_t tile = first_tile; tile < first_tile + num_tiles; ++tile) {
        // Tile (b, tr, tc) starts at element (b * H + tr * 32) * W + tc * 32.
        const uint32_t tc = tile % Wt;
        const uint32_t tr = (tile / Wt) % Ht;
        const uint32_t b = tile / (Wt * Ht);
        const uint32_t base = (b * H + tr * 32) * W + tc * 32;
        uint32_t out_dst = 0;

        tile_regs_acquire();
        if constexpr (!philox) {
            const uint32_t rows_valid = H - tr * 32 < 32 ? H - tr * 32 : 32;
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
            MATH((rng::threefry_tile<width>(0, base, W, rows_valid, w0, w0 ^ 0x1BD11BDA, w1)));
        } else {
            // DST tiles: the counter words x[0..3] (renamed each round), the
            // multiplier, and the two products' high halves.
            constexpr uint32_t kMultiplier = 4;
            constexpr uint32_t kHigh0 = 5;
            constexpr uint32_t kHigh1 = 6;
            uint32_t x[4] = {0, 1, 2, 3};
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
            MATH((rng::index_tile(x[0], base, W)));
            fill_tile_init();
            fill_tile_int<DataFormat::UInt32>(x[1], w1);
            fill_tile_int<DataFormat::UInt32>(x[2], w2);
            fill_tile_int<DataFormat::UInt32>(x[3], 0);
            uint32_t k0 = w0;
            uint32_t k1 = 0;
            for (uint32_t round = 0; round < 10; ++round) {
                if (round > 0) {
                    k0 += 0x9E3779B9;
                    k1 += 0xBB67AE85;
                }
                // (x0, x1, x2, x3) = (hi(x2 * M1) ^ x1 ^ k0, lo(x2 * M1), hi(x0 * M0) ^ x3 ^ k1, lo(x0 * M0))
                fill_tile_init();
                fill_tile_int<DataFormat::UInt32>(kMultiplier, 0xD2511F53);
                mulhi_uint32_tile_init();
                mulhi_uint32_tile(x[0], kMultiplier, kHigh0);
                mul_int_tile_init<DataFormat::UInt32>();
                mul_int_tile<DataFormat::UInt32>(x[0], kMultiplier, x[0]);
                fill_tile_init();
                fill_tile_int<DataFormat::UInt32>(kMultiplier, 0xCD9E8D57);
                mulhi_uint32_tile_init();
                mulhi_uint32_tile(x[2], kMultiplier, kHigh1);
                mul_int_tile_init<DataFormat::UInt32>();
                mul_int_tile<DataFormat::UInt32>(x[2], kMultiplier, x[2]);
                binary_bitwise_tile_init();
                bitwise_xor_binary_tile<DataFormat::UInt32>(kHigh1, x[1], x[1]);
                bitwise_xor_binary_tile<DataFormat::UInt32>(kHigh0, x[3], x[3]);
                bitwise_xor_tile_init();
                bitwise_xor_tile<DataFormat::UInt32>(x[1], k0);
                bitwise_xor_tile<DataFormat::UInt32>(x[3], k1);
                const uint32_t low0 = x[0];
                x[0] = x[1];
                x[1] = x[2];
                x[2] = x[3];
                x[3] = low0;
            }
            if constexpr (width < 32) {
                bitwise_and_tile_init();
                bitwise_and_tile<DataFormat::UInt32>(x[0], (1u << width) - 1);
            }
            out_dst = x[0];
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
