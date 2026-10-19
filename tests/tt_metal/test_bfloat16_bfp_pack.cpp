// SPDX-License-Identifier: Apache-2.0
//
// Tiled bfloat16 is packed as Bfp8_b and Bfp4_b by a table-driven fast path
// (fast_bfloat16_bfp_pack.patch). It must produce the same words as the
// generic packer, which converts every value through float.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt_stl/span.hpp>

namespace {

constexpr size_t kTileElements = 32 * 32;

// Rows of 16 values (one face row, which shares an exponent): random bit
// patterns cover every exponent, zeros, subnormals, infinities and NaNs;
// normally distributed rows at random scales look like weights.
std::vector<bfloat16> values(size_t num_tiles, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> bits(0, 0xffff);
    std::uniform_int_distribution<int> scale(-60, 60);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<bfloat16> out(num_tiles * kTileElements);
    for (size_t row = 0; row < out.size() / 16; ++row) {
        const bool random_bits = row % 2 == 0;
        const float row_scale = std::ldexp(1.0f, scale(rng));
        for (size_t i = row * 16; i < row * 16 + 16; ++i) {
            out[i] = random_bits ? std::bit_cast<bfloat16>(static_cast<uint16_t>(bits(rng)))
                                 : bfloat16(normal(rng) * row_scale);
        }
    }
    return out;
}

std::vector<float> as_float(const std::vector<bfloat16>& in) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = std::bit_cast<float>(static_cast<uint32_t>(std::bit_cast<uint16_t>(in[i])) << 16);
    }
    return out;
}

class BfloatBfpPackTest : public ::testing::TestWithParam<size_t> {};

TEST_P(BfloatBfpPackTest, MatchesGenericPacker) {
    // Enough tiles for the fast path to use several threads at the larger size.
    const std::vector<bfloat16> bf16 = values(GetParam(), static_cast<uint32_t>(GetParam()));
    const std::vector<float> f32 = as_float(bf16);
    const ttsl::Span<const bfloat16> bf16_span(bf16.data(), bf16.size());
    const ttsl::Span<const float> f32_span(f32.data(), f32.size());

    EXPECT_EQ(
        pack_as_bfp4_tiles<bfloat16>(bf16_span, /*row_major_input=*/false, /*is_exp_a=*/false),
        pack_as_bfp4_tiles<float>(f32_span, /*row_major_input=*/false, /*is_exp_a=*/false));
    EXPECT_EQ(
        pack_as_bfp8_tiles<bfloat16>(bf16_span, /*row_major_input=*/false, /*is_exp_a=*/false),
        pack_as_bfp8_tiles<float>(f32_span, /*row_major_input=*/false, /*is_exp_a=*/false));
}

INSTANTIATE_TEST_SUITE_P(Tiles, BfloatBfpPackTest, ::testing::Values(size_t{3}, size_t{2048}));

}  // namespace
