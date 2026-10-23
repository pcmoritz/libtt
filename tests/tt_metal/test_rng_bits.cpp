// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::rng_bits draws StableHLO rng_bit_generator bits on the
// device from a state tensor on the device. Its words must equal host
// Threefry-2x32-20 and Philox-4x32-10 references in row-major order for any
// shape, the new state must advance w1 by one per draw, and a cached
// program must draw new bits from each new state.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/rng_bits/rng_bits.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using ttnn::experimental::RngAlgorithm;

uint32_t rotate_left(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

// The first word of Threefry-2x32 with 20 rounds of counter (c0, c1) under key (k0, 0).
uint32_t threefry(uint32_t k0, uint32_t c0, uint32_t c1) {
    static constexpr int kRotations[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
    const uint32_t ks[3] = {k0, 0, k0 ^ 0x1BD11BDAu};
    uint32_t x0 = c0 + ks[0], x1 = c1 + ks[1];
    for (int i = 0; i < 5; ++i) {
        for (int r : kRotations[i % 2]) {
            x0 += x1;
            x1 = rotate_left(x1, r) ^ x0;
        }
        x0 += ks[(i + 1) % 3];
        x1 += ks[(i + 2) % 3] + i + 1;
    }
    return x0;
}

// The first word of Philox-4x32 with 10 rounds of counter c under key (k0, 0).
uint32_t philox(uint32_t k0, std::array<uint32_t, 4> c) {
    uint32_t k1 = 0;
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        const uint64_t p0 = uint64_t{0xD2511F53u} * c[0];
        const uint64_t p1 = uint64_t{0xCD9E8D57u} * c[2];
        c = {static_cast<uint32_t>(p1 >> 32) ^ c[1] ^ k0, static_cast<uint32_t>(p1),
             static_cast<uint32_t>(p0 >> 32) ^ c[3] ^ k1, static_cast<uint32_t>(p0)};
    }
    return c[0];
}

std::vector<uint32_t> expected(
    RngAlgorithm algorithm, const std::vector<uint32_t>& state, size_t elements) {
    std::vector<uint32_t> words(elements);
    for (size_t i = 0; i < elements; ++i) {
        const uint32_t word =
            algorithm == RngAlgorithm::Philox
                ? philox(state[0], {static_cast<uint32_t>(i), state[1], state.size() > 2 ? state[2] : 0u, 0u})
                : threefry(state[0], static_cast<uint32_t>(i), state[1]);
        words[i] = word;
    }
    return words;
}

TEST(RngBitsReference, KnownAnswers) {
    // Random123's known-answer vectors (first words; the key's second word is 0).
    EXPECT_EQ(threefry(0, 0, 0), 0x6B200159u);
    EXPECT_EQ(philox(0, {0, 0, 0, 0}), 0x6627E8D5u);
}

struct Case {
    std::vector<uint32_t> shape;
    RngAlgorithm algorithm;
    Layout state_layout = Layout::ROW_MAJOR;
    std::vector<uint32_t> state = {0xDEADBEEFu, 3};
};

std::string case_name(const ::testing::TestParamInfo<Case>& info) {
    std::string name = info.param.algorithm == RngAlgorithm::Philox ? "Philox" : "Threefry";
    for (uint32_t d : info.param.shape) {
        name += "_" + std::to_string(d);
    }
    name += info.param.state_layout == Layout::TILE ? "_tile" : "_rm";
    name += "_s" + std::to_string(info.param.state.size());
    return name;
}

class RngBitsTest : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor state_tensor(const std::vector<uint32_t>& words, Layout layout) {
        const TensorSpec spec(
            ttnn::Shape({static_cast<uint32_t>(words.size())}),
            TensorLayout(DataType::UINT32, PageConfig(layout), MemoryConfig{}));
        return ttnn::Tensor::from_vector(words, spec).to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

size_t volume(const std::vector<uint32_t>& shape) {
    size_t v = 1;
    for (uint32_t d : shape) {
        v *= d;
    }
    return v;
}

TEST_P(RngBitsTest, MatchesReference) {
    const Case& c = GetParam();
    const auto outputs =
        ttnn::experimental::rng_bits(state_tensor(c.state, c.state_layout), ttnn::Shape(c.shape), c.algorithm);
    ASSERT_EQ(outputs.size(), 2u);
    EXPECT_EQ(outputs[0].logical_shape(), ttnn::Shape(c.shape));
    EXPECT_EQ(outputs[0].dtype(), DataType::UINT32);
    const auto bits = outputs[0].cpu().to_vector<uint32_t>();
    const auto want = expected(c.algorithm, c.state, volume(c.shape));
    ASSERT_EQ(bits.size(), want.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (bits[i] != want[i] && mismatches++ < 8) {
            ADD_FAILURE() << "element " << i << ": " << bits[i] << " vs " << want[i];
        }
    }
    EXPECT_EQ(mismatches, 0u) << "of " << want.size();

    auto new_state = c.state;
    new_state[1] += 1;
    EXPECT_EQ(outputs[1].cpu().to_vector<uint32_t>(), new_state);
}

INSTANTIATE_TEST_SUITE_P(
    RngBits,
    RngBitsTest,
    ::testing::Values(
        Case{{5, 7}, RngAlgorithm::Threefry},
        Case{{5, 7}, RngAlgorithm::Philox},
        Case{{1}, RngAlgorithm::Threefry},
        Case{{1}, RngAlgorithm::Philox},
        Case{{33}, RngAlgorithm::Threefry},
        Case{{3, 5, 7}, RngAlgorithm::Threefry},
        Case{{2, 1, 33}, RngAlgorithm::Philox},
        Case{{2, 40, 70}, RngAlgorithm::Threefry},
        Case{{2, 40, 70}, RngAlgorithm::Philox},
        Case{{64, 128}, RngAlgorithm::Threefry},
        Case{{64, 128}, RngAlgorithm::Philox},
        Case{{5, 7}, RngAlgorithm::Threefry, Layout::TILE},
        Case{{9}, RngAlgorithm::Philox, Layout::ROW_MAJOR, {0xDEADBEEFu, 3, 5}},
        Case{{9}, RngAlgorithm::Philox, Layout::TILE, {0xDEADBEEFu, 3, 5}},
        Case{{64}, RngAlgorithm::Threefry, Layout::ROW_MAJOR, {0xFFFFFFFFu, 0xFFFFFFFFu}},
        Case{{1000, 999}, RngAlgorithm::Threefry},
        Case{{1000, 999}, RngAlgorithm::Philox},
        Case{{4096, 4096}, RngAlgorithm::Threefry},
        Case{{4096, 4096}, RngAlgorithm::Philox}),
    case_name);

// A cached program reads each call's state on the device: chaining calls
// through the new states draws the reference sequence, from one program.
TEST_F(RngBitsTest, CachedProgramFollowsTheState) {
    for (const RngAlgorithm algorithm : {RngAlgorithm::Threefry, RngAlgorithm::Philox}) {
        std::vector<uint32_t> words = {11, 0};
        ttnn::Tensor state = state_tensor(words, Layout::ROW_MAJOR);
        const size_t entries_before = mesh_device_->num_program_cache_entries();
        for (int call = 0; call < 4; ++call) {
            auto outputs = ttnn::experimental::rng_bits(state, ttnn::Shape({3, 50}), algorithm);
            EXPECT_EQ(outputs[0].cpu().to_vector<uint32_t>(), expected(algorithm, words, 150)) << "call " << call;
            words[1] += 1;
            state = outputs[1];
        }
        EXPECT_EQ(mesh_device_->num_program_cache_entries(), entries_before + 1);
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
