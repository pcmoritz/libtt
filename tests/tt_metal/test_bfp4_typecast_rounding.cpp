// SPDX-License-Identifier: Apache-2.0
//
// The packer truncates to Bfp4_b, so a device typecast of bfloat16 to Bfp4_b
// first rounds each value on the SFPU to nearest even against its face row's
// shared exponent (bfp4_typecast_rounding.patch). The device must then produce
// the same Bfp4_b words as the host packer (pack_as_bfp4_tiles).

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class Bfp4TypecastRoundingTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// The words of a host Bfp4_b tensor.
std::vector<uint32_t> words(const ttnn::Tensor& tensor) {
    std::vector<uint32_t> out;
    tensor.host_storage().host_tensor().buffer().apply([&](const HostBuffer& shard) {
        const auto view = shard.view_as<const uint32_t>();
        out.assign(view.begin(), view.end());
    });
    return out;
}

// Rows of 16 values (one face row, which shares an exponent), cycling through
// weight-like normals, normals at a random scale per row, and random bit
// patterns (many ties) with zeros and denormals mixed in. Exponents stay in
// [3, 254]: Inf, NaN and shared exponents below 3 are out of scope.
std::vector<float> values(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> scale(-100, 100);
    std::uniform_int_distribution<uint32_t> bits(0, 0xffff);
    std::uniform_int_distribution<uint32_t> exponent(3, 254);
    std::vector<float> out(count);
    for (size_t row = 0; row < count / 16; ++row) {
        const float row_scale = std::ldexp(1.0f, scale(rng));
        const uint32_t row_exponent = exponent(rng);
        for (size_t i = row * 16; i < row * 16 + 16; ++i) {
            float value = 0.0f;
            switch (row % 3) {
                case 0: value = normal(rng) * 0.02f; break;
                case 1: value = normal(rng) * row_scale; break;
                default: {
                    // Exponents within 6 of the row's, so values keep bits at Bfp4_b precision.
                    const uint32_t b = bits(rng);
                    const uint32_t e = b % 11 == 0 ? 0 : std::max<int>(3, static_cast<int>(row_exponent) - (b % 7));
                    value = std::bit_cast<float>(((b & 0x8000u) | (e << 7) | (b & 0x7fu)) << 16);
                }
            }
            out[i] = static_cast<float>(bfloat16(value));
        }
    }
    return out;
}

TEST_F(Bfp4TypecastRoundingTest, MatchesHostPacker) {
    const ttnn::Shape shape({1, 2, 512, 1024});
    const TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
    const ttnn::Tensor host = ttnn::Tensor::from_vector(values(shape.volume(), 1), spec);

    const std::vector<uint32_t> expected = words(ttnn::to_dtype(host, DataType::BFLOAT4_B));
    const std::vector<uint32_t> actual =
        words(ttnn::typecast(host.to_device(mesh_device_.get()), DataType::BFLOAT4_B).cpu());

    ASSERT_EQ(actual.size(), expected.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        mismatches += actual[i] != expected[i];
    }
    EXPECT_EQ(mismatches, 0u) << "of " << expected.size() << " words";
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
