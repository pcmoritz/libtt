// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::rms_norm_gated: output = x * rsqrt(mean(x^2) + eps) *
// weight * silu(gate) over each group of the weight's D columns, in FP32,
// against a double reference: within a BF16 ulp for a BF16 output and 1e-5
// relative for FP32. Shapes cover one and several tile rows, partial tiles,
// 3-D inputs, D of 32 to 128, several groups per row (Qwen3.8's [tokens,
// 12 heads * 128] per chip) and a gate read from a wider tensor at an offset
// (its qkvz projection's z columns).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/rms_norm_gated/rms_norm_gated.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class RMSNormGatedTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor device_tensor(std::vector<float>& values, const ttnn::Shape& shape, DataType dtype) {
        if (dtype == DataType::BFLOAT16) {
            for (float& v : values) {
                v = static_cast<float>(bfloat16(v));
            }
        }
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // Returns the number of elements off the reference.
    // A gate of gate_width columns read from gate_offset; a group of `group`
    // columns (the last dimension by default).
    size_t check(
        const ttnn::Shape& shape,
        DataType in_dtype,
        DataType out_dtype,
        uint32_t seed,
        uint32_t group = 0,
        uint32_t gate_width = 0,
        uint32_t gate_offset = 0) {
        const uint32_t width = shape[-1];
        group = group ? group : width;
        gate_width = gate_width ? gate_width : width;
        const size_t rows = shape.volume() / width;
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> x(shape.volume()), gate(rows * gate_width), weight(group);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = normal(rng) * std::ldexp(1.0f, static_cast<int>(i / group % 9) - 4);
        }
        for (float& g : gate) {
            g = 2.0f * normal(rng);
        }
        for (float& w : weight) {
            w = 1.0f + 0.2f * normal(rng);
        }
        auto gate_dims = shape;
        gate_dims[-1] = gate_width;
        const auto input = device_tensor(x, shape, in_dtype);
        const auto gate_tensor = device_tensor(gate, gate_dims, in_dtype);
        const auto weight_tensor = device_tensor(weight, ttnn::Shape({group}), DataType::BFLOAT16);
        constexpr float eps = 1e-6f;
        const auto out =
            ttnn::experimental::rms_norm_gated(input, gate_tensor, weight_tensor, eps, gate_offset, out_dtype);
        EXPECT_EQ(out.logical_shape(), shape);
        EXPECT_EQ(out.dtype(), out_dtype);
        const auto got = out.cpu().to_vector<float>();

        size_t wrong = 0;
        for (size_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < width; ++c) {
                const uint32_t g0 = c / group * group;
                double sum = 0;
                for (uint32_t k = g0; k < g0 + group; ++k) {
                    sum += static_cast<double>(x[r * width + k]) * x[r * width + k];
                }
                const double rstd = 1.0 / std::sqrt(sum / group + eps);
                const size_t i = r * width + c;
                const double z = gate[r * gate_width + gate_offset + c];
                const double want = x[i] * rstd * weight[c - g0] * (z / (1.0 + std::exp(-z)));
                const double tolerance =
                    out_dtype == DataType::BFLOAT16 ? std::ldexp(std::fabs(want), -7) + 1e-30 : std::fabs(want) * 1e-5 + 1e-30;
                if (!(std::fabs(got[i] - want) <= tolerance) && wrong++ < 8) {
                    ADD_FAILURE() << shape << " " << in_dtype << "->" << out_dtype << " [" << r << "][" << c << "] "
                                  << got[i] << " != " << want;
                }
            }
        }
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(RMSNormGatedTest, MatchesReference) {
    const std::vector<ttnn::Shape> shapes = {
        ttnn::Shape({12, 128}),
        ttnn::Shape({1, 12, 128}),
        ttnn::Shape({4, 12, 128}),
        ttnn::Shape({48, 128}),
        ttnn::Shape({70, 64}),
        ttnn::Shape({5, 32}),
        ttnn::Shape({32, 96})};
    uint32_t seed = 1;
    for (const auto& shape : shapes) {
        EXPECT_EQ(check(shape, DataType::BFLOAT16, DataType::BFLOAT16, seed++), 0u) << shape;
    }
    EXPECT_EQ(check(ttnn::Shape({12, 128}), DataType::FLOAT32, DataType::BFLOAT16, seed++), 0u);
    EXPECT_EQ(check(ttnn::Shape({12, 128}), DataType::BFLOAT16, DataType::FLOAT32, seed++), 0u);
    EXPECT_EQ(check(ttnn::Shape({3, 12, 128}), DataType::FLOAT32, DataType::FLOAT32, seed++), 0u);
}

TEST_F(RMSNormGatedTest, GroupsAndGateOffset) {
    // Qwen3.8's decode layout: [tokens, 12 heads * 128], the gate the z
    // columns 2560 .. 4095 of the [tokens, 4096] projection; FP32 input.
    uint32_t seed = 100;
    for (uint32_t tokens : {1u, 4u, 33u}) {
        EXPECT_EQ(
            check(ttnn::Shape({tokens, 1536}), DataType::FLOAT32, DataType::BFLOAT16, seed++, 128, 4096, 2560), 0u)
            << tokens;
    }
    EXPECT_EQ(check(ttnn::Shape({2, 512}), DataType::BFLOAT16, DataType::BFLOAT16, seed++, 64), 0u);
    EXPECT_EQ(check(ttnn::Shape({5, 256}), DataType::BFLOAT16, DataType::FLOAT32, seed++, 128, 384, 128), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
