// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::l2_clip: v = input[:, offset : offset + width] * scale,
// output = v * min(1, threshold / sqrt(sum_group(v^2) + eps^2)) * row_scale,
// against a double reference: within a BF16 ulp for a BF16 output and 1e-5
// relative for FP32. Cases are Inkling-2-Mini's per-chip decode clips at
// TP4 (its queries' six 128-wide heads and its relative projections' six
// 16-wide ones, read from the [tokens, 2176] projection, and its keys' two
// heads), several row tiles and FP32 input; thresholds clip about half the
// heads.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/l2_clip/l2_clip.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct ClipCase {
    uint32_t rows;
    uint32_t columns;
    uint32_t offset;
    uint32_t width;
    uint32_t group;
    bool scale;
    bool row_scale;
    DataType in_dtype;
    DataType out_dtype;
};

class L2ClipTest : public ::testing::Test {
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

    size_t check(const ClipCase& k, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> x(k.rows * k.columns), scale(k.width), row_scale(k.rows);
        for (float& v : x) {
            v = normal(rng);
        }
        for (float& v : scale) {
            v = 1.0f + 0.3f * normal(rng);
        }
        for (float& v : row_scale) {
            v = 1.0f + 0.2f * std::fabs(normal(rng));
        }
        // About the groups' typical norm, so that some clip and some do not.
        const float threshold = std::sqrt(static_cast<float>(k.group)) * 1.05f;
        constexpr float eps = 1e-6f;
        const auto input = device_tensor(x, ttnn::Shape({k.rows, k.columns}), k.in_dtype);
        std::optional<ttnn::Tensor> scale_tensor, row_scale_tensor;
        if (k.scale) {
            scale_tensor = device_tensor(scale, ttnn::Shape({k.width}), DataType::FLOAT32);
        }
        if (k.row_scale) {
            row_scale_tensor = device_tensor(row_scale, ttnn::Shape({k.rows, 1}), DataType::FLOAT32);
        }
        const auto out = ttnn::experimental::l2_clip(
            input, k.group, threshold, eps, k.offset, k.width, scale_tensor, row_scale_tensor, k.out_dtype);
        EXPECT_EQ(out.logical_shape(), ttnn::Shape({k.rows, k.width}));
        const auto got = out.cpu().to_vector<float>();
        size_t wrong = 0, clipped = 0;
        for (uint32_t r = 0; r < k.rows; ++r) {
            for (uint32_t g = 0; g < k.width; g += k.group) {
                double sum = 0;
                std::vector<double> v(k.group);
                for (uint32_t c = 0; c < k.group; ++c) {
                    v[c] = static_cast<double>(x[r * k.columns + k.offset + g + c]) * (k.scale ? scale[g + c] : 1.0);
                    sum += v[c] * v[c];
                }
                const double f = std::min(1.0, threshold / std::sqrt(sum + double(eps) * eps));
                clipped += f < 1.0;
                for (uint32_t c = 0; c < k.group; ++c) {
                    const double want = v[c] * f * (k.row_scale ? row_scale[r] : 1.0);
                    const double value = got[r * k.width + g + c];
                    const double tolerance = k.out_dtype == DataType::BFLOAT16 ? std::ldexp(std::fabs(want), -7) + 1e-30
                                                                                : std::fabs(want) * 1e-5 + 1e-30;
                    if (!(std::fabs(value - want) <= tolerance) && wrong++ < 8) {
                        ADD_FAILURE() << "rows " << k.rows << " group " << k.group << " [" << r << "][" << g + c
                                      << "] " << value << " != " << want;
                    }
                }
            }
        }
        EXPECT_GT(clipped, 0u);
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(L2ClipTest, MatchesReference) {
    const std::vector<ClipCase> cases = {
        // Queries: heads 0..5 of the projection, scaled per element, tau per token.
        {1, 2176, 0, 768, 128, true, true, DataType::BFLOAT16, DataType::BFLOAT16},
        // Keys after their convolution: two heads, no scales.
        {1, 256, 0, 256, 128, false, false, DataType::BFLOAT16, DataType::BFLOAT16},
        // Relative projections: six 16-wide heads at column 1280.
        {1, 2176, 1280, 96, 16, true, true, DataType::BFLOAT16, DataType::BFLOAT16},
        // Four tokens, several row tiles, FP32 in and out, 64-wide groups.
        {4, 2176, 0, 768, 128, true, true, DataType::BFLOAT16, DataType::BFLOAT16},
        {33, 2176, 1280, 96, 16, true, false, DataType::FLOAT32, DataType::FLOAT32},
        {40, 512, 128, 256, 64, false, true, DataType::FLOAT32, DataType::BFLOAT16},
    };
    uint32_t seed = 1;
    for (const auto& k : cases) {
        EXPECT_EQ(check(k, seed++), 0u) << "case " << seed - 1;
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
