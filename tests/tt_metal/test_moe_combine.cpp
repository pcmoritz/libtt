// SPDX-License-Identifier: Apache-2.0

// ttnn::experimental::moe_combine: each token's routing-weighted sum of its
// slots (row j * T + i of moe_compute's row-major [k * T, H] output, some zero
// as for experts other devices own), against a double reference within the
// output's rounding and the matrix unit's FP32 accumulation: BF16 and FP32
// weights, slot counts below a tile's rows, BF16 and FP32 outputs.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/moe_combine/moe_combine.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class MoeCombineTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor device_tensor(std::vector<float>& values, const ttnn::Shape& shape, DataType dtype, Layout layout) {
        if (dtype == DataType::BFLOAT16) {
            for (float& v : values) {
                v = static_cast<float>(bfloat16(v));
            }
        }
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(layout), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    size_t check(uint32_t T, uint32_t k, uint32_t H, DataType weight_dtype, DataType out_dtype, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> slots(k * T * H), weights(T * k);
        for (uint32_t r = 0; r < k * T; ++r) {
            // Every third slot is another device's expert: zero.
            for (uint32_t c = 0; c < H; ++c) {
                slots[r * H + c] = r % 3 == 1 ? 0.0f : 4.0f * normal(rng);
            }
        }
        for (float& v : weights) {
            v = std::fabs(normal(rng)) / k;
        }
        const auto out = ttnn::experimental::moe_combine(
            device_tensor(slots, ttnn::Shape({k * T, H}), DataType::BFLOAT16, Layout::ROW_MAJOR),
            device_tensor(weights, ttnn::Shape({T, k}), weight_dtype, Layout::TILE),
            out_dtype);
        EXPECT_EQ(out.logical_shape(), ttnn::Shape({T, H}));
        const auto got = out.cpu().to_vector<float>();
        size_t wrong = 0;
        for (uint32_t i = 0; i < T; ++i) {
            for (uint32_t c = 0; c < H; ++c) {
                double want = 0, scale = 0;
                for (uint32_t j = 0; j < k; ++j) {
                    want += double(weights[i * k + j]) * slots[(j * T + i) * H + c];
                    scale += std::fabs(double(weights[i * k + j]) * slots[(j * T + i) * H + c]);
                }
                // The output's rounding, and 2^-10 of the products' magnitude: the
                // matrix unit's FP32 accumulation (as for the matmul it replaces) and
                // FP32 weights read as TF32.
                const double tolerance =
                    (out_dtype == DataType::BFLOAT16 ? std::ldexp(std::fabs(want), -8) : 0.0) +
                    std::ldexp(scale, -10) + 1e-30;
                const double value = got[i * H + c];
                if (!(std::fabs(value - want) <= tolerance) && wrong++ < 8) {
                    ADD_FAILURE() << "T " << T << " k " << k << " [" << i << "][" << c << "] " << value
                                  << " != " << want;
                }
            }
        }
        return wrong;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(MoeCombineTest, MatchesReference) {
    // Inkling-2-Mini's decode on four chips: 4 tokens, 8 experts, H 2304.
    EXPECT_EQ(check(4, 8, 2304, DataType::BFLOAT16, DataType::BFLOAT16, 1), 0u);
    EXPECT_EQ(check(1, 8, 2304, DataType::BFLOAT16, DataType::BFLOAT16, 2), 0u);
    EXPECT_EQ(check(3, 8, 256, DataType::FLOAT32, DataType::BFLOAT16, 3), 0u);
    EXPECT_EQ(check(2, 16, 64, DataType::BFLOAT16, DataType::FLOAT32, 4), 0u);
    // More tile columns than cores.
    EXPECT_EQ(check(4, 8, 32 * 150, DataType::BFLOAT16, DataType::BFLOAT16, 5), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
