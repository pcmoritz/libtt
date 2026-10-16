// SPDX-License-Identifier: Apache-2.0

// A decode matmul (one tile row of activations) accumulates in FP32 by
// default, but not under a caller's program config: FP32 accumulation halves
// the Dst tiles an output subblock may use, so a 1x8 subblock sized for the
// BF16 default would no longer validate.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class MatmulDecodeProgramConfigTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // A BF16 TILE tensor in DRAM; `values` come back rounded to BF16.
    ttnn::Tensor to_device(std::vector<float>& values, const ttnn::Shape& shape) {
        const TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
        ttnn::Tensor host = ttnn::Tensor::from_vector(values, spec);
        values = host.to_vector<float>();
        return host.to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(MatmulDecodeProgramConfigTest, ExplicitConfigKeepsBf16AccumulationSubblocks) {
    constexpr uint32_t K = 1024;
    constexpr uint32_t N = 256;  // eight output tiles, all on one core
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> small(-4, 4);
    std::vector<float> a(K), b(K * N);
    for (float& v : a) {
        v = small(rng) / 16.0f;
    }
    for (float& v : b) {
        v = small(rng) / 16.0f;
    }
    const ttnn::Tensor a_device = to_device(a, ttnn::Shape({1, K}));
    const ttnn::Tensor b_device = to_device(b, ttnn::Shape({K, N}));

    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig config{
        .compute_with_storage_grid_size = {1, 1},
        .in0_block_w = 2,
        .out_subblock_h = 1,
        .out_subblock_w = 8,
        .out_block_h = 1,
        .out_block_w = 8,
        .per_core_M = 1,
        .per_core_N = 8,
        .fuse_batch = true,
        .fused_activation = std::nullopt,
        .mcast_in0 = true,
    };
    const ttnn::Tensor output = ttnn::matmul(
        a_device, b_device, /*transpose_a=*/false, /*transpose_b=*/false, /*memory_config=*/std::nullopt,
        /*dtype=*/std::nullopt, config);

    const std::vector<float> actual = output.cpu().to_vector<float>();
    ASSERT_EQ(actual.size(), N);
    double error = 0.0, norm = 0.0;
    for (uint32_t n = 0; n < N; ++n) {
        double expected = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
            expected += static_cast<double>(a[k]) * b[k * N + n];
        }
        error += (actual[n] - expected) * (actual[n] - expected);
        norm += expected * expected;
    }
    // BF16 accumulation of these 1024-term sums: about 0.5% relative error.
    EXPECT_LT(std::sqrt(error / norm), 0.02);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
