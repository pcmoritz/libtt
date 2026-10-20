// SPDX-License-Identifier: Apache-2.0

// ttnn's batch_norm normalization (prim::batch_norm) in float32, with the
// compute config libtt's runtime passes it (HiFi4, FP32 DST, exact math): its
// output must match a float64 reference to float32 accuracy, also for channels
// of one element, whose variance is 0.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/normalization/batch_norm/device/batch_norm_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct Case {
    uint32_t N, C, H, W;
    bool affine;
};

class BatchNormFloat32Test : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor to_device(const std::vector<float>& values, const std::vector<uint32_t>& shape) {
        const TensorSpec spec(
            ttnn::Shape(shape), TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), MemoryConfig{}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_P(BatchNormFloat32Test, NormalizesExactly) {
    const auto [N, C, H, W, affine] = GetParam();
    const float eps = 1e-5f;
    std::mt19937 rng(2);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(N * C * H * W), mean(C), var(C), weight(C), bias(C);
    for (float& v : x) {
        v = 1.0f + 3.0f * normal(rng);
    }
    for (uint32_t c = 0; c < C; ++c) {
        // One element per channel: the batch statistics are the element and 0.
        mean[c] = N * H * W == 1 ? x[c] : normal(rng);
        var[c] = N * H * W == 1 ? 0.0f : 1.0f + std::abs(normal(rng));
        weight[c] = normal(rng);
        bias[c] = normal(rng);
    }
    std::optional<ttnn::Tensor> weight_tensor, bias_tensor;
    if (affine) {
        weight_tensor = to_device(weight, {1, C, 1, 1});
        bias_tensor = to_device(bias, {1, C, 1, 1});
    }
    const ttnn::DeviceComputeKernelConfig config = ttnn::ComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = true};
    const auto got = ttnn::prim::batch_norm(
                         to_device(x, {N, C, H, W}),
                         to_device(mean, {1, C, 1, 1}),
                         to_device(var, {1, C, 1, 1}),
                         eps,
                         weight_tensor,
                         bias_tensor,
                         std::nullopt,
                         std::nullopt,
                         config)
                         .cpu()
                         .to_vector<float>();

    ASSERT_EQ(got.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ASSERT_TRUE(std::isfinite(got[i])) << "element " << i << ": " << got[i];
        const uint32_t c = (i / (H * W)) % C;
        double want = (double(x[i]) - mean[c]) / std::sqrt(double(var[c]) + eps);
        if (affine) {
            want = want * weight[c] + bias[c];
        }
        ASSERT_NEAR(got[i], want, 2e-6 * (1.0 + std::abs(want))) << "element " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    BatchNormFloat32,
    BatchNormFloat32Test,
    ::testing::Values(
        Case{4, 5, 6, 7, false},
        Case{4, 5, 6, 7, true},
        Case{2, 3, 32, 64, true},
        Case{1, 3, 1, 1, true}));

}  // namespace
}  // namespace tt::tt_metal::distributed::test
