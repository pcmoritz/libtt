// SPDX-License-Identifier: Apache-2.0

// Interleaved unary, binary and ternary elementwise programs run only on the
// cores with output tiles, and their program hash includes the operand
// shapes. Programs for differently sized operands, including repeated calls
// that hit the program cache, must still produce the reference results.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class EltwiseWorkingCoresTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // A BF16 TILE tensor in DRAM of random values; `values` come back rounded.
    ttnn::Tensor random_tensor(std::vector<float>& values, const ttnn::Shape& shape, uint32_t seed, bool binary = false) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        values.resize(shape.volume());
        for (float& v : values) {
            v = binary ? static_cast<float>(normal(rng) > 0.0f) : normal(rng);
        }
        const TensorSpec spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
        ttnn::Tensor host = ttnn::Tensor::from_vector(values, spec);
        values = host.to_vector<float>();
        return host.to_device(mesh_device_.get());
    }

    // Largest absolute difference from `expected(i)` over the output.
    static double max_error(const ttnn::Tensor& output, const std::function<double(size_t)>& expected) {
        const std::vector<float> actual = output.cpu().to_vector<float>();
        double worst = 0.0;
        for (size_t i = 0; i < actual.size(); ++i) {
            worst = std::max(worst, std::abs(actual[i] - expected(i)));
        }
        return worst;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// One tile, a row of eleven tiles, more tiles than cores, then the first
// shape again from the program cache.
const std::vector<ttnn::Shape> kShapes = {
    ttnn::Shape({1, 32}), ttnn::Shape({1, 324}), ttnn::Shape({160, 2560}), ttnn::Shape({1, 32})};

TEST_F(EltwiseWorkingCoresTest, UnaryAcrossShapes) {
    uint32_t seed = 0;
    for (const auto& shape : kShapes) {
        std::vector<float> x;
        const ttnn::Tensor input = random_tensor(x, shape, ++seed);
        const ttnn::Tensor output = ttnn::sigmoid(input);
        EXPECT_LT(max_error(output, [&](size_t i) { return 1.0 / (1.0 + std::exp(-x[i])); }), 0.02)
            << "shape " << shape;
    }
}

TEST_F(EltwiseWorkingCoresTest, BinaryAcrossShapesAndBroadcasts) {
    struct Case {
        ttnn::Shape a, b;
    };
    const std::vector<Case> cases = {
        {ttnn::Shape({1, 324}), ttnn::Shape({1, 324})},
        {ttnn::Shape({160, 2560}), ttnn::Shape({1, 2560})},
        {ttnn::Shape({64, 96}), ttnn::Shape({64, 1})},
        {ttnn::Shape({1, 324}), ttnn::Shape({1, 324})},
    };
    uint32_t seed = 100;
    for (const auto& c : cases) {
        std::vector<float> a, b;
        const ttnn::Tensor lhs = random_tensor(a, c.a, ++seed);
        const ttnn::Tensor rhs = random_tensor(b, c.b, ++seed);
        const ttnn::Tensor output = ttnn::multiply(lhs, rhs);
        const uint32_t rows = c.a[0], cols = c.a[1];
        EXPECT_LT(
            max_error(
                output,
                [&](size_t i) {
                    const size_t r = i / cols, k = i % cols;
                    const size_t j = (c.b[0] == 1 ? 0 : r) * c.b[1] + (c.b[1] == 1 ? 0 : k);
                    return static_cast<double>(a[i]) * b[j];
                }),
            0.05)
            << "a " << c.a << " b " << c.b << " rows " << rows;
    }
}

TEST_F(EltwiseWorkingCoresTest, TernaryAcrossShapes) {
    uint32_t seed = 200;
    for (const auto& shape : kShapes) {
        std::vector<float> p, t, f;
        const ttnn::Tensor predicate = random_tensor(p, shape, ++seed, /*binary=*/true);
        const ttnn::Tensor on_true = random_tensor(t, shape, ++seed);
        const ttnn::Tensor on_false = random_tensor(f, shape, ++seed);
        const ttnn::Tensor output = ttnn::where(predicate, on_true, on_false);
        EXPECT_EQ(max_error(output, [&](size_t i) { return p[i] != 0.0f ? t[i] : f[i]; }), 0.0) << "shape " << shape;
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
