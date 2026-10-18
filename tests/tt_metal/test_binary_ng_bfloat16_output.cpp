// SPDX-License-Identifier: Apache-2.0

// A binary op on FP32 operands with a BF16 output dtype rounds its FP32 result
// to BF16 in the op itself. It must produce exactly the BF16 tensor that a
// separate typecast of the FP32 result produces, for every broadcast shape.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class BinaryNgBFloat16OutputTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // A TILE tensor in DRAM of normal values offset by `offset`.
    ttnn::Tensor random_tensor(const ttnn::Shape& shape, DataType dtype, uint32_t seed, float offset = 0.0f) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> values(shape.volume());
        for (float& v : values) {
            v = normal(rng) + offset;
        }
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // Number of elements whose BF16 bits differ.
    static size_t mismatches(const ttnn::Tensor& actual, const ttnn::Tensor& expected) {
        const std::vector<bfloat16> a = actual.cpu().to_vector<bfloat16>();
        const std::vector<bfloat16> e = expected.cpu().to_vector<bfloat16>();
        EXPECT_EQ(a.size(), e.size());
        size_t count = 0;
        for (size_t i = 0; i < a.size() && i < e.size(); ++i) {
            count += std::memcmp(&a[i], &e[i], sizeof(bfloat16)) != 0;
        }
        return count;
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

using BinaryFn = std::function<ttnn::Tensor(const ttnn::Tensor&, const ttnn::Tensor&, std::optional<const DataType>)>;

TEST_F(BinaryNgBFloat16OutputTest, MatchesTypecastOfFp32Result) {
    const std::vector<std::pair<const char*, BinaryFn>> ops = {
        {"add", [](const auto& a, const auto& b, auto dtype) { return ttnn::add(a, b, dtype); }},
        {"subtract", [](const auto& a, const auto& b, auto dtype) { return ttnn::subtract(a, b, dtype); }},
        {"multiply", [](const auto& a, const auto& b, auto dtype) { return ttnn::multiply(a, b, dtype); }},
        {"divide", [](const auto& a, const auto& b, auto dtype) { return ttnn::divide(a, b, dtype); }},
    };
    // No broadcast, then b broadcast along rows, along columns and as a scalar.
    const std::vector<std::pair<ttnn::Shape, ttnn::Shape>> shapes = {
        {ttnn::Shape({4, 2560}), ttnn::Shape({4, 2560})},
        {ttnn::Shape({64, 96}), ttnn::Shape({1, 96})},
        {ttnn::Shape({64, 96}), ttnn::Shape({64, 1})},
        {ttnn::Shape({64, 96}), ttnn::Shape({1, 1})},
    };
    uint32_t seed = 0;
    for (const auto& [name, op] : ops) {
        for (const auto& [a_shape, b_shape] : shapes) {
            const ttnn::Tensor a = random_tensor(a_shape, DataType::FLOAT32, ++seed);
            const ttnn::Tensor b = random_tensor(b_shape, DataType::FLOAT32, ++seed, /*offset=*/3.0f);
            const ttnn::Tensor narrowed = op(a, b, DataType::BFLOAT16);
            ASSERT_EQ(narrowed.dtype(), DataType::BFLOAT16);
            const ttnn::Tensor reference = ttnn::typecast(op(a, b, DataType::FLOAT32), DataType::BFLOAT16);
            EXPECT_EQ(mismatches(narrowed, reference), 0u) << name << " a " << a_shape << " b " << b_shape;
        }
    }
}

TEST_F(BinaryNgBFloat16OutputTest, ScalarOperand) {
    const ttnn::Tensor a = random_tensor(ttnn::Shape({32, 320}), DataType::FLOAT32, 7);
    const ttnn::Tensor narrowed = ttnn::multiply(a, 1.7f, DataType::BFLOAT16);
    const ttnn::Tensor reference = ttnn::typecast(ttnn::multiply(a, 1.7f), DataType::BFLOAT16);
    EXPECT_EQ(mismatches(narrowed, reference), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
