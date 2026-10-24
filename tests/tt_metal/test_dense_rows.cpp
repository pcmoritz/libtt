// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
// ttnn::experimental::dense_rows moves tiled tensors between [..., H, W] and
// [..., H, W / 32, 32] without changing their row-major data.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/dense_rows/dense_rows.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using ttnn::experimental::DenseRowsDirection;

class DenseRowsTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    template <typename T>
    ttnn::Tensor to_device(const std::vector<T>& values, const ttnn::Shape& shape, DataType dtype) {
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), MemoryConfig{}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    template <typename T>
    void expect_move(const ttnn::Shape& from, const ttnn::Shape& to, DataType dtype, DenseRowsDirection direction) {
        std::vector<T> values(from.volume());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<T>(static_cast<int64_t>(i % 100003) - 50001);
        }
        const ttnn::Tensor input = to_device(values, from, dtype);
        ASSERT_EQ(ttnn::experimental::dense_rows_direction(input, to), direction) << from << " -> " << to;
        const ttnn::Tensor output = ttnn::experimental::dense_rows(input, direction);
        ASSERT_EQ(output.logical_shape(), to);
        const std::vector<T> actual = output.cpu().to_vector<T>();
        ASSERT_EQ(actual.size(), values.size());
        size_t mismatches = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            mismatches += actual[i] != values[i];
        }
        EXPECT_EQ(mismatches, 0u) << from << " -> " << to << " " << dtype;

        // Back again, and through ttnn::reshape, which must agree.
        const auto back = direction == DenseRowsDirection::Pack ? DenseRowsDirection::Unpack : DenseRowsDirection::Pack;
        const std::vector<T> round = ttnn::experimental::dense_rows(output, back).cpu().to_vector<T>();
        EXPECT_EQ(round, values) << to << " -> " << from;
        EXPECT_EQ(ttnn::reshape(input, to).cpu().to_vector<T>(), values) << "reshape " << from << " -> " << to;
    }

    template <typename T>
    void expect_pack(const ttnn::Shape& from, DataType dtype) {
        ttsl::SmallVector<uint32_t> dims(from.cbegin(), from.cend());
        dims.back() /= 32;
        dims.push_back(32);
        expect_move<T>(from, ttnn::Shape(dims), dtype, DenseRowsDirection::Pack);
        expect_move<T>(ttnn::Shape(dims), from, dtype, DenseRowsDirection::Unpack);
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(DenseRowsTest, Directions) {
    const ttnn::Tensor x = to_device(std::vector<float>(2 * 64), ttnn::Shape({2, 64}), DataType::FLOAT32);
    EXPECT_EQ(ttnn::experimental::dense_rows_direction(x, ttnn::Shape({2, 2, 32})), DenseRowsDirection::Pack);
    EXPECT_EQ(ttnn::experimental::dense_rows_direction(x, ttnn::Shape({4, 32})), std::nullopt);
    EXPECT_EQ(ttnn::experimental::dense_rows_direction(x, ttnn::Shape({2, 1, 64})), std::nullopt);
    const ttnn::Tensor y = to_device(std::vector<float>(2 * 64), ttnn::Shape({2, 2, 32}), DataType::FLOAT32);
    EXPECT_EQ(ttnn::experimental::dense_rows_direction(y, ttnn::Shape({2, 64})), DenseRowsDirection::Unpack);
    const ttnn::Tensor b = to_device(std::vector<float>(2 * 64), ttnn::Shape({2, 64}), DataType::BFLOAT16);
    EXPECT_EQ(ttnn::experimental::dense_rows_direction(b, ttnn::Shape({2, 2, 32})), std::nullopt);
}

TEST_F(DenseRowsTest, SamplerRows) {
    expect_pack<float>(ttnn::Shape({2, 151936}), DataType::FLOAT32);
    expect_pack<float>(ttnn::Shape({2, 37984}), DataType::FLOAT32);
    expect_pack<int32_t>(ttnn::Shape({2, 37984}), DataType::INT32);
    expect_pack<uint32_t>(ttnn::Shape({2, 151936}), DataType::UINT32);
}

TEST_F(DenseRowsTest, Shapes) {
    expect_pack<float>(ttnn::Shape({1, 32}), DataType::FLOAT32);
    expect_pack<float>(ttnn::Shape({4096}), DataType::FLOAT32);
    expect_pack<float>(ttnn::Shape({1, 1056}), DataType::FLOAT32);
    expect_pack<float>(ttnn::Shape({3, 5, 96}), DataType::FLOAT32);
    expect_pack<float>(ttnn::Shape({40, 320}), DataType::FLOAT32);
    expect_pack<int32_t>(ttnn::Shape({17, 2048}), DataType::INT32);
    expect_pack<float>(ttnn::Shape({64, 2048}), DataType::FLOAT32);
    expect_pack<float>(ttnn::Shape({2, 3, 33, 1024}), DataType::FLOAT32);
}

TEST_F(DenseRowsTest, Timing) {
    // Untraced, so each call includes its dispatch.
    for (uint32_t W : {151936u, 37984u}) {
        const uint32_t N = W / 32;
        const ttnn::Shape from({2, W});
        const ttnn::Shape to({2, N, 32});
        const ttnn::Tensor x = to_device(std::vector<float>(from.volume(), 1.0f), from, DataType::FLOAT32);
        const ttnn::Tensor x3 = to_device(std::vector<float>(from.volume(), 1.0f), ttnn::Shape({1, 2, W}), DataType::FLOAT32);
        const ttnn::Tensor y = ttnn::experimental::dense_rows(x, DenseRowsDirection::Pack);
        auto time = [&](const char* what, auto&& fn) {
            fn();
            Finish(mesh_device_->mesh_command_queue());
            const auto start = std::chrono::steady_clock::now();
            constexpr int reps = 50;
            for (int i = 0; i < reps; ++i) {
                fn();
            }
            Finish(mesh_device_->mesh_command_queue());
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
            std::printf("TIMING %-28s W=%-6u %8.1f us\n", what, W, us / reps);
        };
        time("dense_rows pack", [&] { (void)ttnn::experimental::dense_rows(x, DenseRowsDirection::Pack); });
        time("dense_rows unpack", [&] { (void)ttnn::experimental::dense_rows(y, DenseRowsDirection::Unpack); });
        time("ttnn::reshape pack", [&] { (void)ttnn::reshape(x, to); });
        time("ttnn::reshape unpack", [&] { (void)ttnn::reshape(y, from); });
        time("reshape_view pack", [&] {
            (void)ttnn::prim::reshape_view(
                x3, to, ttnn::Shape({2, (N + 31) / 32 * 32, 32}), x3.memory_config(), false, std::nullopt);
        });
        time("reshape_view unpack", [&] {
            (void)ttnn::prim::reshape_view(
                y, ttnn::Shape({1, 2, W}), ttnn::Shape({1, 32, W}), y.memory_config(), false, std::nullopt);
        });
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
