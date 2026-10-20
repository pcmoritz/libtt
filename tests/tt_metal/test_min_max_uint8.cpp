// SPDX-License-Identifier: Apache-2.0

// ttnn maximum and minimum compare uint8 (as unsigned 32-bit integers in a
// 32-bit DST), with row, column and scalar broadcasts in TILE and ROW_MAJOR
// layout, and so does the tensor clamp, a maximum followed by a minimum.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class MinMaxUint8Test : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    ttnn::Tensor device_tensor(
        const ttnn::Shape& shape, const std::vector<uint8_t>& values, Layout layout = Layout::TILE) {
        const TensorSpec spec(shape, TensorLayout(DataType::UINT8, PageConfig(layout), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

// Every byte, the upper half included, in a different order per seed.
std::vector<uint8_t> bytes(size_t n, size_t seed) {
    std::vector<uint8_t> values(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = static_cast<uint8_t>((i * (37 + 64 * seed) + seed * 7) % 256);
    }
    return values;
}

TEST_F(MinMaxUint8Test, MaximumMinimum) {
    const uint32_t rows = 64, cols = 96;
    const auto a = bytes(rows * cols, 0), b = bytes(rows * cols, 1), row = bytes(cols, 2), col = bytes(rows, 3);
    const std::vector<uint8_t> scalar{200};
    struct Case {
        Layout layout;
        ttnn::Shape shape;
        const std::vector<uint8_t>& values;
        std::function<uint8_t(uint32_t, uint32_t)> at;
    };
    for (const auto& [layout, shape, values, at] :
         {Case{Layout::TILE, ttnn::Shape({rows, cols}), b, [&](uint32_t r, uint32_t c) { return b[r * cols + c]; }},
          Case{Layout::TILE, ttnn::Shape({1, cols}), row, [&](uint32_t, uint32_t c) { return row[c]; }},
          Case{Layout::TILE, ttnn::Shape({rows, 1}), col, [&](uint32_t r, uint32_t) { return col[r]; }},
          Case{Layout::TILE, ttnn::Shape({1, 1}), scalar, [&](uint32_t, uint32_t) { return scalar[0]; }},
          Case{Layout::ROW_MAJOR, ttnn::Shape({1, cols}), row, [&](uint32_t, uint32_t c) { return row[c]; }},
          Case{Layout::ROW_MAJOR, ttnn::Shape({rows, 1}), col, [&](uint32_t r, uint32_t) { return col[r]; }}}) {
        const auto ta = device_tensor(ttnn::Shape({rows, cols}), a, layout);
        const auto tb = device_tensor(shape, values, layout);
        const auto max = ttnn::maximum(ta, tb).cpu().to_vector<uint8_t>();
        const auto min = ttnn::minimum(ta, tb).cpu().to_vector<uint8_t>();
        size_t wrong = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                const uint32_t i = r * cols + c;
                wrong += max[i] != std::max(a[i], at(r, c));
                wrong += min[i] != std::min(a[i], at(r, c));
            }
        }
        EXPECT_EQ(wrong, 0u) << shape << " " << layout;
    }
}

TEST_F(MinMaxUint8Test, ClampTensor) {
    const ttnn::Shape shape({3, 224, 224});
    const auto x = bytes(shape.volume(), 0);
    std::vector<uint8_t> lo(shape.volume()), hi(shape.volume());
    for (size_t i = 0; i < lo.size(); ++i) {
        lo[i] = static_cast<uint8_t>((i * 13) % 128);
        hi[i] = static_cast<uint8_t>(lo[i] + (i * 29) % 128);
    }
    const auto got =
        ttnn::clamp(device_tensor(shape, x), device_tensor(shape, lo), device_tensor(shape, hi)).cpu().to_vector<uint8_t>();
    size_t wrong = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        wrong += got[i] != std::min(std::max(x[i], lo[i]), hi[i]);
    }
    EXPECT_EQ(wrong, 0u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
