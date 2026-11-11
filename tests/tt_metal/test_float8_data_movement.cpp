// SPDX-License-Identifier: Apache-2.0

// TTNN's data-movement ops take float8 tensors (FP8_E4M3, FP8_E5M2) on device as they are: each op returns a float8
// tensor holding exactly the bytes it moves for the same data as UINT8 (float8_data_movement.patch). Before, the
// ones that move data through compute kernels (tilize, untilize, transposes, unaligned tile slices and concats)
// asked for 32-bit DEST, whose float8 conversions aren't exact; others rejected float8 or e5m2, and tilize
// returned FLOAT32.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using Op = std::function<ttnn::Tensor(const ttnn::Tensor&)>;

class Float8DataMovementTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Distinct bytes, NaN codes and subnormals included.
    ttnn::Tensor bytes(const ttnn::Shape& shape, Layout layout) {
        std::vector<uint8_t> values(shape.volume());
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<uint8_t>((i * 37 + i / 7) % 256);
        }
        const TensorSpec spec(shape, TensorLayout(DataType::UINT8, PageConfig(layout), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    // The op on float8 views of the bytes gives float8 tensors of the bytes it gives for UINT8.
    void expect_bytes_kept(const std::string& name, const ttnn::Tensor& input, const Op& op) {
        const auto want = op(input).cpu().to_vector<uint8_t>();
        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            const ttnn::Tensor out = op(ttnn::bitcast(input, dtype));
            EXPECT_EQ(out.dtype(), dtype) << name;
            EXPECT_EQ(ttnn::bitcast(out, DataType::UINT8).cpu().to_vector<uint8_t>(), want) << name << " " << dtype;
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

ttnn::Tensor slice(const ttnn::Tensor& t, std::vector<int32_t> begins, std::vector<int32_t> ends) {
    const std::vector<int32_t> steps(begins.size(), 1);
    return ttnn::slice(
        t,
        ttsl::Span<const int32_t>(begins),
        ttsl::Span<const int32_t>(ends),
        ttsl::Span<const int32_t>(steps));
}

TEST_F(Float8DataMovementTest, Layouts) {
    const auto rm = bytes(ttnn::Shape({64, 96}), Layout::ROW_MAJOR);
    const auto tiled = bytes(ttnn::Shape({64, 96}), Layout::TILE);
    expect_bytes_kept("tilize", rm, [](const ttnn::Tensor& t) { return ttnn::tilize(t); });
    expect_bytes_kept("to_layout TILE", rm, [](const ttnn::Tensor& t) { return ttnn::to_layout(t, Layout::TILE); });
    expect_bytes_kept("tilize_with_zero_padding", bytes(ttnn::Shape({50, 70}), Layout::ROW_MAJOR), [](const ttnn::Tensor& t) {
        return ttnn::tilize_with_zero_padding(t);
    });
    expect_bytes_kept("untilize", tiled, [](const ttnn::Tensor& t) { return ttnn::untilize(t); });
    expect_bytes_kept("to_layout ROW_MAJOR", tiled, [](const ttnn::Tensor& t) {
        return ttnn::to_layout(t, Layout::ROW_MAJOR);
    });
}

TEST_F(Float8DataMovementTest, ReshapeSlicePermuteConcatCopy) {
    for (Layout layout : {Layout::ROW_MAJOR, Layout::TILE}) {
        const std::string l = layout == Layout::TILE ? " TILE" : " ROW_MAJOR";
        const auto x = bytes(ttnn::Shape({64, 96}), layout);
        const auto x3 = bytes(ttnn::Shape({2, 64, 96}), layout);
        expect_bytes_kept("reshape" + l, x, [](const ttnn::Tensor& t) { return ttnn::reshape(t, ttnn::Shape({96, 64})); });
        expect_bytes_kept("reshape 3d" + l, x3, [](const ttnn::Tensor& t) {
            return ttnn::reshape(t, ttnn::Shape({128, 96}));
        });
        expect_bytes_kept("slice" + l, x, [](const ttnn::Tensor& t) { return slice(t, {32, 32}, {64, 96}); });
        expect_bytes_kept("slice unaligned" + l, x, [](const ttnn::Tensor& t) { return slice(t, {5, 3}, {37, 70}); });
        expect_bytes_kept("permute" + l, x, [](const ttnn::Tensor& t) { return ttnn::permute(t, {1, 0}); });
        expect_bytes_kept("permute 021" + l, x3, [](const ttnn::Tensor& t) { return ttnn::permute(t, {0, 2, 1}); });
        expect_bytes_kept("concat 0" + l, x, [](const ttnn::Tensor& t) { return ttnn::concat({t, t}, 0); });
        expect_bytes_kept("concat 1" + l, x, [](const ttnn::Tensor& t) { return ttnn::concat({t, t}, 1); });
        expect_bytes_kept("copy" + l, x, [](const ttnn::Tensor& t) {
            auto dst = ttnn::empty_like(t);
            ttnn::copy(t, dst);
            return dst;
        });
    }
}

TEST_F(Float8DataMovementTest, Full) {
    // Each value's code: 1.0, -0.0 and 0.1 rounded to nearest even.
    struct Case {
        float value;
        uint8_t e4m3, e5m2;
    };
    for (const Case& c : {Case{1.0f, 0x38, 0x3c}, Case{-0.0f, 0x80, 0x80}, Case{0.1f, 0x1d, 0x2e}}) {
        for (Layout layout : {Layout::ROW_MAJOR, Layout::TILE}) {
            for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
                const auto t = ttnn::full(ttnn::Shape({40, 70}), c.value, dtype, layout, std::ref(*mesh_device_));
                ASSERT_EQ(t.dtype(), dtype);
                const auto got = ttnn::bitcast(t, DataType::UINT8).cpu().to_vector<uint8_t>();
                const uint8_t want = dtype == DataType::FP8_E4M3 ? c.e4m3 : c.e5m2;
                EXPECT_TRUE(std::all_of(got.begin(), got.end(), [&](uint8_t b) { return b == want; }))
                    << c.value << " " << dtype << " " << layout;
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
