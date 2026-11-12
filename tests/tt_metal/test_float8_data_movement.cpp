// SPDX-License-Identifier: Apache-2.0

// TTNN's data-movement ops take float8 tensors (FP8_E4M3, FP8_E5M2) on device as they are: each op returns a float8
// tensor holding exactly the bytes it moves for the same data as UINT8. Float8 moves as libtt moves FLOAT16:
// permute, transpose and pad declare its buffers UInt8 (data_movement_format), to_layout and repeat move it as UINT8
// (float8_data_movement.patch), and where selects its bytes as integers. Before, untilize, last-two-dimension permutes, unaligned tile slices and tile
// concats asked for 32-bit DEST, whose float8 conversions aren't exact; data-moving reshapes, full and e5m2 copies
// rejected float8, and tilize returned FLOAT32.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/float8.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
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
    const auto to_tile = [](const ttnn::Tensor& t) { return ttnn::to_layout(t, Layout::TILE); };
    const auto to_rows = [](const ttnn::Tensor& t) { return ttnn::to_layout(t, Layout::ROW_MAJOR); };
    expect_bytes_kept("to_layout TILE", bytes(ttnn::Shape({64, 96}), Layout::ROW_MAJOR), to_tile);
    expect_bytes_kept("to_layout TILE, padded", bytes(ttnn::Shape({50, 70}), Layout::ROW_MAJOR), to_tile);
    expect_bytes_kept("to_layout ROW_MAJOR", bytes(ttnn::Shape({64, 96}), Layout::TILE), to_rows);
    expect_bytes_kept("to_layout ROW_MAJOR, padded", bytes(ttnn::Shape({50, 70}), Layout::TILE), to_rows);
}

TEST_F(Float8DataMovementTest, ReshapeSlicePermuteRepeatConcatCopy) {
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
        expect_bytes_kept("repeat" + l, x, [](const ttnn::Tensor& t) { return ttnn::repeat(t, ttnn::Shape({2, 3})); });
        expect_bytes_kept("repeat rows" + l, bytes(ttnn::Shape({1, 96}), layout), [](const ttnn::Tensor& t) {
            return ttnn::repeat(t, ttnn::Shape({40, 1}));
        });
        expect_bytes_kept("concat 0" + l, x, [](const ttnn::Tensor& t) { return ttnn::concat({t, t}, 0); });
        expect_bytes_kept("concat 1" + l, x, [](const ttnn::Tensor& t) { return ttnn::concat({t, t}, 1); });
        // Padded tiles: concat untilizes and retilizes them.
        const auto padded = bytes(ttnn::Shape({50, 70}), layout);
        expect_bytes_kept("concat 0 padded" + l, padded, [](const ttnn::Tensor& t) { return ttnn::concat({t, t}, 0); });
        expect_bytes_kept("concat 1 padded" + l, padded, [](const ttnn::Tensor& t) { return ttnn::concat({t, t}, 1); });
        expect_bytes_kept("copy" + l, x, [](const ttnn::Tensor& t) {
            auto dst = ttnn::empty_like(t);
            ttnn::copy(t, dst);
            return dst;
        });
    }
}

// Fill values encode as libtt's exact typecast does (and ml_dtypes): rounded to nearest even, subnormals
// kept, and past the largest finite value NaN for e4m3 and infinity for e5m2.
struct Code {
    float value;
    uint8_t e4m3, e5m2;
};

constexpr Code kCodes[] = {
    {1.0f, 0x38, 0x3c},
    {-0.0f, 0x80, 0x80},
    {0.1f, 0x1d, 0x2e},
    {0x1p-9f, 0x01, 0x18},   // e4m3's smallest subnormal
    {0x1p-10f, 0x00, 0x14},  // half of it: a tie, to even
    {0x3p-10f, 0x02, 0x1a},
    {0x1p-16f, 0x00, 0x01},  // e5m2's smallest subnormal
    {464.0f, 0x7e, 0x5f},    // a tie between e4m3's largest finite value and NaN
    {500.0f, 0x7f, 0x60},
    {61440.0f, 0x7f, 0x7c},  // a tie between e5m2's largest finite value and infinity
    {std::numeric_limits<float>::infinity(), 0x7f, 0x7c},
    {-std::numeric_limits<float>::infinity(), 0xff, 0xfc},
};

TEST_F(Float8DataMovementTest, Full) {
    for (const Code& c : kCodes) {
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

// A reshape into tiles with padding fills the padding with the pad value's code.
TEST_F(Float8DataMovementTest, ReshapePadding) {
    const auto x = bytes(ttnn::Shape({64, 96}), Layout::TILE);
    const auto values = x.cpu().to_vector<uint8_t>();
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const auto reshaped =
            ttnn::reshape(ttnn::bitcast(x, dtype), ttnn::Shape({48, 128}), std::nullopt, ttnn::PadValue(1.0f));
        ASSERT_EQ(reshaped.dtype(), dtype);
        // The whole padded tensor, 64 x 128, as a view of the bytes.
        const auto all = ttnn::reshape(ttnn::bitcast(reshaped, DataType::UINT8), ttnn::Shape({64, 128}), ttnn::Shape({64, 128}))
                             .cpu()
                             .to_vector<uint8_t>();
        const uint8_t one = dtype == DataType::FP8_E4M3 ? 0x38 : 0x3c;
        for (size_t i = 0; i < all.size(); ++i) {
            ASSERT_EQ(all[i], i < values.size() ? values[i] : one) << dtype << " at " << i;
        }
    }
}

// where selects bytes: a bfloat16 predicate picks each float8 code of either tensor unchanged.
TEST_F(Float8DataMovementTest, Where) {
    const ttnn::Shape shape({64, 96});
    const auto a = bytes(shape, Layout::TILE);
    const std::vector<uint8_t> a_bytes = a.cpu().to_vector<uint8_t>();
    const std::vector<uint8_t> b_bytes(a_bytes.rbegin(), a_bytes.rend());
    const TensorSpec bytes_spec(shape, TensorLayout(DataType::UINT8, PageConfig(Layout::TILE), {}));
    const auto b = ttnn::Tensor::from_vector(b_bytes, bytes_spec).to_device(mesh_device_.get());
    std::vector<bfloat16> predicate(shape.volume());
    std::vector<uint8_t> want(shape.volume());
    for (size_t i = 0; i < predicate.size(); ++i) {
        const bool take_a = (i * 7 + i / 5) % 3 != 0;
        predicate[i] = bfloat16(take_a ? 1.0f : 0.0f);
        want[i] = take_a ? a_bytes[i] : b_bytes[i];
    }
    const TensorSpec predicate_spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), {}));
    const auto p = ttnn::Tensor::from_vector(predicate, predicate_spec).to_device(mesh_device_.get());
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const auto out = ttnn::where(p, ttnn::bitcast(a, dtype), ttnn::bitcast(b, dtype));
        ASSERT_EQ(out.dtype(), dtype);
        EXPECT_EQ(ttnn::bitcast(out, DataType::UINT8).cpu().to_vector<uint8_t>(), want) << dtype;
    }
}

// pad fills float8 padding with the pad value's code, in both layouts (before the data too in row-major: tile padding
// never pads in front).
TEST_F(Float8DataMovementTest, Pad) {
    for (Layout layout : {Layout::ROW_MAJOR, Layout::TILE}) {
        const auto x = bytes(ttnn::Shape({50, 70}), layout);
        const auto values = x.cpu().to_vector<uint8_t>();
        const uint32_t top = layout == Layout::TILE ? 0 : 3, left = layout == Layout::TILE ? 0 : 2;
        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            const ttsl::SmallVector<std::array<uint32_t, 2>> padding = {{top, 14 - top}, {left, 26 - left}};
            const auto out = ttnn::pad(ttnn::bitcast(x, dtype), padding, 1.0f);
            ASSERT_EQ(out.dtype(), dtype);
            ASSERT_EQ(out.logical_shape(), ttnn::Shape({64, 96}));
            const auto got = ttnn::bitcast(out, DataType::UINT8).cpu().to_vector<uint8_t>();
            const uint8_t one = dtype == DataType::FP8_E4M3 ? 0x38 : 0x3c;
            for (uint32_t r = 0; r < 64; ++r) {
                for (uint32_t c = 0; c < 96; ++c) {
                    const bool inside = r >= top && r < top + 50 && c >= left && c < left + 70;
                    ASSERT_EQ(got[r * 96 + c], inside ? values[(r - top) * 70 + c - left] : one)
                        << dtype << " " << layout << " at " << r << ", " << c;
                }
            }
        }
    }
}

// to_layout into or out of float8 converts exactly (ttnn::typecast through FLOAT32), as it does when the compiler
// moves a float8 op to FLOAT32: every finite code survives FLOAT32 -> float8 tiles and float8 tiles -> FLOAT32 rows.
TEST_F(Float8DataMovementTest, LayoutsWithConversion) {
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        std::vector<uint8_t> codes;
        for (uint32_t i = 0; i < 64 * 96; ++i) {
            const uint8_t code = static_cast<uint8_t>((i * 37 + i / 7) % 256);
            const float value = dtype == DataType::FP8_E4M3 ? float(float8_e4m3::from_bits(code))
                                                            : float(float8_e5m2::from_bits(code));
            codes.push_back(std::isfinite(value) ? code : 0);
        }
        std::vector<float> values;
        for (uint8_t code : codes) {
            values.push_back(
                dtype == DataType::FP8_E4M3 ? float(float8_e4m3::from_bits(code)) : float(float8_e5m2::from_bits(code)));
        }
        const ttnn::Shape shape({64, 96});
        const TensorSpec rows_spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), {}));
        const auto rows = ttnn::Tensor::from_vector(values, rows_spec).to_device(mesh_device_.get());
        const auto tiles = ttnn::to_layout(rows, Layout::TILE, dtype);
        ASSERT_EQ(tiles.dtype(), dtype);
        EXPECT_EQ(ttnn::bitcast(tiles, DataType::UINT8).cpu().to_vector<uint8_t>(), codes) << dtype;
        const auto back = ttnn::to_layout(tiles, Layout::ROW_MAJOR, DataType::FLOAT32);
        const auto got = back.cpu().to_vector<float>();
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_EQ(std::bit_cast<uint32_t>(got[i]), std::bit_cast<uint32_t>(values[i])) << dtype << " at " << i;
        }
    }
}

// A preallocated output must have the input's type: slice copies bytes, which another float8 format reads as
// other values.
TEST_F(Float8DataMovementTest, SliceOutputType) {
    const auto x = bytes(ttnn::Shape({64, 96}), Layout::ROW_MAJOR);
    const auto out = ttnn::bitcast(slice(x, {0, 0}, {32, 32}), DataType::FP8_E5M2);
    const std::vector<int32_t> b{0, 0}, e{32, 32}, s{1, 1};
    EXPECT_ANY_THROW((void)ttnn::slice(
        ttnn::bitcast(x, DataType::FP8_E4M3),
        ttsl::Span<const int32_t>(b),
        ttsl::Span<const int32_t>(e),
        ttsl::Span<const int32_t>(s),
        std::nullopt,
        out));
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
