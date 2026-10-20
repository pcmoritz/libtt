// SPDX-License-Identifier: Apache-2.0

// ttnn.pad pads uint8 tensors natively: row-major sticks of any byte width,
// with front padding, and whole tiles. Pad values are the type's own: bytes up
// to 255, and negative int32 values down to INT32_MIN in every pad factory
// (multi-core and single-core row-major, height- and width-sharded, tiled).

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using Padding = std::vector<std::pair<uint32_t, uint32_t>>;

class PadUint8Test : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    // Pads on the device and compares with padding on the host.
    template <typename T>
    void expect_pad(
        const ttnn::Shape& shape,
        const Padding& padding,
        float value,
        DataType dtype,
        Layout layout,
        bool use_multicore = true,
        const MemoryConfig& input_memory = {},
        const std::optional<MemoryConfig>& output_memory = std::nullopt) {
        const size_t rank = shape.rank();
        std::vector<T> values(shape.volume());
        for (size_t i = 0; i < values.size(); ++i) {
            // Every byte, the upper half of uint8 included; negative for int32.
            values[i] = static_cast<T>(static_cast<int64_t>((i * 37) % 256) - (std::is_signed_v<T> ? 128 : 0));
        }
        const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(layout), input_memory));
        const ttnn::Tensor input = ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
        ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> pad_spec;
        for (const auto& [before, after] : padding) {
            pad_spec.push_back({before, after});
        }
        const ttnn::Tensor output = ttnn::pad(input, pad_spec, value, use_multicore, output_memory);
        const std::vector<T> actual = output.cpu().to_vector<T>();

        std::vector<uint32_t> out_shape(rank);
        for (size_t d = 0; d < rank; ++d) {
            out_shape[d] = padding[d].first + shape[d] + padding[d].second;
        }
        size_t volume = 1;
        for (uint32_t s : out_shape) {
            volume *= s;
        }
        ASSERT_EQ(actual.size(), volume) << output.logical_shape();
        size_t mismatches = 0;
        std::vector<uint32_t> index(rank, 0);
        for (size_t o = 0; o < volume; ++o) {
            bool inside = true;
            size_t source = 0;
            for (size_t d = 0; d < rank; ++d) {
                inside &= index[d] >= padding[d].first && index[d] < padding[d].first + shape[d];
                source = source * shape[d] + (index[d] - padding[d].first);
            }
            const T want = inside ? values[source] : static_cast<T>(value);
            mismatches += actual[o] != want;
            for (size_t d = rank; d-- > 0;) {
                if (++index[d] < out_shape[d]) {
                    break;
                }
                index[d] = 0;
            }
        }
        EXPECT_EQ(mismatches, 0u) << shape << " " << dtype << " " << layout << " value " << value << " multicore "
                                  << use_multicore << " sharded " << input_memory.is_sharded();
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(PadUint8Test, RowMajor) {
    for (float value : {0.0f, 7.0f, 128.0f, 255.0f}) {
        // Front and back padding of the width, by odd byte counts.
        expect_pad<uint8_t>(ttnn::Shape({1, 1, 5, 7}), {{0, 0}, {0, 0}, {1, 0}, {2, 3}}, value, DataType::UINT8, Layout::ROW_MAJOR);
        expect_pad<uint8_t>(ttnn::Shape({1, 1, 5, 7}), {{0, 0}, {0, 0}, {0, 2}, {0, 1}}, value, DataType::UINT8, Layout::ROW_MAJOR);
    }
    // Sticks that aren't a multiple of the DRAM alignment, wide sticks, and many sticks.
    expect_pad<uint8_t>(ttnn::Shape({1, 1, 3, 1}), {{0, 0}, {0, 0}, {0, 0}, {5, 0}}, 9.0f, DataType::UINT8, Layout::ROW_MAJOR);
    expect_pad<uint8_t>(ttnn::Shape({1, 2, 9, 33}), {{0, 0}, {1, 1}, {2, 0}, {3, 30}}, 200.0f, DataType::UINT8, Layout::ROW_MAJOR);
    expect_pad<uint8_t>(ttnn::Shape({1, 1, 300, 100}), {{0, 0}, {0, 0}, {4, 5}, {1, 27}}, 128.0f, DataType::UINT8, Layout::ROW_MAJOR);
    expect_pad<uint8_t>(ttnn::Shape({2, 3, 4, 64}), {{1, 0}, {0, 2}, {0, 0}, {0, 0}}, 255.0f, DataType::UINT8, Layout::ROW_MAJOR);
}

TEST_F(PadUint8Test, Tiled) {
    expect_pad<uint8_t>(ttnn::Shape({1, 1, 32, 32}), {{0, 0}, {0, 0}, {0, 32}, {0, 64}}, 255.0f, DataType::UINT8, Layout::TILE);
    expect_pad<uint8_t>(ttnn::Shape({1, 2, 64, 32}), {{0, 0}, {0, 0}, {0, 0}, {0, 32}}, 7.0f, DataType::UINT8, Layout::TILE);
}

// Height sharding in L1 with one [height, width] shard, on core (0, 0).
MemoryConfig height_sharded(uint32_t height, uint32_t width) {
    return MemoryConfig(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(CoreRangeSet(CoreRange({0, 0}, {0, 0})), {height, width}, ShardOrientation::ROW_MAJOR));
}

TEST_F(PadUint8Test, ShardedUint8) {
    expect_pad<uint8_t>(
        ttnn::Shape({1, 1, 32, 16}), {{0, 0}, {0, 0}, {2, 6}, {0, 0}}, 250.0f, DataType::UINT8, Layout::ROW_MAJOR,
        true, height_sharded(32, 16), height_sharded(40, 16));
}

TEST_F(PadUint8Test, NegativeInt32PadValue) {
    // INT32_MIN is the -inf sentinel of integer max pooling.
    for (float value : {-3.0f, -1000.0f, -2147483648.0f}) {
        // The single-core row-major factory pads only the back, with 16-bit values, so these
        // single-core pads take the multi-core one.
        const Padding hw = {{0, 0}, {0, 0}, {1, 2}, {2, 3}};
        expect_pad<int32_t>(ttnn::Shape({1, 1, 5, 7}), hw, value, DataType::INT32, Layout::ROW_MAJOR);
        expect_pad<int32_t>(ttnn::Shape({1, 1, 5, 7}), hw, value, DataType::INT32, Layout::ROW_MAJOR, false);
        const Padding tiles = {{0, 0}, {0, 0}, {0, 32}, {0, 32}};
        expect_pad<int32_t>(ttnn::Shape({1, 1, 32, 32}), tiles, value, DataType::INT32, Layout::TILE);
        expect_pad<int32_t>(ttnn::Shape({1, 1, 32, 32}), tiles, value, DataType::INT32, Layout::TILE, false);
        // Height- and width-only padding of sharded tensors take their own factories.
        expect_pad<int32_t>(
            ttnn::Shape({1, 1, 32, 16}), {{0, 0}, {0, 0}, {2, 6}, {0, 0}}, value, DataType::INT32, Layout::ROW_MAJOR,
            true, height_sharded(32, 16), height_sharded(40, 16));
        expect_pad<int32_t>(
            ttnn::Shape({1, 1, 32, 16}), {{0, 0}, {0, 0}, {0, 0}, {4, 12}}, value, DataType::INT32, Layout::ROW_MAJOR,
            true, height_sharded(32, 16), height_sharded(32, 32));
    }
    // Front padding, and zeros at the back, which the single-core factory pads itself.
    expect_pad<int32_t>(ttnn::Shape({1, 1, 5, 7}), {{0, 0}, {0, 0}, {1, 2}, {2, 3}}, 0.0f, DataType::INT32, Layout::ROW_MAJOR, false);
    expect_pad<int32_t>(ttnn::Shape({1, 1, 5, 7}), {{0, 0}, {0, 0}, {0, 2}, {0, 3}}, 0.0f, DataType::INT32, Layout::ROW_MAJOR, false);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
