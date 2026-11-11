// SPDX-License-Identifier: Apache-2.0

// The block tilize and untilize programs split a tensor into square blocks of
// tiles, one per core, and give the blocks cut short at the right and bottom
// edges kernels of their own. A tensor narrower than a block has no full-width
// blocks, and one shorter than a block no full-height ones; the programs run
// such tensors on their edge blocks alone. Blackhole tilizes all uint8 tensors
// with the block programs, so narrow tall ones (FP8 experts' bytes) took that
// path: 131072 x 512 failed with "Kernel 'compute_cliff_col' references
// unknown DFB 'in_full'".

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/data_movement/tilize/tilize.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class BlockTilizeSplitTest : public ::testing::Test {
protected:
    void SetUp() override { mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1})); }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
    }

    template <typename T>
    ttnn::Tensor row_major(uint32_t rows, uint32_t cols, DataType dtype, const std::vector<T>& values) {
        const TensorSpec spec(ttnn::Shape({rows, cols}), TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR), {}));
        return ttnn::Tensor::from_vector(values, spec).to_device(mesh_device_.get());
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

std::vector<uint8_t> bytes(size_t n) {
    std::vector<uint8_t> values(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = static_cast<uint8_t>((i * 37 + i / 7) % 251);
    }
    return values;
}

TEST_F(BlockTilizeSplitTest, Uint8Tilize) {
    // 16 tiles wide, narrower than a block, 4096 tiles tall: the gate or up
    // projection bytes of 64 of Qwen3.5-35B-A3B's experts. Then the transpose,
    // 32 rows, shorter than a block.
    for (const auto& [rows, cols] : {std::pair<uint32_t, uint32_t>{131072, 512}, {32, 131072}}) {
        const auto values = bytes(size_t{rows} * cols);
        const ttnn::Tensor tiled = ttnn::tilize(row_major(rows, cols, DataType::UINT8, values));
        ASSERT_EQ(tiled.layout(), Layout::TILE);
        EXPECT_EQ(tiled.cpu().to_vector<uint8_t>(), values) << rows << "x" << cols;
    }
}

TEST_F(BlockTilizeSplitTest, Uint8TilizeWithPadding) {
    // As above, with the last tile row and column padded.
    for (const auto& [rows, cols] : {std::pair<uint32_t, uint32_t>{131070, 500}, {20, 131070}}) {
        const uint32_t padded_rows = (rows + 31) / 32 * 32;
        const uint32_t padded_cols = (cols + 31) / 32 * 32;
        const auto values = bytes(size_t{rows} * cols);
        const ttnn::Tensor tiled = ttnn::tilize_with_val_padding(
            row_major(rows, cols, DataType::UINT8, values),
            ttnn::Shape({padded_rows, padded_cols}),
            ttnn::PadValue(uint32_t{0}));
        ASSERT_EQ(tiled.layout(), Layout::TILE);
        ASSERT_EQ(tiled.padded_shape(), ttnn::Shape({padded_rows, padded_cols}));
        EXPECT_EQ(tiled.cpu().to_vector<uint8_t>(), values) << rows << "x" << cols;
    }
}

TEST_F(BlockTilizeSplitTest, Bfloat16Untilize) {
    // 32 rows by 4096 tiles: shorter than a block, with full blocks only in
    // the bottom edge's row.
    const uint32_t rows = 32, cols = 131072;
    std::vector<bfloat16> values(size_t{rows} * cols);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = bfloat16(static_cast<float>(static_cast<int>(i % 509) - 254));
    }
    const ttnn::Tensor tiled = ttnn::tilize(row_major(rows, cols, DataType::BFLOAT16, values));
    const ttnn::Tensor untiled = ttnn::untilize(tiled);
    ASSERT_EQ(untiled.layout(), Layout::ROW_MAJOR);
    EXPECT_EQ(untiled.cpu().to_vector<bfloat16>(), values);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
