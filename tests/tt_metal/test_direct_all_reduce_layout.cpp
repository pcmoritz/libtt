// SPDX-License-Identifier: Apache-2.0

// A single-tile-row BF16 all-reduce takes the direct path when direct_all_reduce_layout finds reduction cores for
// it, and reduce-scatter plus all-gather otherwise. Both give the same sums, so these tests check the layout itself:
// whether there is one, and on how many reduction cores.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/mesh_device.hpp>

#include "mesh_device_fixture.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class DirectAllReduceLayoutTest : public GenericMeshDeviceFixture {
protected:
    // The number of reduction cores for a [rows, width] BF16 DRAM tensor on ring_size ranks, or nullopt for the
    // fallback.
    std::optional<uint32_t> workers(uint32_t rows, uint32_t width, uint32_t ring_size) {
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape({rows, width}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), {}));
        const ttnn::Tensor input = ttnn::distributed::distribute_tensor(
            ttnn::Tensor::from_vector(std::vector<float>(static_cast<size_t>(rows) * width), spec),
            *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
            *mesh_device_);
        const auto layout = ttnn::experimental::prim::direct_all_reduce_layout(input, ring_size);
        if (!layout) {
            return std::nullopt;
        }
        const uint32_t cores = layout->grid.num_cores();
        // The cores reduce equal slices of the width.
        EXPECT_EQ(layout->shape[1] * cores, width);
        return cores;
    }
};

TEST_F(DirectAllReduceLayoutTest, PackedRowsUseAboutOneTileOfPayloadPerCore) {
    // One BF16 row packs into 128 bytes per tile, so 16 tiles make one tile of payload.
    EXPECT_EQ(workers(1, 2048, 4), 4u);
    EXPECT_EQ(workers(1, 4096, 4), 8u);
    EXPECT_EQ(workers(1, 5120, 2), 10u);
    // Rows that are not packed send whole tiles, on up to 32 cores.
    EXPECT_EQ(workers(17, 5120, 4), 32u);
}

TEST_F(DirectAllReduceLayoutTest, MoreCoresWhenFewerDoNotFit) {
    // 77 tiles aim for 5 cores, and of 1 to 5 only 1 divides 77. At four ranks that core's reduction CB would hold
    // 616 KiB, more than fits, so the next divisor takes it: 7 cores of 88 KiB. At two ranks 308 KiB fits.
    EXPECT_EQ(workers(1, 2464, 4), 7u);
    EXPECT_EQ(workers(1, 2464, 2), 1u);
    // 61 tiles have no other divisor up to 32. At four ranks one core holds 488 KiB, which fits.
    EXPECT_EQ(workers(5, 1952, 4), 1u);
}

TEST_F(DirectAllReduceLayoutTest, FallsBackWhenNoSlicesFit) {
    // 257 tiles, a prime: one core would hold 2 MiB.
    EXPECT_EQ(workers(1, 8224, 4), std::nullopt);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
