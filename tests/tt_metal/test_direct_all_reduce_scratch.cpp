// SPDX-License-Identifier: Apache-2.0

// A single-tile-row BF16 all-reduce takes the direct path when direct_all_reduce_scratch_config finds an L1
// scratch layout for it, and reduce-scatter plus all-gather otherwise. Both give the same sums, so these tests
// check the layout itself: whether there is one, and on how many reduction cores.

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

class DirectAllReduceScratchTest : public GenericMeshDeviceFixture {
protected:
    // The number of reduction cores of the scratch for a [rows, width] BF16 DRAM tensor on ring_size ranks, or
    // nullopt for the fallback.
    std::optional<uint32_t> workers(uint32_t rows, uint32_t width, uint32_t ring_size) {
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape({rows, width}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), {}));
        const ttnn::Tensor input = ttnn::distributed::distribute_tensor(
            ttnn::Tensor::from_vector(std::vector<float>(static_cast<size_t>(rows) * width), spec),
            *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
            *mesh_device_);
        const auto config = ttnn::experimental::prim::direct_all_reduce_scratch_config(input, ring_size);
        if (!config) {
            return std::nullopt;
        }
        const auto& shard = config->shard_spec().value();
        const uint32_t cores = shard.grid.num_cores();
        // Every core holds each rank's copy of an equal slice.
        EXPECT_EQ(shard.shape[1] * cores, ring_size * width);
        return cores;
    }
};

TEST_F(DirectAllReduceScratchTest, PackedRowsUseAboutOneTileOfPayloadPerCore) {
    // One BF16 row packs into 128 bytes per tile, so 16 tiles make one tile of payload.
    EXPECT_EQ(workers(1, 2048, 4), 4u);
    EXPECT_EQ(workers(1, 4096, 4), 8u);
    EXPECT_EQ(workers(1, 5120, 2), 10u);
    // Rows that are not packed send whole tiles, on up to 32 cores.
    EXPECT_EQ(workers(17, 5120, 4), 32u);
}

TEST_F(DirectAllReduceScratchTest, MoreCoresWhenFewerDoNotFit) {
    // 77 tiles aim for 5 cores, and of 1 to 5 only 1 divides 77. At four ranks that core would hold 616 KiB of
    // scratch, more than fits, so the next divisor takes it: 7 cores of 88 KiB. At two ranks 308 KiB fits.
    EXPECT_EQ(workers(1, 2464, 4), 7u);
    EXPECT_EQ(workers(1, 2464, 2), 1u);
    // 61 tiles have no other divisor up to 32. At four ranks one core holds 488 KiB, which fits.
    EXPECT_EQ(workers(5, 1952, 4), 1u);
}

TEST_F(DirectAllReduceScratchTest, FallsBackWhenNoSlicesFit) {
    // 257 tiles, a prime: one core would hold 2 MiB.
    EXPECT_EQ(workers(1, 8224, 4), std::nullopt);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
