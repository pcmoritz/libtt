// SPDX-License-Identifier: Apache-2.0

// topk_select_plan splits each tile row's cores into column segments and row groups. These tests check the split
// on a fixed number of cores; topk_select_regression_test.py checks the selection against lax.top_k.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "mesh_device_fixture.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/reduction/topk/device/topk_select.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

class TopKSelectPlanTest : public GenericMeshDeviceFixture {
protected:
    // The plan for a [rows, width] FP32 DRAM tensor on an 8-row grid of `cores` cores.
    std::optional<ttnn::prim::TopKSelectPlan> plan(uint32_t rows, uint32_t width, uint32_t k, uint32_t cores) {
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape({1, 1, rows, width}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), {}));
        const ttnn::Tensor input = ttnn::distributed::distribute_tensor(
            ttnn::Tensor::from_vector(std::vector<float>(static_cast<size_t>(rows) * width), spec),
            *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
            *mesh_device_);
        const CoreRangeSet grid(CoreRange(CoreCoord(0, 0), CoreCoord(cores / 8 - 1, 7)));
        return ttnn::prim::topk_select_plan(input, k, false, false, input.memory_config(), grid);
    }
};

TEST_F(TopKSelectPlanTest, FewerSegmentsLeaveCoresToRowGroups) {
    // One segment per 16 tiles makes 64 segments of one row group, and a group's first-segment core would gather
    // 32 rows x 64 segments x 64 candidates, 1 MiB. The 64 KiB candidate limit allows 4 segments, which leaves 16
    // cores per segment for row groups of 2 rows: 64 KiB of input and 4 KiB of candidates per core.
    const auto split = plan(32, 32768, 64, 64);
    ASSERT_TRUE(split.has_value());
    EXPECT_EQ(split->segments, 4u);
    EXPECT_EQ(split->segment_tiles, 256u);
    EXPECT_EQ(split->row_groups, 16u);
    EXPECT_EQ(split->rows_per_core, 2u);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
