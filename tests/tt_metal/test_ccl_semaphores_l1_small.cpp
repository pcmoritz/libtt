// SPDX-License-Identifier: Apache-2.0

// A collective creates its global semaphores when its program is first built and keeps them as long as the
// cached program. In L1 they would lower the L1 budget that ops such as matmul size their programs by, so a
// traced program would need different programs on its next run and its capture would fail. With an L1_SMALL
// region (as tt-mlir configures), the semaphores must leave L1 untouched.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

// tt-mlir's default L1_SMALL size.
constexpr size_t kL1SmallSize = 1 << 16;

// Opens a 1xN mesh on a 1D ring fabric with an L1_SMALL region, as tt-mlir does for collectives.
class CclMeshDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D_RING);
        const auto num_devices = static_cast<uint32_t>(tt::tt_metal::GetNumAvailableDevices());
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, num_devices}), kL1SmallSize);
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }

    // Sums a replicated [1, 1, rows, cols] tensor of ones across the mesh and checks every element.
    void all_reduce_ones(uint32_t rows, uint32_t cols) {
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape({1, 1, rows, cols}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), {}));
        const std::vector<float> ones(static_cast<size_t>(rows) * cols, 1.0f);
        const ttnn::Tensor input = ttnn::distributed::distribute_tensor(
            ttnn::Tensor::from_vector(ones, spec),
            *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
            *mesh_device_);

        const ttnn::Tensor output = ttnn::all_reduce(input, /*cluster_axis=*/1);

        const auto num_devices = static_cast<float>(mesh_device_->num_devices());
        for (const ttnn::Tensor& shard : ttnn::distributed::get_device_tensors(output)) {
            for (float value : shard.cpu().to_vector<float>()) {
                ASSERT_EQ(value, num_devices);
            }
        }
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(CclMeshDeviceFixture, ReduceScatterAllReduceLeavesL1Free) {
    ASSERT_FALSE(mesh_device_->lowest_occupied_compute_l1_address().has_value());
    // Several tile rows: reduce-scatter followed by all-gather.
    all_reduce_ones(/*rows=*/64, /*cols=*/1024);
    EXPECT_FALSE(mesh_device_->lowest_occupied_compute_l1_address().has_value())
        << "the collective left a buffer in L1 at " << *mesh_device_->lowest_occupied_compute_l1_address();
}

TEST_F(CclMeshDeviceFixture, DirectAllReduceLeavesL1Free) {
    ASSERT_FALSE(mesh_device_->lowest_occupied_compute_l1_address().has_value());
    // One tile row: exchanged through L1 scratch and reduced in one program.
    all_reduce_ones(/*rows=*/32, /*cols=*/1024);
    EXPECT_FALSE(mesh_device_->lowest_occupied_compute_l1_address().has_value())
        << "the collective left a buffer in L1 at " << *mesh_device_->lowest_occupied_compute_l1_address();
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
