// SPDX-License-Identifier: Apache-2.0

// The line reduce-scatter's writers send to their neighbor's intermediate at the address the neighbor traded in
// the handshake, also when the caller shards the intermediate explicitly: the neighbor has the same shards on the
// same cores, at its own base address. Across processes the base addresses differ; within one, they agree, so this
// checks the sharded path's results on a 1xN line.

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
#include "ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

// Opens a 1xN mesh on a 1D line fabric, so reduce-scatters take the line algorithm.
class LineCclMeshDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
        const auto num_devices = static_cast<uint32_t>(tt::tt_metal::GetNumAvailableDevices());
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{1, num_devices}), /*l1_small_size=*/1 << 16);
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }

    std::shared_ptr<MeshDevice> mesh_device_;
};

TEST_F(LineCclMeshDeviceFixture, ReduceScatterWithShardedIntermediate) {
    const uint32_t num_devices = mesh_device_->num_devices();
    if (num_devices < 2) {
        GTEST_SKIP() << "needs two devices";
    }
    constexpr uint32_t rows = 32;
    constexpr uint32_t cores = 4;
    const uint32_t cols = 128 * num_devices;  // four tiles of each device's slice
    const tt::tt_metal::TensorSpec spec(
        ttnn::Shape({1, 1, rows, cols}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), {}));
    std::vector<float> values(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < values.size(); ++i) {
        // Partial sums of up to four stay below 256, where every integer is exact in BF16.
        values[i] = static_cast<float>(i % 61);
    }
    const ttnn::Tensor input = ttnn::distributed::distribute_tensor(
        ttnn::Tensor::from_vector(values, spec),
        *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
        *mesh_device_);

    // The line algorithm's intermediate holds both directions, [2, 1, rows, cols]: width-sharded over four cores.
    const tt::tt_metal::ShardSpec shard_spec(
        CoreRangeSet(CoreRange({0, 0}, {cores - 1, 0})), {2 * rows, cols / cores}, ShardOrientation::ROW_MAJOR);
    const ttnn::MemoryConfig intermediate_config(
        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED, tt::tt_metal::BufferType::L1, shard_spec);

    const ttnn::Tensor output = ttnn::reduce_scatter(
        input,
        /*dim=*/3,
        /*cluster_axis=*/1,
        /*subdevice_id=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        intermediate_config);

    const uint32_t slice = cols / num_devices;
    const auto shards = ttnn::distributed::get_device_tensors(output);
    ASSERT_EQ(shards.size(), num_devices);
    for (uint32_t device = 0; device < num_devices; ++device) {
        const std::vector<float> got = shards[device].cpu().to_vector<float>();
        ASSERT_EQ(got.size(), static_cast<size_t>(rows) * slice);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < slice; ++c) {
                const float want = num_devices * values[(static_cast<size_t>(r) * cols) + (device * slice) + c];
                ASSERT_EQ(got[(static_cast<size_t>(r) * slice) + c], want)
                    << "device " << device << ", row " << r << ", column " << c;
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
