// SPDX-License-Identifier: Apache-2.0

// A captured trace reuses the DRAM of the transient buffers it allocated and
// freed during capture on every replay. Those regions stay reserved after the
// capture, so a buffer allocated between replays cannot land there and make the
// trace unsafe to replay.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "tests/tt_metal/mesh_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

constexpr DeviceAddr kPageSize = 2048;
constexpr DeviceAddr kTransientSize = 64 * kPageSize;

std::shared_ptr<MeshBuffer> create_dram_buffer(MeshDevice* mesh_device, DeviceAddr size) {
    return MeshBuffer::create(
        ReplicatedBufferConfig{.size = size},
        DeviceLocalBufferConfig{.page_size = kPageSize, .buffer_type = BufferType::DRAM},
        mesh_device);
}

// Per-bank address range [begin, end) of a DRAM buffer of `size` bytes at `address`.
struct BankRange {
    DeviceAddr begin;
    DeviceAddr end;

    bool overlaps(const BankRange& other) const { return begin < other.end && other.begin < end; }
};

BankRange bank_range(MeshDevice* mesh_device, DeviceAddr address, DeviceAddr size) {
    const DeviceAddr num_banks = mesh_device->allocator()->get_num_banks(BufferType::DRAM);
    const DeviceAddr pages_per_bank = (size / kPageSize + num_banks - 1) / num_banks;
    return {address, address + pages_per_bank * kPageSize};
}

class MeshTraceReservedDramTest : public GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        GenericMeshDeviceFixture::SetUp();
        blank_ = single_core_workload("blank.cpp");
        // Compile and load the program before any capture.
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), blank_, true);
    }

    // Captures a trace that allocates and frees one transient DRAM buffer, and
    // returns the range that buffer occupied.
    BankRange capture_with_transient(MeshTraceId& trace_id) {
        auto& cq = mesh_device_->mesh_command_queue();
        trace_id = BeginTraceCapture(mesh_device_.get(), cq.id());
        DeviceAddr address = 0;
        {
            auto transient = create_dram_buffer(mesh_device_.get(), kTransientSize);
            address = transient->address();
            EnqueueMeshWorkload(cq, blank_, false);
        }
        mesh_device_->end_mesh_trace(cq.id(), trace_id);
        return bank_range(mesh_device_.get(), address, kTransientSize);
    }

    BankRange allocated_range(const MeshBuffer& buffer) {
        return bank_range(mesh_device_.get(), buffer.address(), buffer.size());
    }

    MeshWorkload blank_;
};

TEST_F(MeshTraceReservedDramTest, BufferAllocatedAfterCaptureAvoidsTransients) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshTraceId trace_id;
    const BankRange transient = capture_with_transient(trace_id);
    ASSERT_TRUE(mesh_device_->is_mesh_trace_replay_safe(trace_id));

    // First fit would place this buffer exactly where the transient was.
    auto buffer = create_dram_buffer(mesh_device_.get(), kTransientSize);
    EXPECT_FALSE(allocated_range(*buffer).overlaps(transient))
        << "buffer at " << buffer->address() << " landed in the trace's transient region at " << transient.begin;
    EXPECT_TRUE(mesh_device_->is_mesh_trace_replay_safe(trace_id));

    mesh_device_->replay_mesh_trace(cq.id(), trace_id, false);
    Finish(cq);
    mesh_device_->release_mesh_trace(trace_id);
}

TEST_F(MeshTraceReservedDramTest, CaptureWithoutAllocationsKeepsEarlierReservation) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshTraceId first;
    const BankRange transient = capture_with_transient(first);

    // Capturing frees the reserved regions while it runs. This capture uses only
    // buffers that already exist, so it allocates nothing.
    const MeshTraceId second = BeginTraceCapture(mesh_device_.get(), cq.id());
    EnqueueMeshWorkload(cq, blank_, false);
    mesh_device_->end_mesh_trace(cq.id(), second);

    auto buffer = create_dram_buffer(mesh_device_.get(), kTransientSize);
    EXPECT_FALSE(allocated_range(*buffer).overlaps(transient))
        << "buffer at " << buffer->address() << " landed in the first trace's transient region at "
        << transient.begin;
    EXPECT_TRUE(mesh_device_->is_mesh_trace_replay_safe(first));

    mesh_device_->replay_mesh_trace(cq.id(), first, false);
    mesh_device_->replay_mesh_trace(cq.id(), second, false);
    Finish(cq);
    mesh_device_->release_mesh_trace(second);
    mesh_device_->release_mesh_trace(first);
}

TEST_F(MeshTraceReservedDramTest, ReleasingTheLastTraceFreesItsTransients) {
    MeshTraceId trace_id;
    const BankRange transient = capture_with_transient(trace_id);
    mesh_device_->release_mesh_trace(trace_id);

    // With no trace left, first fit reuses the transient region again.
    auto buffer = create_dram_buffer(mesh_device_.get(), kTransientSize);
    EXPECT_TRUE(allocated_range(*buffer).overlaps(transient));
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
