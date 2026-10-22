// SPDX-License-Identifier: Apache-2.0

// A trace can branch on a value the device computed: the prefetcher reads the
// predicate once the workloads before the branch have completed and skips the
// arm that is not taken. Replays alternate between the arms, the arms may hold
// different numbers of workloads, and workloads before and after the branch
// run on every replay.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "tests/tt_metal/mesh_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

constexpr uint32_t kPageSize = 64;

class TraceBranchTest : public GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        GenericMeshDeviceFixture::SetUp();
        out_ = dram_buffer();
        predicate_ = dram_buffer();
        // The kernels stage their page through L1 of core (0, 0); an L1 buffer with a page per bank
        // owns that address on every core.
        scratch_ = MeshBuffer::create(
            ReplicatedBufferConfig{.size = kPageSize * mesh_device_->allocator()->get_num_banks(BufferType::L1)},
            DeviceLocalBufferConfig{.page_size = kPageSize, .buffer_type = BufferType::L1},
            mesh_device_.get());
    }

    std::shared_ptr<MeshBuffer> dram_buffer() {
        return MeshBuffer::create(
            ReplicatedBufferConfig{.size = kPageSize},
            DeviceLocalBufferConfig{.page_size = kPageSize, .buffer_type = BufferType::DRAM},
            mesh_device_.get());
    }

    MeshWorkload write(uint32_t value) {
        return single_core_workload(
            "write_value.cpp",
            {static_cast<uint32_t>(out_->address()), value, static_cast<uint32_t>(scratch_->address())});
    }

    MeshWorkload add(uint32_t value) {
        return single_core_workload(
            "add_value.cpp",
            {static_cast<uint32_t>(out_->address()), value, static_cast<uint32_t>(scratch_->address())});
    }

    // Runs the workload once outside any trace so that its binaries are on device.
    void warm(MeshWorkload& workload) { EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, true); }

    void set_predicate(uint32_t value) {
        std::vector<uint32_t> words(kPageSize / sizeof(uint32_t), value);
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), predicate_, words, true);
    }

    // The first word of the output page on every device.
    std::vector<uint32_t> read_out() {
        std::vector<uint32_t> firsts;
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> words(kPageSize / sizeof(uint32_t));
            ReadShard(mesh_device_->mesh_command_queue(), words, out_, coord, true);
            firsts.push_back(words[0]);
        }
        return firsts;
    }

    void expect_out(uint32_t expected) {
        for (uint32_t value : read_out()) {
            EXPECT_EQ(value, expected);
        }
    }

    std::shared_ptr<MeshBuffer> out_;
    std::shared_ptr<MeshBuffer> predicate_;
    std::shared_ptr<MeshBuffer> scratch_;
};

TEST_F(TraceBranchTest, PlainTraceWithTheseKernels) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshWorkload start = write(100);
    MeshWorkload add_one = add(1);
    warm(start);
    warm(add_one);
    expect_out(101);
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq);
    EnqueueMeshWorkload(cq, start, false);
    EnqueueMeshWorkload(cq, add_one, false);
    EnqueueMeshWorkload(cq, add_one, false);
    mesh_device_->end_mesh_trace(cq, trace);
    for (int i = 0; i < 3; i++) {
        mesh_device_->replay_mesh_trace(cq, trace, true);
        expect_out(102);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceBranchTest, ReplaysTheArmThePredicateSelects) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshWorkload start = write(100);
    MeshWorkload add_one = add(1);
    MeshWorkload add_ten = add(10);
    MeshWorkload finish = add(1000);
    for (MeshWorkload* workload : {&start, &add_one, &add_ten, &finish}) {
        warm(*workload);
    }
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq);
    EnqueueMeshWorkload(cq, start, false);
    TraceBranchBegin(cq, *predicate_);
    // Three launches in the taken arm against one in the other.
    EnqueueMeshWorkload(cq, add_one, false);
    EnqueueMeshWorkload(cq, add_one, false);
    EnqueueMeshWorkload(cq, add_one, false);
    TraceBranchElse(cq);
    EnqueueMeshWorkload(cq, add_ten, false);
    TraceBranchEnd(cq);
    EnqueueMeshWorkload(cq, finish, false);
    mesh_device_->end_mesh_trace(cq, trace);

    for (uint32_t predicate : {1u, 0u, 1u, 1u, 0u, 0u, 7u}) {
        set_predicate(predicate);
        mesh_device_->replay_mesh_trace(cq, trace, true);
        expect_out(predicate ? 1103 : 1110);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceBranchTest, BranchesAtBothEndsOfTheTrace) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshWorkload start = write(5);
    MeshWorkload add_one = add(1);
    MeshWorkload add_ten = add(10);
    MeshWorkload add_hundred = add(100);
    for (MeshWorkload* workload : {&start, &add_one, &add_ten, &add_hundred}) {
        warm(*workload);
    }
    // The first branch reads a predicate no traced workload wrote; the second ends the trace.
    EnqueueMeshWorkload(cq, start, true);

    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq);
    TraceBranchBegin(cq, *predicate_);
    EnqueueMeshWorkload(cq, add_one, false);
    TraceBranchElse(cq);
    EnqueueMeshWorkload(cq, add_ten, false);
    TraceBranchEnd(cq);
    EnqueueMeshWorkload(cq, add_hundred, false);
    TraceBranchBegin(cq, *predicate_);
    EnqueueMeshWorkload(cq, add_one, false);
    EnqueueMeshWorkload(cq, add_one, false);
    TraceBranchElse(cq);
    TraceBranchEnd(cq);
    mesh_device_->end_mesh_trace(cq, trace);

    uint32_t expected = 5;
    for (uint32_t predicate : {0u, 1u, 1u, 0u}) {
        set_predicate(predicate);
        mesh_device_->replay_mesh_trace(cq, trace, true);
        expected += predicate ? 1 + 100 + 2 : 10 + 100;
        expect_out(expected);
    }
    mesh_device_->release_mesh_trace(trace);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
