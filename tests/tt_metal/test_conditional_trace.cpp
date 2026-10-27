// SPDX-License-Identifier: Apache-2.0

// A conditional trace is replayed or skipped on a value the device computed:
// enqueue_trace_if drains the work enqueued before it, and the prefetcher reads
// the predicate and skips the trace unless it matches. A conditional trace ends
// with the reset a replay starts with, so workloads and plain traces enqueued
// after it run as usual whether it ran or not. A dry launch dispatches a
// workload without running it, which prepares an unrun workload for capture.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "tests/tt_metal/mesh_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

constexpr uint32_t kPageSize = 64;

class ConditionalTraceTest : public GenericMeshDeviceFixture {
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

    MeshWorkload write(const MeshBuffer& buffer, uint32_t value) {
        return single_core_workload(
            "write_value.cpp",
            {static_cast<uint32_t>(buffer.address()), value, static_cast<uint32_t>(scratch_->address())});
    }

    MeshWorkload add(uint32_t value) {
        return single_core_workload(
            "add_value.cpp",
            {static_cast<uint32_t>(out_->address()), value, static_cast<uint32_t>(scratch_->address())});
    }

    // Runs the workload once outside any trace so that its binaries are on device.
    void warm(MeshWorkload& workload) { EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, true); }

    // Records the workloads as a conditional trace.
    MeshTraceId capture_conditional(const std::vector<MeshWorkload*>& workloads) {
        auto& cq = mesh_device_->mesh_command_queue();
        cq.set_conditional_capture(true);
        const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq);
        for (MeshWorkload* workload : workloads) {
            EnqueueMeshWorkload(cq, *workload, false);
        }
        mesh_device_->end_mesh_trace(cq, trace);
        cq.set_conditional_capture(false);
        return trace;
    }

    void set_predicate(uint32_t value) {
        std::vector<uint32_t> words(kPageSize / sizeof(uint32_t), value);
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), predicate_, words, true);
    }

    void expect_out(uint32_t expected) {
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> words(kPageSize / sizeof(uint32_t));
            ReadShard(mesh_device_->mesh_command_queue(), words, out_, coord, true);
            EXPECT_EQ(words[0], expected);
        }
    }

    std::shared_ptr<MeshBuffer> out_;
    std::shared_ptr<MeshBuffer> predicate_;
    std::shared_ptr<MeshBuffer> scratch_;
};

TEST_F(ConditionalTraceTest, ReplaysOnlyTheTraceThePredicateSelects) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshWorkload start = write(*out_, 100);
    MeshWorkload add_one = add(1);
    MeshWorkload add_ten = add(10);
    MeshWorkload finish = add(1000);
    for (MeshWorkload* workload : {&start, &add_one, &add_ten, &finish}) {
        warm(*workload);
    }
    // Three launches in one trace against one in the other.
    const MeshTraceId ones = capture_conditional({&add_one, &add_one, &add_one});
    const MeshTraceId tens = capture_conditional({&add_ten});

    for (uint32_t predicate : {1u, 0u, 1u, 1u, 0u, 0u, 7u}) {
        set_predicate(predicate);
        EnqueueMeshWorkload(cq, start, false);
        cq.enqueue_trace_if(ones, *predicate_, 1);
        cq.enqueue_trace_if(tens, *predicate_, 0);
        EnqueueMeshWorkload(cq, finish, false);
        expect_out(100 + (predicate == 1 ? 3 : 0) + (predicate == 0 ? 10 : 0) + 1000);
    }
    mesh_device_->release_mesh_trace(ones);
    mesh_device_->release_mesh_trace(tens);
}

TEST_F(ConditionalTraceTest, ReadsAPredicateTheWorkBeforeItWrote) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshWorkload start = write(*out_, 5);
    MeshWorkload select_ones = write(*predicate_, 1);
    MeshWorkload select_tens = write(*predicate_, 0);
    MeshWorkload add_one = add(1);
    MeshWorkload add_ten = add(10);
    for (MeshWorkload* workload : {&start, &select_ones, &select_tens, &add_one, &add_ten}) {
        warm(*workload);
    }
    const MeshTraceId ones = capture_conditional({&add_one});
    const MeshTraceId tens = capture_conditional({&add_ten});

    // Nothing waits between the workload that writes the predicate and the replays that read it.
    EnqueueMeshWorkload(cq, start, true);
    uint32_t expected = 5;
    for (bool take_ones : {true, false, false, true, true, false}) {
        EnqueueMeshWorkload(cq, take_ones ? select_ones : select_tens, false);
        cq.enqueue_trace_if(ones, *predicate_, 1);
        cq.enqueue_trace_if(tens, *predicate_, 0);
        expected += take_ones ? 1 : 10;
    }
    expect_out(expected);
    mesh_device_->release_mesh_trace(ones);
    mesh_device_->release_mesh_trace(tens);
}

TEST_F(ConditionalTraceTest, PlainTracesRunAfterConditionalOnes) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshWorkload start = write(*out_, 0);
    MeshWorkload add_one = add(1);
    MeshWorkload add_ten = add(10);
    for (MeshWorkload* workload : {&start, &add_one, &add_ten}) {
        warm(*workload);
    }
    const MeshTraceId conditional = capture_conditional({&add_ten, &add_ten});
    const MeshTraceId plain = mesh_device_->begin_mesh_trace(cq);
    EnqueueMeshWorkload(cq, add_one, false);
    mesh_device_->end_mesh_trace(cq, plain);

    EnqueueMeshWorkload(cq, start, true);
    uint32_t expected = 0;
    for (uint32_t predicate : {0u, 1u, 1u, 0u}) {
        set_predicate(predicate);
        cq.enqueue_trace_if(conditional, *predicate_, 1);
        mesh_device_->replay_mesh_trace(cq, plain, false);
        EnqueueMeshWorkload(cq, add_one, false);
        expected += (predicate == 1 ? 20 : 0) + 2;
    }
    expect_out(expected);
    mesh_device_->release_mesh_trace(conditional);
    mesh_device_->release_mesh_trace(plain);
}

TEST_F(ConditionalTraceTest, DryLaunchPreparesAWorkloadWithoutRunningIt) {
    auto& cq = mesh_device_->mesh_command_queue();
    MeshWorkload start = write(*out_, 50);
    warm(start);
    // Never run before: the dry launch compiles it and writes its binaries.
    MeshWorkload add_seven = add(7);
    cq.set_dry_launches(true);
    EnqueueMeshWorkload(cq, add_seven, false);
    EnqueueMeshWorkload(cq, add_seven, false);
    cq.set_dry_launches(false);
    expect_out(50);

    const MeshTraceId sevens = capture_conditional({&add_seven});
    for (uint32_t predicate : {1u, 0u, 1u}) {
        set_predicate(predicate);
        cq.enqueue_trace_if(sevens, *predicate_, 1);
    }
    expect_out(64);
    // A launch after the dry ones runs its kernel again.
    EnqueueMeshWorkload(cq, add_seven, false);
    expect_out(71);
    mesh_device_->release_mesh_trace(sevens);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
