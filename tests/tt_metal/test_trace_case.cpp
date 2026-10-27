// SPDX-License-Identifier: Apache-2.0

// A trace can hold cases whose branch the device selects: the prefetcher reads
// the index once the work before the case has completed, skips the branches it
// does not select, and every branch ends by returning the device to the state a
// replay starts in. Replays alternate between branches of different lengths,
// cases nest, a case may start or end the trace, and eager work after a replay
// sees consistent dispatch state. A trace can record workloads that were only
// prepared and never ran.

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

class TraceCaseTest : public GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        GenericMeshDeviceFixture::SetUp();
        out_ = dram_buffer();
        index_ = dram_buffer();
        // The kernels stage their page through L1 of core (0, 0); an L1 buffer with a page per bank
        // owns that address on every core.
        scratch_ = MeshBuffer::create(
            ReplicatedBufferConfig{.size = kPageSize * mesh_device_->allocator()->get_num_banks(BufferType::L1)},
            DeviceLocalBufferConfig{.page_size = kPageSize, .buffer_type = BufferType::L1},
            mesh_device_.get());
    }

    MeshCommandQueue& cq() { return mesh_device_->mesh_command_queue(); }

    std::shared_ptr<MeshBuffer> dram_buffer() {
        return MeshBuffer::create(
            ReplicatedBufferConfig{.size = kPageSize},
            DeviceLocalBufferConfig{.page_size = kPageSize, .buffer_type = BufferType::DRAM},
            mesh_device_.get());
    }

    // Workloads on out_, run once outside any trace so that their binaries are on device.
    MeshWorkload& write(uint32_t value) { return warmed("write_value.cpp", {address(*out_), value, address(*scratch_)}); }
    MeshWorkload& add(uint32_t value) { return warmed("add_value.cpp", {address(*out_), value, address(*scratch_)}); }
    MeshWorkload& copy(const MeshBuffer& src, const MeshBuffer& dst, uint32_t delay) {
        return warmed("copy_value.cpp", {address(src), address(dst), address(*scratch_), delay});
    }

    static uint32_t address(const MeshBuffer& buffer) { return static_cast<uint32_t>(buffer.address()); }

    MeshWorkload& warmed(const std::string& kernel, const std::vector<uint32_t>& args) {
        workloads_.push_back(std::make_unique<MeshWorkload>(single_core_workload(kernel, args)));
        EnqueueMeshWorkload(cq(), *workloads_.back(), true);
        return *workloads_.back();
    }

    // A workload on out_ that is compiled and has its binaries on device, but never ran.
    MeshWorkload& prepared(const std::string& kernel, uint32_t value) {
        workloads_.push_back(
            std::make_unique<MeshWorkload>(single_core_workload(kernel, {address(*out_), value, address(*scratch_)})));
        cq().set_prepare_only(true);
        EnqueueMeshWorkload(cq(), *workloads_.back(), false);
        cq().set_prepare_only(false);
        return *workloads_.back();
    }

    void enqueue(MeshWorkload& workload, int times = 1) {
        for (int i = 0; i < times; i++) {
            EnqueueMeshWorkload(cq(), workload, false);
        }
    }

    void set(const std::shared_ptr<MeshBuffer>& buffer, uint32_t value) {
        std::vector<uint32_t> words(kPageSize / sizeof(uint32_t), value);
        cq().enqueue_write_mesh_buffer(buffer, words.data(), /*blocking=*/true);
    }

    void expect_out(uint32_t expected) {
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            std::vector<uint32_t> words(kPageSize / sizeof(uint32_t));
            ReadShard(cq(), words, out_, coord, true);
            EXPECT_EQ(words[0], expected) << "on device " << coord;
        }
    }

    std::shared_ptr<MeshBuffer> out_;
    std::shared_ptr<MeshBuffer> index_;
    std::shared_ptr<MeshBuffer> scratch_;
    std::vector<std::unique_ptr<MeshWorkload>> workloads_;
};

TEST_F(TraceCaseTest, PlainTraceWithTheseKernels) {
    MeshWorkload& start = write(100);
    MeshWorkload& add_one = add(1);
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    enqueue(start);
    enqueue(add_one, 2);
    mesh_device_->end_mesh_trace(cq(), trace);
    for (int i = 0; i < 3; i++) {
        mesh_device_->replay_mesh_trace(cq(), trace, true);
        expect_out(102);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceCaseTest, ReplaysTheBranchTheIndexSelects) {
    MeshWorkload& start = write(100);
    MeshWorkload& add_one = add(1);
    MeshWorkload& add_ten = add(10);
    MeshWorkload& finish = add(1000);
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    enqueue(start);
    cq().record_case_begin(*index_);
    cq().record_case_branch();
    // Three launches in one branch against one in the other.
    enqueue(add_one, 3);
    cq().record_case_branch();
    enqueue(add_ten);
    cq().record_case_end();
    enqueue(finish);
    mesh_device_->end_mesh_trace(cq(), trace);

    // Any index but 0 selects the last branch.
    for (uint32_t index : {0u, 1u, 0u, 0u, 1u, 1u, 7u, 0xFFFFFFFFu}) {
        set(index_, index);
        mesh_device_->replay_mesh_trace(cq(), trace, true);
        expect_out(index == 0 ? 1103 : 1110);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceCaseTest, TestsTheLowBitsOfTheIndex) {
    MeshWorkload& start = write(0);
    MeshWorkload& add_one = add(1);
    MeshWorkload& add_ten = add(10);
    // A bfloat16 predicate in the low half of the word, without its sign bit.
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    enqueue(start);
    cq().record_case_begin(*index_, /*index_bits=*/15);
    cq().record_case_branch();
    enqueue(add_one);
    cq().record_case_branch();
    enqueue(add_ten);
    cq().record_case_end();
    mesh_device_->end_mesh_trace(cq(), trace);

    for (uint32_t index : {0x12343F80u, 0xFFFF0000u, 0x00008000u, 0xABCD3F80u, 0x00000000u}) {
        set(index_, index);
        mesh_device_->replay_mesh_trace(cq(), trace, true);
        expect_out((index & 0x7FFF) == 0 ? 1 : 10);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceCaseTest, SelectsOneOfManyBranches) {
    MeshWorkload& start = write(0);
    MeshWorkload& add_one = add(1);
    MeshWorkload& add_ten = add(10);
    MeshWorkload& add_hundred = add(100);
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    enqueue(start);
    cq().record_case_begin(*index_);
    cq().record_case_branch();
    enqueue(add_one);
    cq().record_case_branch();
    enqueue(add_ten, 2);
    cq().record_case_branch();
    cq().record_case_branch();
    enqueue(add_hundred, 3);
    cq().record_case_end();
    mesh_device_->end_mesh_trace(cq(), trace);

    const std::vector<uint32_t> results = {1, 20, 0, 300};
    for (uint32_t index : {3u, 0u, 2u, 1u, 1u, 2u, 0u, 4u, 0xFFFFFFFFu}) {
        set(index_, index);
        mesh_device_->replay_mesh_trace(cq(), trace, true);
        expect_out(results[index < 4 ? index : 3]);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceCaseTest, CasesStartAndEndTheTrace) {
    MeshWorkload& add_one = add(1);
    MeshWorkload& add_ten = add(10);
    MeshWorkload& add_hundred = add(100);
    MeshWorkload& start = write(5);
    EnqueueMeshWorkload(cq(), start, true);

    // The first case reads an index no traced workload wrote; the second ends the trace, so no workload of the
    // trace is still running when it ends.
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    cq().record_case_begin(*index_);
    cq().record_case_branch();
    enqueue(add_one);
    cq().record_case_branch();
    enqueue(add_ten);
    cq().record_case_end();
    enqueue(add_hundred);
    cq().record_case_begin(*index_);
    cq().record_case_branch();
    enqueue(add_one, 2);
    cq().record_case_branch();
    cq().record_case_end();
    mesh_device_->end_mesh_trace(cq(), trace);

    uint32_t expected = 5;
    for (uint32_t index : {1u, 0u, 0u, 1u}) {
        set(index_, index);
        // Eager work right after the replay shares its dispatch state.
        mesh_device_->replay_mesh_trace(cq(), trace, false);
        enqueue(add_one, 9);
        expected += (index == 0 ? 1 + 100 + 2 : 10 + 100) + 9;
        expect_out(expected);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceCaseTest, CasesNest) {
    std::shared_ptr<MeshBuffer> inner = dram_buffer();
    MeshWorkload& start = write(0);
    MeshWorkload& add_one = add(1);
    MeshWorkload& add_ten = add(10);
    MeshWorkload& add_hundred = add(100);
    MeshWorkload& add_thousand = add(1000);
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    enqueue(start);
    cq().record_case_begin(*index_);
    cq().record_case_branch();
    enqueue(add_one);
    cq().record_case_begin(*inner);
    cq().record_case_branch();
    enqueue(add_ten);
    cq().record_case_branch();
    enqueue(add_hundred, 2);
    cq().record_case_end();
    enqueue(add_thousand);
    cq().record_case_branch();
    // A case at the start of a branch, with an empty branch.
    cq().record_case_begin(*inner);
    cq().record_case_branch();
    cq().record_case_branch();
    enqueue(add_ten, 3);
    cq().record_case_end();
    cq().record_case_end();
    enqueue(add_one);
    mesh_device_->end_mesh_trace(cq(), trace);

    for (auto [outer_index, inner_index] : std::vector<std::pair<uint32_t, uint32_t>>{
             {0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 1}, {1, 1}, {0, 0}, {1, 0}}) {
        set(index_, outer_index);
        set(inner, inner_index);
        mesh_device_->replay_mesh_trace(cq(), trace, true);
        const uint32_t branch = outer_index == 0 ? 1 + (inner_index == 0 ? 10 : 200) + 1000 : (inner_index ? 30 : 0);
        expect_out(branch + 1);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceCaseTest, ReadsAnIndexTheTraceJustWrote) {
    std::shared_ptr<MeshBuffer> source = dram_buffer();
    MeshWorkload& start = write(0);
    // The copy finishes well after the prefetcher reaches the case, unless the case waits for it.
    MeshWorkload& compute_index = copy(*source, *index_, /*delay=*/200);
    MeshWorkload& add_one = add(1);
    MeshWorkload& add_ten = add(10);
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    enqueue(start);
    enqueue(compute_index);
    cq().record_case_begin(*index_);
    cq().record_case_branch();
    enqueue(add_one);
    cq().record_case_branch();
    enqueue(add_ten);
    cq().record_case_end();
    mesh_device_->end_mesh_trace(cq(), trace);

    uint32_t previous = 1;
    for (uint32_t index : {0u, 1u, 1u, 0u, 1u, 0u}) {
        set(source, index);
        // The index buffer still holds the previous replay's value.
        set(index_, previous);
        mesh_device_->replay_mesh_trace(cq(), trace, true);
        expect_out(index == 0 ? 1 : 10);
        previous = index;
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceCaseTest, SkipsBranchesLargerThanThePrefetchBuffer) {
    MeshWorkload& start = write(0);
    MeshWorkload& add_one = add(1);
    MeshWorkload& add_ten = add(10);
    constexpr int kLaunches = 1500;
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    enqueue(start);
    cq().record_case_begin(*index_);
    cq().record_case_branch();
    enqueue(add_one, kLaunches);
    cq().record_case_branch();
    enqueue(add_ten);
    cq().record_case_branch();
    enqueue(add_ten, kLaunches);
    cq().record_case_end();
    enqueue(add_one);
    mesh_device_->end_mesh_trace(cq(), trace);

    for (uint32_t index : {1u, 0u, 2u, 1u, 2u, 0u}) {
        set(index_, index);
        mesh_device_->replay_mesh_trace(cq(), trace, true);
        expect_out((index == 0 ? kLaunches : index == 1 ? 10 : 10 * kLaunches) + 1);
    }
    mesh_device_->release_mesh_trace(trace);
}

TEST_F(TraceCaseTest, RecordsWorkloadsThatWereOnlyPrepared) {
    set(out_, 5);
    MeshWorkload& start = prepared("write_value.cpp", 100);
    MeshWorkload& add_one = prepared("add_value.cpp", 1);
    MeshWorkload& add_ten = prepared("add_value.cpp", 10);
    expect_out(5);
    const MeshTraceId trace = mesh_device_->begin_mesh_trace(cq());
    enqueue(start);
    cq().record_case_begin(*index_);
    cq().record_case_branch();
    enqueue(add_one);
    cq().record_case_branch();
    enqueue(add_ten);
    cq().record_case_end();
    mesh_device_->end_mesh_trace(cq(), trace);

    for (uint32_t index : {1u, 0u, 1u}) {
        set(index_, index);
        mesh_device_->replay_mesh_trace(cq(), trace, true);
        expect_out(index == 0 ? 101 : 110);
    }
    mesh_device_->release_mesh_trace(trace);
    // They run eagerly as well.
    enqueue(add_one);
    cq().finish();
    expect_out(111);
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
