// SPDX-License-Identifier: Apache-2.0

// A blocking read waits for all work queued ahead of it. It waits without
// holding the mesh device's API lock, so other threads can keep queueing work
// while the device drains.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tests/tt_metal/mesh_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

constexpr DeviceAddr kPageSize = 2048;
constexpr uint32_t kFlagSize = 32;

class MeshCommandQueueReadTest : public GenericMeshDeviceFixture {
protected:
    void SetUp() override {
        GenericMeshDeviceFixture::SetUp();
        // The flag lives in L1 of core (0, 0); an L1 buffer with a page per bank owns that address on every core.
        flag_ = MeshBuffer::create(
            ReplicatedBufferConfig{.size = kFlagSize * mesh_device_->allocator()->get_num_banks(BufferType::L1)},
            DeviceLocalBufferConfig{.page_size = kFlagSize, .buffer_type = BufferType::L1},
            mesh_device_.get());
        gate_ = single_core_workload("spin_until_flag.cpp", {static_cast<uint32_t>(flag_->address())});
        data_ = MeshBuffer::create(
            ReplicatedBufferConfig{.size = kPageSize},
            DeviceLocalBufferConfig{.page_size = kPageSize, .buffer_type = BufferType::DRAM},
            mesh_device_.get());
    }

    void write_flag(uint32_t value) {
        std::vector<uint32_t> data{value};
        for (IDevice* device : mesh_device_->get_devices()) {
            ASSERT_TRUE(tt::tt_metal::detail::WriteToDeviceL1(
                device, CoreCoord{0, 0}, static_cast<uint32_t>(flag_->address()), data));
        }
    }

    struct Attempt {
        // Whether the read was queued before the write, which the data it returned shows.
        bool read_queued_first = false;
        // Whether the write was queued while the read was still waiting for the gated device.
        bool write_queued_during_read = false;
    };

    // Holds the device in a spinning kernel, starts a blocking read on one thread,
    // then queues a write to the same buffer on another thread before releasing the device.
    Attempt write_during_gated_read() {
        auto& cq = mesh_device_->mesh_command_queue();
        const size_t num_words = data_->size() / sizeof(uint32_t);
        const std::vector<uint32_t> before(num_words, 1);
        const std::vector<uint32_t> after(num_words, 2);
        cq.enqueue_write_mesh_buffer(data_, before.data(), true);

        write_flag(0);
        EnqueueMeshWorkload(cq, gate_, false);

        auto host = DistributedHostBuffer::create(mesh_device_->shape());
        for (const auto& coord : MeshCoordinateRange(mesh_device_->shape())) {
            host.emplace_shard(coord, [&] { return HostBuffer(std::vector<uint32_t>(num_words)); });
        }
        std::atomic<bool> read_started = false;
        std::atomic<bool> read_done = false;
        std::thread reader([&] {
            read_started = true;
            cq.enqueue_read(data_, host, std::nullopt, /*blocking=*/true);
            read_done = true;
        });
        while (!read_started) {
            std::this_thread::yield();
        }
        // Give the read time to be queued; the data it returns shows whether it was.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        std::atomic<bool> write_queued = false;
        std::thread writer([&] {
            cq.enqueue_write_mesh_buffer(data_, after.data(), /*blocking=*/false);
            write_queued = true;
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!write_queued && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Attempt attempt;
        // The device is still held, so the read cannot have finished.
        attempt.write_queued_during_read = write_queued && !read_done;

        write_flag(1);
        reader.join();
        writer.join();
        Finish(cq);

        const auto bytes = host.get_shard(MeshCoordinate(0, 0))->view_bytes();
        std::vector<uint32_t> read(num_words);
        std::memcpy(read.data(), bytes.data(), num_words * sizeof(uint32_t));
        EXPECT_TRUE(read == before || read == after) << "read returned neither the old nor the new data";
        attempt.read_queued_first = read == before;
        return attempt;
    }

    std::shared_ptr<MeshBuffer> flag_;
    std::shared_ptr<MeshBuffer> data_;
    MeshWorkload gate_;
};

TEST_F(MeshCommandQueueReadTest, WriteQueuesWhileBlockingReadWaits) {
    // An attempt where the write was queued before the read proves nothing; retry it.
    for (int i = 0; i < 5; ++i) {
        const Attempt attempt = write_during_gated_read();
        if (attempt.read_queued_first) {
            EXPECT_TRUE(attempt.write_queued_during_read) << "queueing a write waited for another thread's read";
            return;
        }
    }
    FAIL() << "the read was never queued before the write in 5 attempts";
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
