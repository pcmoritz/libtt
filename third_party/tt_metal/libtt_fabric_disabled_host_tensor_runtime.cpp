// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/tile.hpp>

#include "tt_metal/distributed/distributed_coordinate_translator.hpp"
#include "tt_metal/distributed/pinned_memory_cache.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tt::tt_metal {

Tile::Tile(std::array<uint32_t, 2> tile_shape, bool transpose_tile) : tile_shape(tile_shape) {
    static constexpr std::array<std::array<std::array<uint32_t, 2>, 2>, 12> kTileFaceShapes = {{
        {{{32, 32}, {16, 16}}},
        {{{16, 32}, {16, 16}}},
        {{{32, 16}, {16, 16}}},
        {{{16, 16}, {16, 16}}},
        {{{8, 32}, {8, 16}}},
        {{{4, 32}, {4, 16}}},
        {{{2, 32}, {2, 16}}},
        {{{1, 32}, {1, 16}}},
        {{{8, 16}, {8, 16}}},
        {{{4, 16}, {4, 16}}},
        {{{2, 16}, {2, 16}}},
        {{{1, 16}, {1, 16}}},
    }};

    const auto* choice = std::find_if(kTileFaceShapes.begin(), kTileFaceShapes.end(), [this](const auto& pair) {
        if (pair[0] == this->tile_shape) {
            this->face_shape = pair[1];
            return true;
        }
        return false;
    });
    TT_FATAL(choice != kTileFaceShapes.end(), "Tile size is not valid for TT hardware");

    transpose_within_face = transpose_tile;
    transpose_of_faces = transpose_tile;
    if (transpose_tile) {
        TT_FATAL(
            tile_shape[0] == constants::FACE_HEIGHT || tile_shape[0] == constants::TILE_HEIGHT,
            "Tile height must equal 16 or 32 in transpose mode");
    }

    tile_hw = tile_shape[0] * tile_shape[1];
    face_hw = face_shape[0] * face_shape[1];
    num_faces = tile_hw / face_hw;
    partial_face = static_cast<uint32_t>(tile_shape[0] < constants::TILE_HEIGHT);
    narrow_tile = static_cast<uint32_t>(tile_shape[1] < constants::TILE_WIDTH);
}

uint32_t Tile::get_tile_size(const DataFormat& format) const {
    constexpr uint32_t kHostOnlyL1Alignment = 16;
    const uint32_t aligned_exp_size = tt::round_up(face_shape[0] * num_faces, kHostOnlyL1Alignment);
    switch (format) {
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b: return (tile_hw / 4) + aligned_exp_size;
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b: return (tile_hw / 2) + aligned_exp_size;
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b: return tile_hw + aligned_exp_size;
        case DataFormat::MxFp4: return tt::round_up(tile_hw / 32, kHostOnlyL1Alignment) + (tile_hw / 2);
        case DataFormat::MxFp6P:
        case DataFormat::MxFp6R:
        case DataFormat::MxFp8R:
        case DataFormat::MxFp8P: return tt::round_up(tile_hw / 32, kHostOnlyL1Alignment) + tile_hw;
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::UInt16:
        case DataFormat::Int16:
        case DataFormat::RawUInt16: return tile_hw * 2;
        case DataFormat::Float32:
        case DataFormat::UInt32:
        case DataFormat::Int32:
        case DataFormat::RawUInt32: return tile_hw * 4;
        case DataFormat::Fp8_e4m3:
        case DataFormat::Int8:
        case DataFormat::Lf8:
        case DataFormat::UInt8:
        case DataFormat::RawUInt8: return tile_hw;
        case DataFormat::Tf32: throw std::invalid_argument("TF32 unsupported");
        case DataFormat::Invalid: throw std::invalid_argument("Invalid data format");
        default: throw std::invalid_argument("Unknown data format");
    }
}

bool Tile::operator==(const Tile& other) const {
    return tile_shape == other.tile_shape && face_shape == other.face_shape;
}

std::ostream& operator<<(std::ostream& os, const Tile& tile) {
    os << "Tile(shape=[" << tile.get_height() << ", " << tile.get_width() << "], face=["
       << tile.get_face_shape()[0] << ", " << tile.get_face_shape()[1] << "])";
    return os;
}

DistributedHostBuffer DistributedHostBuffer::create(const distributed::MeshShape& shape) {
    return DistributedHostBuffer::create(
        shape,
        shape,
        distributed::MeshCoordinate::zero_coordinate(shape.dims()),
        /*context=*/nullptr);
}

DistributedHostBuffer DistributedHostBuffer::create(
    const distributed::MeshShape& global_shape,
    const distributed::MeshShape& local_shape,
    const distributed::MeshCoordinate& local_offset,
    const std::shared_ptr<distributed::multihost::DistributedContext>& context) {
    DistributedCoordinateTranslator translator(global_shape, local_shape, local_offset);
    std::vector<distributed::MaybeRemote<Shard>> shards(
        global_shape.mesh_size(), distributed::MaybeRemote<Shard>::remote());

    size_t shard_index = 0;
    for (const auto& coord : distributed::MeshCoordinateRange(global_shape)) {
        if (translator.is_local(coord)) {
            shards[shard_index] = distributed::MaybeRemote<Shard>::local(Shard{.is_populated = false});
        }
        ++shard_index;
    }

    return DistributedHostBuffer(
        distributed::DistributedMeshContainer<Shard>(global_shape, std::move(shards)),
        /*populated_shards=*/{},
        context);
}

DistributedHostBuffer DistributedHostBuffer::create(const distributed::MeshDeviceView& mesh_device_view) {
    return DistributedHostBuffer::create(mesh_device_view.shape());
}

std::vector<size_t> DistributedHostBuffer::get_populated_shard_indices() const {
    const auto& shards = shards_.values();
    std::vector<size_t> indices;
    indices.reserve(shards.size());
    for (size_t i = 0; i < shards.size(); ++i) {
        if (shards[i].is_local() && shards[i]->is_populated) {
            indices.push_back(i);
        }
    }
    return indices;
}

std::optional<HostBuffer> DistributedHostBuffer::get_shard(const distributed::MeshCoordinate& coord) const {
    const auto& shard = shards_.at(coord);
    if (shard.is_local() && shard->is_populated) {
        return shard->buffer;
    }
    return std::nullopt;
}

void DistributedHostBuffer::emplace_shard(
    const distributed::MeshCoordinate& coord, const std::function<HostBuffer()>& produce_buffer) {
    shard_coords_.insert(coord);
    auto& shard = shards_.at(coord);
    if (shard.is_local()) {
        shard->buffer = produce_buffer();
        shard->is_populated = true;
    }
}

bool DistributedHostBuffer::is_local(const distributed::MeshCoordinate& coord) const { return shards_.is_local(coord); }

DistributedHostBuffer DistributedHostBuffer::transform(
    const TransformFn& fn, ProcessShardExecutionPolicy /*policy*/) const {
    const auto& shards = shards_.values();
    std::vector<distributed::MaybeRemote<Shard>> transformed_shards(
        shards.size(), distributed::MaybeRemote<Shard>::remote());

    for (size_t index : get_populated_shard_indices()) {
        transformed_shards[index] =
            distributed::MaybeRemote<Shard>::local(Shard{.buffer = fn(shards[index]->buffer), .is_populated = true});
    }

    return DistributedHostBuffer(
        distributed::DistributedMeshContainer<Shard>(shards_.shape(), std::move(transformed_shards)),
        shard_coords_,
        context_);
}

void DistributedHostBuffer::apply(const ApplyFn& fn, ProcessShardExecutionPolicy /*policy*/) const {
    const auto& shards = shards_.values();
    for (size_t index : get_populated_shard_indices()) {
        fn(shards[index]->buffer);
    }
}

void DistributedHostBuffer::emplace_shards(
    const std::vector<distributed::MeshCoordinate>& coords,
    const ProduceBufferFn& produce_buffer,
    ProcessShardExecutionPolicy /*policy*/) {
    for (const auto& coord : coords) {
        shard_coords_.insert(coord);
        auto& shard = shards_.at(coord);
        if (shard.is_local()) {
            shard->buffer = produce_buffer(coord);
            shard->is_populated = true;
        }
    }
}

const distributed::MeshShape& DistributedHostBuffer::shape() const { return shards_.shape(); }

const std::set<distributed::MeshCoordinate>& DistributedHostBuffer::shard_coords() const { return shard_coords_; }

const std::shared_ptr<distributed::multihost::DistributedContext>& DistributedHostBuffer::context() const {
    return context_;
}

std::ostream& operator<<(std::ostream& os, const ShardSpec& spec) {
    os << "ShardSpec{grid_ranges=" << spec.grid.ranges().size() << ", shape=[" << spec.shape[0] << ", "
       << spec.shape[1] << "], orientation=";
    switch (spec.orientation) {
        case ShardOrientation::ROW_MAJOR: os << "ROW_MAJOR"; break;
        case ShardOrientation::COL_MAJOR: os << "COL_MAJOR"; break;
    }
    os << "}";
    return os;
}

bool is_sharded(const TensorMemoryLayout& layout) {
    return layout == TensorMemoryLayout::HEIGHT_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED ||
           layout == TensorMemoryLayout::BLOCK_SHARDED || layout == TensorMemoryLayout::ND_SHARDED;
}

bool ShardSpec::operator==(const ShardSpec&) const = default;
bool ShardSpec::operator!=(const ShardSpec&) const = default;

std::array<uint32_t, 2> ShardSpecBuffer::shape_in_pages() const {
    const uint32_t height_in_pages = page_shape[0] == 0 ? 0 : tensor_shard_spec.shape[0] / page_shape[0];
    const uint32_t width_in_pages = page_shape[1] == 0 ? 0 : tensor_shard_spec.shape[1] / page_shape[1];
    return {height_in_pages, width_in_pages};
}

DeviceAddr ShardSpecBuffer::num_pages() const {
    const auto shape = shape_in_pages();
    return static_cast<DeviceAddr>(shape[0]) * static_cast<DeviceAddr>(shape[1]);
}

uint32_t Allocator::get_alignment(BufferType) const { return 64; }

}  // namespace tt::tt_metal

namespace ttsl::json {

tt::tt_metal::ShardSpec from_json_t<tt::tt_metal::ShardSpec>::operator()(const nlohmann::json&) const {
    TT_THROW("ShardSpec JSON parsing is not available in the fabric-disabled host tensor overlay");
}

}  // namespace ttsl::json

namespace tt::tt_metal::distributed {

const MeshShape& MeshDeviceView::shape() const noexcept {
    static const MeshShape shape(1);
    return shape;
}

}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {

PinnedMemoryCache& PinnedMemoryCache::instance() {
    static PinnedMemoryCache cache;
    return cache;
}

std::shared_ptr<PinnedMemory> PinnedMemoryCache::try_pin(
    distributed::MeshDevice&, const distributed::MeshCoordinateRangeSet&, HostBuffer&, bool) {
    return nullptr;
}

void PinnedMemoryCache::release(const void*) {}

void PinnedMemoryCache::release_for_device(distributed::MeshDevice&) {}

size_t PinnedMemoryCache::num_entries() const { return 0; }

}  // namespace tt::tt_metal::experimental
