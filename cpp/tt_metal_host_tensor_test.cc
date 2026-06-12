#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

int main() {
  tt::tt_metal::Shape logical_shape({2, 3});
  tt::tt_metal::Shape padded_shape({32, 32});
  tt::tt_metal::TensorLayout layout = tt::tt_metal::TensorLayout::fromPaddedShape(
      tt::tt_metal::DataType::FLOAT32,
      tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
      tt::tt_metal::MemoryConfig{},
      logical_shape,
      padded_shape);
  tt::tt_metal::TensorSpec spec(logical_shape, std::move(layout));

  std::vector<std::byte> bytes(32 * 32 * sizeof(float));
  bytes[0] = std::byte{0x7b};
  tt::tt_metal::HostTensor tensor(
      tt::tt_metal::HostBuffer(std::move(bytes)), std::move(spec), tt::tt_metal::TensorTopology{});

  assert(tensor.logical_shape() == logical_shape);
  assert(tensor.padded_shape() == padded_shape);

  auto shard = tensor.buffer().get_shard(tt::tt_metal::distributed::MeshCoordinate(0, 0));
  assert(shard.has_value());
  auto view = shard->view_bytes();
  assert(view.size() == 32 * 32 * sizeof(float));
  assert(view[0] == std::byte{0x7b});
  return 0;
}
