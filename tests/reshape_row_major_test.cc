#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_device.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

// Keep both inputs and outputs alive so cached programs must rebind addresses.
template <typename T>
void CheckReshape(distributed::MeshDevice* device, DataType dtype,
                  const ttnn::Shape& from, const ttnn::Shape& to) {
  size_t count = 1;
  for (auto size : from.view()) count *= size;
  std::vector<ttnn::Tensor> tensors;
  for (int iteration = 0; iteration < 2; ++iteration) {
    std::vector<T> values(count);
    for (size_t i = 0; i < count; ++i) {
      const int value = int((i + iteration * 37) % 1009) - 504;
      if constexpr (std::is_floating_point_v<T>) {
        values[i] = float(value) / 1009;
        if (dtype == DataType::BFLOAT16) values[i] = float(bfloat16(values[i]));
      } else {
        values[i] = T(value);
      }
    }
    tensors.push_back(ttnn::Tensor::from_vector(
        values, TensorSpec(from, TensorLayout(dtype, PageConfig(Layout::ROW_MAJOR),
                                              ttnn::DRAM_MEMORY_CONFIG)))
                          .to_device(device));
    tensors.push_back(ttnn::reshape(tensors.back(), to));
    if (tensors.back().template to_vector<T>() != values) {
      throw std::runtime_error("Row-major reshape changed values");
    }
    tensors.push_back(ttnn::reshape(tensors.back(), from));
    if (tensors.back().template to_vector<T>() != values) {
      throw std::runtime_error("Row-major reshape round trip changed values");
    }
  }
}

int main() {
  auto device = ttnn::open_mesh_device(0);
  device->enable_program_cache();
  const std::vector<std::pair<ttnn::Shape, ttnn::Shape>> shapes = {
      {ttnn::Shape{3, 32, 128, 128}, ttnn::Shape{3, 524288}},  // Rows exceed L1.
      {ttnn::Shape{3, 8192, 3}, ttnn::Shape{3, 24576}},       // Convolution history.
      {ttnn::Shape{5, 7}, ttnn::Shape{7, 5}},                // Unaligned rows.
      {ttnn::Shape{3, 2050}, ttnn::Shape{2, 3075}},          // Split row boundaries.
  };
  for (const auto& [from, to] : shapes) {
    CheckReshape<float>(device.get(), DataType::FLOAT32, from, to);
    CheckReshape<float>(device.get(), DataType::BFLOAT16, from, to);
    CheckReshape<uint32_t>(device.get(), DataType::UINT32, from, to);
    CheckReshape<int32_t>(device.get(), DataType::INT32, from, to);
    CheckReshape<uint16_t>(device.get(), DataType::UINT16, from, to);
  }
  device->close();
  std::cout << "80 exact reshape/round-trip checks passed\n";
}
