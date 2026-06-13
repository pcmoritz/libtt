#include <tt-metalium/experimental/tensor/spec/layout/alignment.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <ttnn/tensor/tensor.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

#define CHECK(condition)                                             \
  do {                                                               \
    if (!(condition)) {                                               \
      std::cerr << __FILE__ << ":" << __LINE__ << ": check failed: " \
                << #condition << "\n";                               \
      return 1;                                                       \
    }                                                                \
  } while (false)

int main() {
  tt::tt_metal::Shape logical_shape({2, 3});
  ttnn::TensorSpec spec(
      logical_shape,
      tt::tt_metal::TensorLayout(
          tt::tt_metal::DataType::FLOAT32,
          tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
          tt::tt_metal::MemoryConfig{},
          tt::tt_metal::Alignment({32, 32})));

  std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  ttnn::Tensor tensor = ttnn::Tensor::from_vector(std::move(values), spec);

  CHECK(tensor.logical_shape() == logical_shape);
  CHECK(tensor.physical_volume() >= 6);
  CHECK(tensor.element_size() == sizeof(float));

  std::vector<float> roundtrip = tensor.to_vector<float>();
  CHECK(roundtrip == std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  return 0;
}
