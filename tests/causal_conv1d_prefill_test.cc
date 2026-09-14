#include <bit>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ttnn/device.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/transformer/gated_delta_attn/causal_conv1d_prefill.hpp"
#include <tt-metalium/mesh_device.hpp>

using namespace tt::tt_metal;

int main() {
  auto device = ttnn::open_mesh_device(0);
  device->enable_program_cache();
  std::vector<ttnn::Tensor> inputs, outputs;
  std::vector<std::vector<float>> expected;
  for (auto [rows, channels] :
       {std::pair{32u, 32u}, {64u, 96u}, {256u, 8192u}}) {
    // Both invocations stay live to check cached-program address rebinding.
    for (uint32_t seed : {42u, 1337u}) {
      std::mt19937 rng(seed);
      std::normal_distribution<float> normal;
      std::vector<float> history((rows + 3) * channels), weight(channels * 4);
      for (auto &x : history)
        x = float(bfloat16(normal(rng)));
      for (auto &x : weight)
        x = float(bfloat16(normal(rng) * 0.2f));
      std::vector<float> sum(rows * channels);
      for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t channel = 0; channel < channels; ++channel) {
          for (uint32_t tap = 0; tap < 4; ++tap) {
            sum[row * channels + channel] +=
                history[(row + tap) * channels + channel] *
                weight[channel * 4 + tap];
          }
        }
      }
      auto tensor = [&](const auto &values, const ttnn::Shape &shape,
                        DataType dtype) {
        return ttnn::Tensor::from_vector(
            values,
            TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE),
                                           ttnn::DRAM_MEMORY_CONFIG)),
            device.get());
      };
      // Compute the convolution independently on the CPU, then use the existing
      // device SiLU/typecast path as the activation and rounding reference.
      auto reference =
          tensor(sum, ttnn::Shape{rows, channels}, DataType::FLOAT32);
      expected.push_back(
          ttnn::typecast(ttnn::silu(reference), DataType::BFLOAT16)
              .to_vector<float>());
      auto h =
          tensor(history, ttnn::Shape{rows + 3, channels}, DataType::BFLOAT16);
      auto w = tensor(weight, ttnn::Shape{channels, 4}, DataType::BFLOAT16);
      inputs.insert(inputs.end(), {h, w});
      auto out = ttnn::transformer::causal_conv1d_prefill(h, w);
      if (out.dtype() != DataType::FLOAT32 ||
          out.logical_shape() != ttnn::Shape{rows, channels})
        throw std::runtime_error("Unexpected prefill output type or shape");
      outputs.push_back(ttnn::typecast(out, DataType::BFLOAT16));
    }
  }
  for (size_t case_id = 0; case_id < outputs.size(); ++case_id) {
    auto actual = outputs[case_id].to_vector<float>();
    for (size_t i = 0; i < actual.size(); ++i) {
      if (std::bit_cast<uint32_t>(actual[i]) !=
          std::bit_cast<uint32_t>(expected[case_id][i])) {
        throw std::runtime_error("Prefill mismatch in case " +
                                 std::to_string(case_id) + " at element " +
                                 std::to_string(i));
      }
    }
  }
  device->close();
  std::cout << outputs.size() << " exact prefill convolution cases passed\n";
}
