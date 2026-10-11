// SPDX-FileCopyrightText: © 2026 libtt authors
// SPDX-License-Identifier: Apache-2.0
#include "ttnn/operations/experimental/moe_router/moe_router_topk.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {

constexpr uint32_t max_rows = 32;
constexpr uint32_t max_width = 512;
constexpr uint32_t max_k = 32;

bool standard_tiles(const Tensor &tensor) {
  return tensor.storage_type() == StorageType::DEVICE &&
         tensor.buffer() != nullptr && tensor.layout() == Layout::TILE &&
         tensor.tensor_spec().tile() == Tile() &&
         !tensor.memory_config().is_sharded();
}

struct RouterParams {
  uint32_t k = 0;
  DataType weights_dtype = DataType::FLOAT32;
  DataType indices_dtype = DataType::INT32;
};

struct RouterInputs {
  Tensor logits;
  std::optional<Tensor> mask;
  std::optional<Tensor> fill;
};

struct RouterOperation {
  using operation_attributes_t = RouterParams;
  using tensor_args_t = RouterInputs;
  using spec_return_value_t = std::vector<TensorSpec>;
  using tensor_return_value_t = std::vector<Tensor>;

  static void validate_on_program_cache_miss(const RouterParams &params,
                                             const RouterInputs &in) {
    TT_FATAL(ttnn::experimental::moe_router_topk_supported(
                 in.logits, params.k, in.mask, in.fill),
             "Unsupported moe_router_topk inputs");
    TT_FATAL(params.weights_dtype == DataType::FLOAT32 ||
                 params.weights_dtype == DataType::BFLOAT16,
             "The weights must be FLOAT32 or BFLOAT16");
    TT_FATAL(params.indices_dtype == DataType::INT32 ||
                 params.indices_dtype == DataType::UINT32,
             "The indices must be INT32 or UINT32");
  }

  static spec_return_value_t compute_output_specs(const RouterParams &params,
                                                  const RouterInputs &in) {
    const Shape shape({in.logits.logical_shape()[0], params.k});
    auto spec = [&](DataType dtype) {
      return TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE),
                                            DRAM_MEMORY_CONFIG));
    };
    return {spec(params.weights_dtype), spec(params.indices_dtype)};
  }

  static tensor_return_value_t create_output_tensors(const RouterParams &params,
                                                     const RouterInputs &in) {
    tensor_return_value_t outputs;
    for (const auto &spec : compute_output_specs(params, in)) {
      outputs.push_back(create_device_tensor(spec, in.logits.device()));
    }
    return outputs;
  }

  // Each of up to 32 cores takes the rows i, i + cores, ... of the single
  // tile row, selects their top k with integer compares and copies them into
  // the first core's tiles. That core masks the indices, has its compute
  // kernel take the softmax of the k values, and writes both.
  static ProgramDescriptor create_descriptor(const RouterParams &params,
                                             const RouterInputs &in,
                                             tensor_return_value_t &outputs) {
    const auto &logits = in.logits.mesh_tensor();
    const auto &weights = outputs.at(0).mesh_tensor();
    const auto &indices = outputs.at(1).mesh_tensor();
    auto *device = in.logits.device();
    const uint32_t rows = in.logits.logical_shape()[0];
    const uint32_t width = in.logits.logical_shape()[1];
    const uint32_t Wt = in.logits.padded_shape()[1] / 32;

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t group = std::min(rows, static_cast<uint32_t>(grid.x * grid.y));
    std::vector<CoreCoord> cores;
    std::vector<CoreRange> ranges;
    for (uint32_t i = 0; i < group; ++i) {
      cores.emplace_back(i % grid.x, i / grid.x);
      ranges.emplace_back(cores.back(), cores.back());
    }
    const CoreRangeSet all_cores = CoreRangeSet(ranges).merge_ranges();
    const CoreRangeSet leader_core{CoreRange(cores[0], cores[0])};

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t index, uint32_t tiles, tt::DataFormat format) {
      const uint32_t page_size = tt::tile_size(format);
      desc.cbs.push_back(CBDescriptor{
          .total_size = tiles * page_size,
          .core_ranges = all_cores,
          .format_descriptors = {{CBFormatDescriptor{
              .buffer_index = static_cast<uint8_t>(index),
              .data_format = format,
              .page_size = page_size}}}});
    };
    constexpr uint32_t cb_in = 0, cb_mask = 1, cb_fill = 2, cb_scaler = 3,
                       cb_exp = 4, cb_sum = 5, cb_values = 16, cb_indices = 17,
                       cb_weights = 18;
    const bool has_mask = in.mask.has_value();
    add_cb(cb_in, Wt, datatype_to_dataformat_converter(in.logits.dtype()));
    add_cb(cb_values, 1, tt::DataFormat::Float32);
    add_cb(cb_indices, 1,
           datatype_to_dataformat_converter(params.indices_dtype));
    add_cb(cb_scaler, 1, tt::DataFormat::Float32);
    add_cb(cb_exp, 1, tt::DataFormat::Float32);
    add_cb(cb_sum, 1, tt::DataFormat::Float32);
    add_cb(cb_weights, 1,
           datatype_to_dataformat_converter(params.weights_dtype));
    if (has_mask) {
      add_cb(cb_mask, 1, datatype_to_dataformat_converter(in.mask->dtype()));
      add_cb(cb_fill, 1, datatype_to_dataformat_converter(in.fill->dtype()));
    }
    constexpr uint32_t sem_id = 0;
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = sem_id, .core_ranges = all_cores, .initial_value = 0});

    std::vector<uint32_t> compile_args;
    TensorAccessorArgs(logits).append_to(compile_args);
    TensorAccessorArgs(weights).append_to(compile_args);
    TensorAccessorArgs(indices).append_to(compile_args);
    // The kernel always declares the mask's and the fill's accessors; without
    // a mask, the logits' stand in.
    TensorAccessorArgs(has_mask ? in.mask->mesh_tensor() : logits).append_to(compile_args);
    TensorAccessorArgs(has_mask ? in.fill->mesh_tensor() : logits).append_to(compile_args);
    const DataType mask_dtype = has_mask ? in.mask->dtype() : DataType::INT32;
    const std::unordered_map<std::string, uint32_t> named = {
        {"Wt", Wt},
        {"k", params.k},
        {"width", width},
        {"rows", rows},
        {"value_bytes", static_cast<uint32_t>(in.logits.element_size())},
        {"index_bytes", 4u},
        {"weight_bytes", params.weights_dtype == DataType::FLOAT32 ? 4u : 2u},
        {"has_mask", has_mask ? 1u : 0u},
        {"mask_rows", has_mask ? in.mask->logical_shape()[0] : 1u},
        {"mask_bytes", has_mask ? static_cast<uint32_t>(in.mask->element_size()) : 4u},
        {"mask_float", mask_dtype == DataType::FLOAT32 || mask_dtype == DataType::BFLOAT16 ? 1u : 0u},
        {"cb_in", cb_in},
        {"cb_mask", cb_mask},
        {"cb_fill", cb_fill},
        {"cb_scaler", cb_scaler},
        {"cb_exp", cb_exp},
        {"cb_sum", cb_sum},
        {"cb_values", cb_values},
        {"cb_indices", cb_indices},
        {"cb_weights", cb_weights},
        {"sem", sem_id},
    };
    const std::string dir =
        "ttnn/cpp/ttnn/operations/experimental/moe_router/kernels/";
    KernelDescriptor reader;
    reader.kernel_source = dir + "router_topk.cpp";
    reader.core_ranges = all_cores;
    reader.compile_time_args = compile_args;
    reader.named_compile_time_args = {named.begin(), named.end()};
    reader.config = ReaderConfigDescriptor{};
    const CoreCoord leader = device->worker_core_from_logical_core(cores[0]);
    for (uint32_t i = 0; i < group; ++i) {
      std::vector<std::variant<uint32_t, std::reference_wrapper<const MeshTensor>>> args = {
          logits, weights, indices};
      if (has_mask) {
        args.emplace_back(std::cref(in.mask->mesh_tensor()));
        args.emplace_back(std::cref(in.fill->mesh_tensor()));
      } else {
        args.emplace_back(0u);
        args.emplace_back(0u);
      }
      for (uint32_t value : {i, group, static_cast<uint32_t>(leader.x),
                             static_cast<uint32_t>(leader.y)}) {
        args.emplace_back(value);
      }
      reader.emplace_runtime_args(cores[i], args);
    }
    desc.kernels.push_back(std::move(reader));

    KernelDescriptor compute;
    compute.kernel_source = dir + "router_softmax.cpp";
    compute.core_ranges = leader_core;
    compute.named_compile_time_args = {named.begin(), named.end()};
    compute.config = ComputeConfigDescriptor{.math_fidelity = MathFidelity::HiFi4,
                                             .fp32_dest_acc_en = true,
                                             .math_approx_mode = false};
    desc.kernels.push_back(std::move(compute));
    return desc;
  }
};

} // namespace
} // namespace ttnn::prim

namespace ttnn::experimental {

bool moe_router_topk_supported(const Tensor &logits, uint32_t k,
                               const std::optional<Tensor> &mask,
                               const std::optional<Tensor> &fill) {
  using prim::max_k;
  using prim::max_rows;
  using prim::max_width;
  const auto &shape = logits.logical_shape();
  if (!prim::standard_tiles(logits) || shape.rank() != 2 || shape[0] < 1 ||
      shape[0] > max_rows || shape[1] < k || shape[1] > max_width || k < 1 ||
      k > max_k ||
      (logits.dtype() != DataType::FLOAT32 && logits.dtype() != DataType::BFLOAT16)) {
    return false;
  }
  if (mask.has_value() != fill.has_value()) {
    return false;
  }
  if (mask) {
    const auto &mask_shape = mask->logical_shape();
    if (!prim::standard_tiles(*mask) || !prim::standard_tiles(*fill) ||
        mask_shape.rank() != 2 || mask_shape[1] != 1 ||
        (mask_shape[0] != 1 && mask_shape[0] != shape[0]) ||
        fill->logical_shape().volume() != 1 ||
        (fill->dtype() != DataType::INT32 && fill->dtype() != DataType::UINT32) ||
        mask->element_size() < 2) {
      return false;
    }
  }
  return true;
}

std::vector<Tensor> moe_router_topk(const Tensor &logits, uint32_t k,
                                    tt::tt_metal::DataType weights_dtype,
                                    tt::tt_metal::DataType indices_dtype,
                                    const std::optional<Tensor> &mask,
                                    const std::optional<Tensor> &fill) {
  return device_operation::launch<prim::RouterOperation>(
      prim::RouterParams{.k = k,
                         .weights_dtype = weights_dtype,
                         .indices_dtype = indices_dtype},
      prim::RouterInputs{.logits = logits, .mask = mask, .fill = fill});
}

} // namespace ttnn::experimental
