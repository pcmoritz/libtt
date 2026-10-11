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
// Each gather tile has a row for each core of a column of experts.
constexpr uint32_t max_gate_cores_per_column = 32;

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
  std::optional<Tensor> weight;
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
                 in.logits, in.weight, params.k, in.mask, in.fill),
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
  // kernel take the softmax of the k values, and writes both. With a weight,
  // gate_kc further cores for each column of 32 experts compute the logits
  // (see router_gate.cpp) and send them to those cores.
  static ProgramDescriptor create_descriptor(const RouterParams &params,
                                             const RouterInputs &in,
                                             tensor_return_value_t &outputs) {
    const auto &logits = in.logits.mesh_tensor();
    const auto &weights = outputs.at(0).mesh_tensor();
    const auto &indices = outputs.at(1).mesh_tensor();
    auto *device = in.logits.device();
    const bool fused = in.weight.has_value();
    const Tensor &scores = fused ? *in.weight : in.logits;
    const uint32_t rows = in.logits.logical_shape()[0];
    const uint32_t width = scores.logical_shape()[1];
    const uint32_t Wt = scores.padded_shape()[1] / 32;

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t group = std::min(rows, grid_cores);
    uint32_t gate_kt = 0;
    uint32_t gate_kc = 0;
    if (fused) {
      const uint32_t Kt = in.logits.padded_shape()[1] / 32;
      gate_kc = gate_cores_per_column(Kt, Wt, grid_cores - group);
      gate_kt = Kt / gate_kc;
    }
    std::vector<CoreCoord> cores;
    std::vector<CoreRange> ranges;
    for (uint32_t i = 0; i < group + Wt * gate_kc; ++i) {
      cores.emplace_back(i % grid.x, i / grid.x);
      ranges.emplace_back(cores.back(), cores.back());
    }
    auto core_set = [&](uint32_t begin, uint32_t end) {
      return CoreRangeSet(std::vector<CoreRange>(ranges.begin() + begin,
                                                 ranges.begin() + end))
          .merge_ranges();
    };
    // Every circular buffer and semaphore is on every core, so it has the
    // same address on each: cores write into each other's.
    const CoreRangeSet all_cores = core_set(0, cores.size());
    const CoreRangeSet topk_cores = core_set(0, group);
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
                       cb_exp = 4, cb_sum = 5, cb_x = 6, cb_w = 7,
                       cb_gather = 8, cb_values = 16, cb_indices = 17,
                       cb_weights = 18, cb_partial = 19, cb_sums = 20;
    const bool has_mask = in.mask.has_value();
    // The gate's logits are FP32.
    const DataType logits_dtype = fused ? DataType::FLOAT32 : in.logits.dtype();
    add_cb(cb_in, Wt, datatype_to_dataformat_converter(logits_dtype));
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
    if (fused) {
      add_cb(cb_x, gate_kt, datatype_to_dataformat_converter(in.logits.dtype()));
      add_cb(cb_w, gate_kt, datatype_to_dataformat_converter(in.weight->dtype()));
      add_cb(cb_partial, 1, tt::DataFormat::Float32);
      add_cb(cb_gather, rows, tt::DataFormat::Float32);
      add_cb(cb_sums, rows, tt::DataFormat::Float32);
    }
    constexpr uint32_t sem_id = 0, logits_sem_id = 1, gather_sem_id = 2;
    for (uint32_t id : {sem_id, logits_sem_id, gather_sem_id}) {
      desc.semaphores.push_back(SemaphoreDescriptor{
          .id = id, .core_ranges = all_cores, .initial_value = 0});
    }

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
        {"value_bytes", fused ? 4u : static_cast<uint32_t>(in.logits.element_size())},
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
        {"fused", fused ? 1u : 0u},
        {"group", group},
        {"gate_kt", gate_kt},
        {"gate_kc", gate_kc},
        {"cb_x", cb_x},
        {"cb_w", cb_w},
        {"cb_partial", cb_partial},
        {"cb_gather", cb_gather},
        {"cb_sums", cb_sums},
        {"logits_sem", logits_sem_id},
        {"gather_sem", gather_sem_id},
    };
    const std::string dir =
        "ttnn/cpp/ttnn/operations/experimental/moe_router/kernels/";
    KernelDescriptor reader;
    reader.kernel_source = dir + "router_topk.cpp";
    reader.core_ranges = topk_cores;
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
    if (fused) {
      add_gate_kernels(desc, in, named, cores, group, gate_kt, gate_kc, Wt);
    }
    return desc;
  }

  // The cores per column of experts that split K: as many as there are cores
  // for, up to 32, with at least 8 K tiles each, splitting K evenly. Fewer
  // tiles per core gain little in the matmul and take longer to dispatch.
  static uint32_t gate_cores_per_column(uint32_t Kt, uint32_t Wt,
                                        uint32_t free_cores) {
    uint32_t best = 1;
    for (uint32_t kc = 2; kc <= max_gate_cores_per_column; ++kc) {
      if (Kt % kc == 0 && Kt / kc >= 8 && kc * Wt <= free_cores) {
        best = kc;
      }
    }
    return best;
  }

  // Core group + j * gate_kc + c multiplies K tiles [c * gate_kt, (c + 1) *
  // gate_kt) of the input by the weight's column j; c = 0 owns the column.
  static void add_gate_kernels(
      ProgramDescriptor &desc, const RouterInputs &in,
      const std::unordered_map<std::string, uint32_t> &named,
      const std::vector<CoreCoord> &cores, uint32_t group, uint32_t gate_kt,
      uint32_t gate_kc, uint32_t Wt) {
    auto *device = in.logits.device();
    const auto &input = in.logits.mesh_tensor();
    const auto &weight = in.weight->mesh_tensor();
    std::vector<CoreRange> gate_ranges;
    for (uint32_t i = group; i < cores.size(); ++i) {
      gate_ranges.emplace_back(cores[i], cores[i]);
    }
    const CoreRangeSet gate_cores = CoreRangeSet(gate_ranges).merge_ranges();
    const std::string dir =
        "ttnn/cpp/ttnn/operations/experimental/moe_router/kernels/";

    KernelDescriptor reader;
    reader.kernel_source = dir + "router_gate_reader.cpp";
    reader.core_ranges = gate_cores;
    TensorAccessorArgs(input).append_to(reader.compile_time_args);
    TensorAccessorArgs(weight).append_to(reader.compile_time_args);
    reader.named_compile_time_args = {named.begin(), named.end()};
    reader.config = ReaderConfigDescriptor{};

    KernelDescriptor writer;
    writer.kernel_source = dir + "router_gate_writer.cpp";
    writer.core_ranges = gate_cores;
    writer.named_compile_time_args = {named.begin(), named.end()};
    writer.config = WriterConfigDescriptor{};

    KernelDescriptor compute;
    compute.kernel_source = dir + "router_gate.cpp";
    compute.core_ranges = gate_cores;
    compute.named_compile_time_args = {named.begin(), named.end()};
    // The fidelity ttnn's matmul picks: HiFi4 with an FP32 operand, else HiFi2.
    const bool fp32 = in.logits.dtype() == DataType::FLOAT32 ||
                      in.weight->dtype() == DataType::FLOAT32;
    ComputeConfigDescriptor config{
        .math_fidelity = fp32 ? MathFidelity::HiFi4 : MathFidelity::HiFi2,
        .fp32_dest_acc_en = true,
        .math_approx_mode = false};
    // The partials go to DST as FP32, not through the 19-bit source registers.
    config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS,
                                      UnpackToDestMode::Default);
    config.unpack_to_dest_mode[named.at("cb_gather")] =
        UnpackToDestMode::UnpackToDestFp32;
    compute.config = config;

    for (uint32_t j = 0; j < Wt; ++j) {
      const CoreCoord owner =
          device->worker_core_from_logical_core(cores[group + j * gate_kc]);
      for (uint32_t c = 0; c < gate_kc; ++c) {
        const CoreCoord core = cores[group + j * gate_kc + c];
        reader.emplace_runtime_args(
            core,
            std::vector<std::variant<uint32_t, std::reference_wrapper<const MeshTensor>>>{
                input, weight, c * gate_kt, j});
        std::vector<uint32_t> writer_args = {
            static_cast<uint32_t>(owner.x), static_cast<uint32_t>(owner.y), c, j};
        if (c == 0) {
          for (uint32_t i = 0; i < group; ++i) {
            const CoreCoord topk = device->worker_core_from_logical_core(cores[i]);
            writer_args.push_back(topk.x);
            writer_args.push_back(topk.y);
          }
        }
        writer.runtime_args.emplace_back(core, std::move(writer_args));
        compute.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{c == 0 ? 1u : 0u});
      }
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
  }
};

} // namespace
} // namespace ttnn::prim

namespace ttnn::experimental {

bool moe_router_topk_supported(const Tensor &logits,
                               const std::optional<Tensor> &weight, uint32_t k,
                               const std::optional<Tensor> &mask,
                               const std::optional<Tensor> &fill) {
  using prim::max_k;
  using prim::max_rows;
  using prim::max_width;
  const auto &shape = logits.logical_shape();
  if (!prim::standard_tiles(logits) || shape.rank() != 2 || shape[0] < 1 ||
      shape[0] > max_rows || k < 1 || k > max_k ||
      (logits.dtype() != DataType::FLOAT32 && logits.dtype() != DataType::BFLOAT16)) {
    return false;
  }
  uint32_t width = shape[1];
  if (weight) {
    // Whole K tiles: the padding of the input and the weight need not be zero.
    const auto &weight_shape = weight->logical_shape();
    const DataType dtype = weight->dtype();
    if (!prim::standard_tiles(*weight) || weight_shape.rank() != 2 ||
        weight_shape[0] != shape[1] || shape[1] % 32 != 0 ||
        weight->device() != logits.device() ||
        (dtype != DataType::FLOAT32 && dtype != DataType::BFLOAT16 &&
         dtype != DataType::BFLOAT8_B && dtype != DataType::BFLOAT4_B)) {
      return false;
    }
    width = weight_shape[1];
  }
  if (width < k || width > max_width) {
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

std::vector<Tensor> moe_router_topk(const Tensor &logits,
                                    const std::optional<Tensor> &weight,
                                    uint32_t k,
                                    tt::tt_metal::DataType weights_dtype,
                                    tt::tt_metal::DataType indices_dtype,
                                    const std::optional<Tensor> &mask,
                                    const std::optional<Tensor> &fill) {
  return device_operation::launch<prim::RouterOperation>(
      prim::RouterParams{.k = k,
                         .weights_dtype = weights_dtype,
                         .indices_dtype = indices_dtype},
      prim::RouterInputs{
          .logits = logits, .weight = weight, .mask = mask, .fill = fill});
}

} // namespace ttnn::experimental
