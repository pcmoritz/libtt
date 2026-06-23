#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute_utils.hpp"
#include "ttnn/operations/experimental/conv3d/conv3d.hpp"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/tensor/serialization.hpp"

#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void unsupported(const char *op_name) {
  throw std::runtime_error(std::string(op_name) +
                           " is not linked in this libtt build");
}

} // namespace

namespace ttnn {

tt::tt_metal::Tensor mesh_partition(
    const tt::tt_metal::Tensor &, int32_t, std::optional<uint32_t>,
    const std::optional<tt::tt_metal::MemoryConfig> &) {
  unsupported("ttnn::mesh_partition");
}

Conv2dResultWithOptions conv2d(
    const Tensor &, const Tensor &, MeshDevice *, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, std::array<uint32_t, 2>,
    std::array<uint32_t, 2>,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>,
    std::array<uint32_t, 2>, uint32_t, const std::optional<const DataType> &,
    const std::optional<const Tensor> &,
    const std::optional<const Conv2dConfig> &,
    const std::optional<const DeviceComputeKernelConfig> &,
    const std::optional<const MemoryConfig> &,
    const std::optional<const Conv2dSliceConfig> &, bool, bool) {
  unsupported("ttnn::conv2d");
}

void ring_attention_all_gather_async_multi_core_with_workers_helper(
    tt::tt_metal::ProgramDescriptor &, const std::vector<Tensor> &,
    const MeshCoordinate &, std::optional<MeshCoordinate>,
    std::optional<MeshCoordinate>, std::vector<Tensor> &, int32_t, uint32_t,
    uint32_t, uint32_t, ttnn::ccl::Topology,
    const std::vector<GlobalSemaphore> &,
    const std::optional<tt::tt_metal::SubDeviceId> &,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> &,
    CoreCoord, ttnn::ccl::CoreAllocationStrategy,
    std::optional<uint32_t>) {
  unsupported("ttnn::ring_attention_all_gather_async");
}

} // namespace ttnn

namespace ttnn::operations::conv::conv2d {

ttnn::Tensor convert_conv_weight_tensor_to_grouped_layout_for_conv_transpose2d(
    const ttnn::Tensor &, uint32_t, DataType) {
  unsupported(
      "ttnn::operations::conv::conv2d::"
      "convert_conv_weight_tensor_to_grouped_layout_for_conv_transpose2d");
}

std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>>
prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor &, const std::optional<const ttnn::Tensor> &,
    Conv2dWeightsBiasPrepConfig &, MeshDevice *) {
  unsupported(
      "ttnn::operations::conv::conv2d::"
      "prepare_conv_weights_biases_and_move_to_device");
}

ttnn::Tensor prepare_conv_weights(
    const ttnn::Tensor &, const ttnn::MemoryConfig &, Layout,
    const std::string &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    std::array<uint32_t, 2>, std::array<uint32_t, 2>,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>,
    std::array<uint32_t, 2>, bool, uint32_t, MeshDevice *, DataType,
    const std::optional<const DataType> &,
    const std::optional<const Conv2dConfig> &,
    const std::optional<const DeviceComputeKernelConfig> &,
    const std::optional<const Conv2dSliceConfig> &) {
  unsupported("ttnn::prepare_conv_weights");
}

ttnn::Tensor prepare_conv_bias(
    const ttnn::Tensor &, const ttnn::MemoryConfig &, Layout, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, std::array<uint32_t, 2>,
    std::array<uint32_t, 2>,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>,
    std::array<uint32_t, 2>, uint32_t, MeshDevice *, DataType,
    const std::optional<const DataType> &,
    const std::optional<const Conv2dConfig> &,
    const std::optional<const DeviceComputeKernelConfig> &,
    const std::optional<const Conv2dSliceConfig> &) {
  unsupported("ttnn::prepare_conv_bias");
}

} // namespace ttnn::operations::conv::conv2d

namespace ttnn::prim {

ttnn::Tensor conv2d(
    const ttnn::Tensor &, const ttnn::Tensor &,
    const std::optional<const ttnn::Tensor> &,
    const ttnn::operations::sliding_window::SlidingWindowConfig &, uint32_t,
    uint32_t, bool,
    const std::optional<ttnn::operations::unary::UnaryWithParam> &,
    const Conv2dParallelizationConfig &, const Conv2dBlockConfig &,
    const tt::tt_metal::MemoryConfig &, tt::tt_metal::DataType,
    std::array<std::uint32_t, 4>, const ttnn::DeviceComputeKernelConfig &,
    bool, bool, bool, bool, bool, std::optional<bool>) {
  unsupported("ttnn::prim::conv2d");
}

std::vector<CBInfo> get_cb_info(
    const DeviceComputeKernelConfig &, const Conv2dBlockConfig &,
    const Conv2dParallelizationConfig &, const ttnn::Shape &,
    std::array<uint32_t, 2>, std::array<uint32_t, 2>,
    std::array<uint32_t, 2>, const Conv2dConfig &, DataType, DataType,
    std::array<uint32_t, 2>, uint32_t, bool, bool, bool, uint32_t,
    std::optional<uint32_t>) {
  unsupported("ttnn::prim::get_cb_info");
}

} // namespace ttnn::prim

namespace ttnn::operations::pool {

std::vector<Tensor> max_pool2d(
    const Tensor &, uint32_t, uint32_t, uint32_t, uint32_t,
    std::array<uint32_t, 2>, std::array<uint32_t, 2>,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>,
    std::array<uint32_t, 2>, bool, const std::optional<const MemoryConfig> &,
    const std::optional<Op2DSliceConfig> &,
    std::optional<const TensorMemoryLayout>, bool, bool, bool, DataType, Layout,
    bool) {
  unsupported("ttnn::max_pool2d");
}

Tensor avg_pool2d(
    const Tensor &, uint32_t, uint32_t, uint32_t, uint32_t,
    std::array<uint32_t, 2>, std::array<uint32_t, 2>,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>, bool, bool,
    std::optional<int32_t>, const std::optional<const MemoryConfig> &,
    const std::optional<Op2DSliceConfig> &,
    std::optional<const TensorMemoryLayout>,
    const std::optional<DeviceComputeKernelConfig> &, bool, bool, DataType,
    Layout, bool) {
  unsupported("ttnn::avg_pool2d");
}

} // namespace ttnn::operations::pool

namespace ttnn::operations::upsample {

ttnn::Tensor upsample(
    const ttnn::Tensor &,
    std::variant<int, std::array<int, 2>, float, std::array<float, 2>>,
    const std::string &, const std::optional<MemoryConfig> &,
    const std::optional<DeviceComputeKernelConfig> &) {
  unsupported("ttnn::upsample");
}

} // namespace ttnn::operations::upsample

namespace ttnn::experimental {

ttnn::Tensor conv3d(
    const ttnn::Tensor &, const ttnn::Tensor &, std::optional<MeshDevice *>,
    const std::optional<ttnn::Tensor> &,
    const std::optional<ttnn::experimental::prim::Conv3dConfig> &,
    DataType, uint32_t, const std::array<uint32_t, 3> &,
    const std::array<uint32_t, 3> &, const std::array<uint32_t, 3> &,
    const std::array<uint32_t, 3> &, const std::string &, uint32_t,
    const std::optional<MemoryConfig> &,
    std::optional<DeviceComputeKernelConfig>) {
  unsupported("ttnn::conv3d");
}

Tensor gelu_bw(const Tensor &, const Tensor &, const std::string &,
               const std::optional<MemoryConfig> &, std::optional<Tensor>) {
  unsupported("ttnn::experimental::gelu_bw");
}

WeightMemoryConfigs get_weight_mem_configs(MeshDevice *, uint32_t, uint32_t,
                                           uint32_t, uint32_t, bool,
                                           uint32_t) {
  unsupported("ttnn::experimental::get_weight_mem_configs");
}

Tensor prepare_w0_w1_tensor_for_moe_compute(const Tensor &, const Tensor &,
                                            uint32_t, uint32_t, uint32_t,
                                            uint32_t, uint32_t) {
  unsupported("ttnn::experimental::prepare_w0_w1_tensor_for_moe_compute");
}

Tensor prepare_w2_tensor_for_moe_compute(const Tensor &, uint32_t, uint32_t,
                                         uint32_t, uint32_t, uint32_t) {
  unsupported("ttnn::experimental::prepare_w2_tensor_for_moe_compute");
}

Tensor prepare_w0_w1_tensor_with_bias(const Tensor &, const Tensor &,
                                      const Tensor &, const Tensor &, uint32_t,
                                      uint32_t, uint32_t, uint32_t, uint32_t) {
  unsupported("ttnn::experimental::prepare_w0_w1_tensor_with_bias");
}

Tensor prepare_w2_tensor_with_bias(const Tensor &, const Tensor &, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t) {
  unsupported("ttnn::experimental::prepare_w2_tensor_with_bias");
}

Tensor quantize_weights_via_host(const Tensor &, DataType,
                                 const std::optional<MemoryConfig> &) {
  unsupported("ttnn::experimental::quantize_weights_via_host");
}

} // namespace ttnn::experimental

namespace ttnn::experimental::prim {

ttsl::hash::hash_t RingAttentionAllGatherAsyncDeviceOperation::compute_program_hash(
    const operation_attributes_t &, const tensor_args_t &) {
  return 0;
}

} // namespace ttnn::experimental::prim

namespace ttnn::experimental::ccl {

void AllGatherFusedOpSignaler::init_fused_op(
    const std::vector<CoreCoord> &receiver_cores,
    const std::vector<uint32_t> &receiver_signal_semaphores,
    FusedOpSignalerMode mode) {
  num_fused_op_cores_to_signal = receiver_cores.size();
  fused_op_receiver_cores_noc = receiver_cores;
  fused_op_receiver_signal_semaphores = receiver_signal_semaphores;
  fused_op_signaler_mode = mode;
  initialized_fused_op = true;
}

void MatmulFusedOpSignaler::init_fused_op(
    tt::tt_metal::Program &, const tt::tt_metal::IDevice *,
    const CoreRange &, const std::vector<CoreCoord> &) {
  unsupported("ttnn::MatmulFusedOpSignaler::init_fused_op");
}

void MatmulFusedOpSignaler::init_fused_op(
    tt::tt_metal::Program &, const tt::tt_metal::IDevice *,
    const std::variant<CoreRange, CoreRangeSet> &, FusedOpSignalerMode) {
  unsupported("ttnn::MatmulFusedOpSignaler::init_fused_op");
}

void MatmulFusedOpSignaler::init_llama_rs_cores_mm(
    const CoreRangeSet &, tt::tt_metal::Program &,
    const tt::tt_metal::IDevice *, int) {
  unsupported("ttnn::MatmulFusedOpSignaler::init_llama_rs_cores_mm");
}

void MatmulFusedOpSignaler::push_matmul_fused_op_rt_args(
    std::vector<uint32_t> &, bool) {
  unsupported("ttnn::MatmulFusedOpSignaler::push_matmul_fused_op_rt_args");
}

void MatmulFusedOpSignaler::push_matmul_fused_op_rt_args(
    std::vector<uint32_t> &, uint32_t, uint32_t) {
  unsupported("ttnn::MatmulFusedOpSignaler::push_matmul_fused_op_rt_args");
}

void MatmulFusedOpSignaler::push_llama_rs_rt_args_for_mm(
    std::vector<uint32_t> &, CoreCoord, tt::tt_metal::NOC,
    const tt::tt_metal::IDevice *) const {
  unsupported("ttnn::MatmulFusedOpSignaler::push_llama_rs_rt_args_for_mm");
}

bool MatmulFusedOpSignaler::is_all_gather() const {
  return fused_op_type == MatmulFusedOpSignalerType::ALL_GATHER ||
         fused_op_type == MatmulFusedOpSignalerType::LLAMA_ALL_GATHER;
}

bool MatmulFusedOpSignaler::is_reduce_scatter() const {
  return fused_op_type == MatmulFusedOpSignalerType::REDUCE_SCATTER ||
         fused_op_type == MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER;
}

} // namespace tt::tt_metal

namespace tt::tt_metal {

void dump_tensor_flatbuffer(const std::string &, const Tensor &,
                            DumpTensorMode) {
  unsupported("tt::tt_metal::dump_tensor_flatbuffer");
}

Tensor load_tensor_flatbuffer(const std::string &, distributed::MeshDevice *) {
  unsupported("tt::tt_metal::load_tensor_flatbuffer");
}

} // namespace ttnn::experimental::ccl

namespace ttnn::ccl {

tt::tt_fabric::Topology
convert_2d_to_1d_topology(tt::tt_fabric::Topology topology) {
  return topology;
}

std::tuple<size_t, size_t, bool>
get_forward_backward_configuration(size_t, size_t, Topology) {
  return {0, 0, false};
}

uint32_t get_linearized_index_from_physical_coord(
    const Tensor &, const MeshCoordinate &, const std::optional<uint32_t> &) {
  return 0;
}

std::optional<MeshCoordinate> get_physical_neighbor_from_physical_coord(
    const Tensor &, const MeshCoordinate &, int, Topology,
    const std::optional<uint32_t> &) {
  return std::nullopt;
}

} // namespace ttnn::ccl
