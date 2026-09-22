#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/mesh_partition/mesh_partition.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute_utils.hpp"
#include "ttnn/operations/experimental/conv3d/conv3d.hpp"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward.hpp"
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

ttnn::Tensor mesh_partition(
    const ttnn::Tensor &, int32_t, std::optional<uint32_t>,
    const std::optional<tt::tt_metal::MemoryConfig> &) {
  unsupported("ttnn::mesh_partition");
}

void ring_attention_all_gather_async_multi_core_with_workers_helper(
    tt::tt_metal::ProgramDescriptor &, const std::vector<Tensor> &,
    const MeshCoordinate &, std::optional<MeshCoordinate>,
    std::optional<MeshCoordinate>, std::vector<Tensor> &, int32_t, uint32_t,
    uint32_t, uint32_t, ttnn::ccl::Topology,
    const std::vector<GlobalSemaphore> &,
    const std::optional<tt::tt_metal::SubDeviceId> &,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> &,
    CoreCoord, ttnn::ccl::CoreAllocationStrategy, std::optional<uint32_t>,
    std::optional<uint32_t>, std::optional<Tensor>, std::optional<Tensor>,
    uint32_t, uint32_t, uint32_t) {
  unsupported("ttnn::ring_attention_all_gather_async");
}

} // namespace ttnn

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

CoreCoord get_moe_tilize_drain_core(MeshDevice *, uint32_t, uint32_t,
                                    uint32_t, const CoreRangeSet &) {
  unsupported("ttnn::experimental::get_moe_tilize_drain_core");
}

WeightMemoryConfigs get_weight_mem_configs(MeshDevice *, uint32_t, uint32_t,
                                           uint32_t, uint32_t, bool) {
  unsupported("ttnn::experimental::get_weight_mem_configs");
}

Tensor prepare_w0_w1_tensor_for_moe_compute(const Tensor &, const Tensor &,
                                            uint32_t, uint32_t, uint32_t,
                                            uint32_t) {
  unsupported("ttnn::experimental::prepare_w0_w1_tensor_for_moe_compute");
}

Tensor prepare_w2_tensor_for_moe_compute(const Tensor &, uint32_t, uint32_t,
                                         uint32_t, uint32_t) {
  unsupported("ttnn::experimental::prepare_w2_tensor_for_moe_compute");
}

Tensor prepare_w0_w1_tensor_with_bias(const Tensor &, const Tensor &,
                                      const Tensor &, const Tensor &, uint32_t,
                                      uint32_t, uint32_t, uint32_t) {
  unsupported("ttnn::experimental::prepare_w0_w1_tensor_with_bias");
}

Tensor prepare_w2_tensor_with_bias(const Tensor &, const Tensor &, uint32_t,
                                   uint32_t, uint32_t, uint32_t) {
  unsupported("ttnn::experimental::prepare_w2_tensor_with_bias");
}

Tensor quantize_weights_via_host(const Tensor &, DataType,
                                 const std::optional<MemoryConfig> &) {
  unsupported("ttnn::experimental::quantize_weights_via_host");
}

} // namespace ttnn::experimental


namespace ttnn {

void dump_tensor_flatbuffer(const std::string &, const Tensor &,
                            DumpTensorMode) {
  unsupported("ttnn::dump_tensor_flatbuffer");
}

Tensor load_tensor_flatbuffer(
    const std::string &, tt::tt_metal::distributed::MeshDevice *) {
  unsupported("ttnn::load_tensor_flatbuffer");
}

} // namespace ttnn
