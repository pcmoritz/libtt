#include "operations/ccl/aggregate_tensor.h"
#include "operations/ccl/all_gather.h"
#include "operations/ccl/all_reduce.h"
#include "operations/ccl/all_reduce_async.h"
#include "operations/ccl/all_to_all_combine.h"
#include "operations/ccl/all_to_all_dispatch.h"
#include "operations/ccl/all_to_all_dispatch_metadata.h"
#include "operations/ccl/distribute_tensor.h"
#include "operations/ccl/mesh_partition.h"
#include "operations/ccl/moe_expert_token_remap.h"
#include "operations/ccl/moe_compute.h"
#include "operations/ccl/moe_gpt.h"
#include "operations/ccl/point_to_point.h"
#include "operations/ccl/prepare_moe_compute_w0_w1_weights.h"
#include "operations/ccl/prepare_moe_compute_w2_weights.h"
#include "operations/ccl/reduce_scatter.h"
#include "operations/ccl/selective_reduce_combine.h"
#include "operations/cpu/cpu.h"
#include "operations/experimental/gelu_bw.h"
#include "operations/kv_cache/update_cache.h"
#include "operations/normalization/distributed_rms_norm.h"
#include "operations/normalization/layer_norm_post_all_gather.h"
#include "operations/normalization/layer_norm_pre_all_gather.h"
#include "operations/normalization/rms_norm_pre_all_gather.h"
#include "operations/pool/pool2d.h"
#include "operations/pool/upsample.h"
#include "operations/rand/rand.h"
#include "operations/tensor_serialization/dump_tensor.h"
#include "operations/tensor_serialization/load_tensor.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void unsupported(const char *op_name) {
  throw std::runtime_error(std::string(op_name) +
                           " is not linked in this libtt build");
}

} // namespace

namespace tt::runtime::ttnn::operations::ccl {

void run(const ::tt::target::ttnn::AggregateTensorOp *, ProgramContext &) {
  unsupported("ttnn.aggregate_tensor");
}

void run(const ::tt::target::ttnn::AllGatherOp *, ProgramContext &) {
  unsupported("ttnn.all_gather");
}

void run(const ::tt::target::ttnn::AllReduceOp *, ProgramContext &) {
  unsupported("ttnn.all_reduce");
}

void run(const ::tt::target::ttnn::AllReduceAsyncOp *, ProgramContext &) {
  unsupported("ttnn.all_reduce_async");
}

void run(const ::tt::target::ttnn::AllToAllCombineOp *, ProgramContext &) {
  unsupported("ttnn.all_to_all_combine");
}

void run(const ::tt::target::ttnn::AllToAllDispatchOp *, ProgramContext &) {
  unsupported("ttnn.all_to_all_dispatch");
}

void run(const ::tt::target::ttnn::AllToAllDispatchMetadataOp *,
         ProgramContext &) {
  unsupported("ttnn.all_to_all_dispatch_metadata");
}

void run(const ::tt::target::ttnn::DistributeTensorOp *, ProgramContext &) {
  unsupported("ttnn.distribute_tensor");
}

void run(const ::tt::target::ttnn::MeshPartitionOp *, ProgramContext &) {
  unsupported("ttnn.mesh_partition");
}

void run(const ::tt::target::ttnn::MoeExpertTokenRemapOp *, ProgramContext &) {
  unsupported("ttnn.moe_expert_token_remap");
}

void run(const ::tt::target::ttnn::MoeComputeOp *, ProgramContext &) {
  unsupported("ttnn.moe_compute");
}

void run(const ::tt::target::ttnn::MoeGptOp *, ProgramContext &) {
  unsupported("ttnn.moe_gpt");
}

void run(const ::tt::target::ttnn::PointToPointOp *, ProgramContext &) {
  unsupported("ttnn.point_to_point");
}

void run(const ::tt::target::ttnn::PrepareMoEComputeW0W1WeightsOp *,
         ProgramContext &) {
  unsupported("ttnn.prepare_moe_compute_w0_w1_weights");
}

void run(const ::tt::target::ttnn::PrepareMoEComputeW2WeightsOp *,
         ProgramContext &) {
  unsupported("ttnn.prepare_moe_compute_w2_weights");
}

void run(const ::tt::target::ttnn::ReduceScatterOp *, ProgramContext &) {
  unsupported("ttnn.reduce_scatter");
}

void run(const ::tt::target::ttnn::SelectiveReduceCombineOp *,
         ProgramContext &) {
  unsupported("ttnn.selective_reduce_combine");
}

} // namespace tt::runtime::ttnn::operations::ccl

namespace tt::runtime::ttnn::operations::cpu {

void run(const ::tt::target::ttnn::CpuOp *, ProgramContext &) {
  unsupported("ttnn.cpu");
}

std::vector<::tt::runtime::Tensor>
invokeCpuOp(ProgramContext &, const ::tt::target::ttnn::CpuOp *,
            const std::vector<::tt::runtime::Tensor> &) {
  unsupported("ttnn.cpu");
}

} // namespace tt::runtime::ttnn::operations::cpu

namespace tt::runtime::ttnn::operations::experimental {

void run(const ::tt::target::ttnn::ExperimentalEltwiseBinaryBackwardOp *,
         ProgramContext &) {
  unsupported("ttnn.experimental_eltwise_binary_backward");
}

} // namespace tt::runtime::ttnn::operations::experimental

namespace tt::runtime::ttnn::operations::kv_cache {

void run(const ::tt::target::ttnn::UpdateCacheOp *, ProgramContext &) {
  unsupported("ttnn.update_cache");
}

} // namespace tt::runtime::ttnn::operations::kv_cache

namespace tt::runtime::ttnn::operations::distributed_rms_norm {

void run(const ::tt::target::ttnn::DistributedRMSNormOp *, ProgramContext &) {
  unsupported("ttnn.distributed_rms_norm");
}

} // namespace tt::runtime::ttnn::operations::distributed_rms_norm

namespace tt::runtime::ttnn::operations::layer_norm_post_all_gather {

void run(const ::tt::target::ttnn::LayerNormPostAllGatherOp *,
         ProgramContext &) {
  unsupported("ttnn.layer_norm_post_all_gather");
}

} // namespace tt::runtime::ttnn::operations::layer_norm_post_all_gather

namespace tt::runtime::ttnn::operations::layer_norm_pre_all_gather {

void run(const ::tt::target::ttnn::LayerNormPreAllGatherOp *,
         ProgramContext &) {
  unsupported("ttnn.layer_norm_pre_all_gather");
}

} // namespace tt::runtime::ttnn::operations::layer_norm_pre_all_gather

namespace tt::runtime::ttnn::operations::rms_norm_pre_all_gather {

void run(const ::tt::target::ttnn::RMSNormPreAllGatherOp *, ProgramContext &) {
  unsupported("ttnn.rms_norm_pre_all_gather");
}

} // namespace tt::runtime::ttnn::operations::rms_norm_pre_all_gather

namespace tt::runtime::ttnn::operations::pool {

void run(const ::tt::target::ttnn::Pool2dOp *, ProgramContext &) {
  unsupported("ttnn.pool2d");
}

void run(const ::tt::target::ttnn::MaxPool2dWithIndicesOp *,
         ProgramContext &) {
  unsupported("ttnn.max_pool2d_with_indices");
}

void run(const ::tt::target::ttnn::GlobalAvgPool2dOp *, ProgramContext &) {
  unsupported("ttnn.global_avg_pool2d");
}

void run(const ::tt::target::ttnn::UpsampleOp *, ProgramContext &) {
  unsupported("ttnn.upsample");
}

} // namespace tt::runtime::ttnn::operations::pool

namespace tt::runtime::ttnn::operations::rand {

void run(const ::tt::target::ttnn::RandOp *, ProgramContext &) {
  unsupported("ttnn.rand");
}

} // namespace tt::runtime::ttnn::operations::rand

namespace tt::runtime::ttnn::operations::tensor_serialization {

void run(const ::tt::target::ttnn::DumpTensorOp *, ProgramContext &) {
  unsupported("ttnn.dump_tensor");
}

void run(const ::tt::target::ttnn::LoadTensorOp *, ProgramContext &) {
  unsupported("ttnn.load_tensor");
}

} // namespace tt::runtime::ttnn::operations::tensor_serialization
