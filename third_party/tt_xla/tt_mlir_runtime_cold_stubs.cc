#include "operations/ccl/all_gather.h"
#include "operations/ccl/all_reduce.h"
#include "operations/ccl/all_reduce_async.h"
#include "operations/ccl/all_to_all_combine.h"
#include "operations/ccl/all_to_all_dispatch.h"
#include "operations/ccl/all_to_all_dispatch_metadata.h"
#include "operations/ccl/moe_expert_token_remap.h"
#include "operations/ccl/moe_gpt.h"
#include "operations/ccl/reduce_scatter.h"
#include "operations/ccl/selective_reduce_combine.h"
#include "operations/cpu/cpu.h"
#include "operations/experimental/gelu_bw.h"
#include "operations/kv_cache/update_cache.h"
#include "operations/normalization/distributed_rms_norm.h"
#include "operations/normalization/layer_norm_post_all_gather.h"
#include "operations/normalization/layer_norm_pre_all_gather.h"
#include "operations/normalization/rms_norm_pre_all_gather.h"
#include "operations/pool/upsample.h"
#include "operations/rand/rand.h"
#include "operations/tensor_serialization/dump_tensor.h"
#include "operations/tensor_serialization/load_tensor.h"
#include "operations/ttml/adamw.h"
#include "operations/ttml/cross_entropy_bw.h"
#include "operations/ttml/cross_entropy_fw.h"
#include "operations/ttml/layernorm_bw.h"
#include "operations/ttml/layernorm_fw.h"
#include "operations/ttml/rmsnorm_bw.h"
#include "operations/ttml/rmsnorm_fw.h"
#include "operations/ttml/sdpa_bw.h"
#include "operations/ttml/sdpa_fw.h"
#include "operations/ttml/softmax_backward.h"
#include "operations/ttml/swiglu_elemwise_bw.h"

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

void run(const ::tt::target::ttnn::MoeExpertTokenRemapOp *, ProgramContext &) {
  unsupported("ttnn.moe_expert_token_remap");
}

void run(const ::tt::target::ttnn::MoeGptOp *, ProgramContext &) {
  unsupported("ttnn.moe_gpt");
}

void run(const ::tt::target::ttnn::SelectiveReduceCombineOp *,
         ProgramContext &) {
  unsupported("ttnn.selective_reduce_combine");
}

} // namespace tt::runtime::ttnn::operations::ccl

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

namespace tt::runtime::ttnn::operations::ttml {

void run(const ::tt::target::ttnn::AdamWOp *, ProgramContext &) {
  unsupported("ttnn.adamw");
}

void run(const ::tt::target::ttnn::SDPAForwardOp *, ProgramContext &) {
  unsupported("ttnn.sdpa_forward");
}

void run(const ::tt::target::ttnn::SDPABackwardOp *, ProgramContext &) {
  unsupported("ttnn.sdpa_backward");
}

void run(const ::tt::target::ttnn::LayerNormForwardOp *, ProgramContext &) {
  unsupported("ttnn.layernorm_forward");
}

void run(const ::tt::target::ttnn::CrossEntropyForwardOp *, ProgramContext &) {
  unsupported("ttnn.cross_entropy_forward");
}

void run(const ::tt::target::ttnn::CrossEntropyBackwardOp *, ProgramContext &) {
  unsupported("ttnn.cross_entropy_backward");
}

void run(const ::tt::target::ttnn::LayerNormBackwardOp *, ProgramContext &) {
  unsupported("ttnn.layernorm_backward");
}

void run(const ::tt::target::ttnn::RMSNormBackwardOp *, ProgramContext &) {
  unsupported("ttnn.rmsnorm_backward");
}

void run(const ::tt::target::ttnn::RMSNormForwardOp *, ProgramContext &) {
  unsupported("ttnn.rmsnorm_forward");
}

void run(const ::tt::target::ttnn::SoftmaxBackwardOp *, ProgramContext &) {
  unsupported("ttnn.softmax_backward");
}

void run(const ::tt::target::ttnn::SwigluElemwiseBackwardOp *, ProgramContext &) {
  unsupported("ttnn.swiglu_elemwise_backward");
}

} // namespace tt::runtime::ttnn::operations::ttml
