#include "cpp/tt_metal_runtime_root.h"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/trace.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
  EnsureTtMetalRuntimeReady();
  if (const char* root = std::getenv("LIBTT_BENCH_RUNTIME_ROOT")) setenv("TT_METAL_RUNTIME_ROOT",root,1);
  const unsigned k = argc > 1 ? std::stoul(argv[1]) : 6144;
  const unsigned n = argc > 2 ? std::stoul(argv[2]) : 4096;
  std::optional<ttnn::DeviceComputeKernelConfig> compute;
  if (const char* mode = std::getenv("BENCH_COMPUTE")) {
    compute = ttnn::ComputeKernelConfig{.math_fidelity=tt::tt_metal::MathFidelity::HiFi2,
      .math_approx_mode=false, .fp32_dest_acc_en=std::string(mode)!="bf16", .packer_l1_acc=std::string(mode)=="packer"};
  }
  auto device = ttnn::MeshDevice::create_unit_mesh(0, 32768, 8 * 1024 * 1024);
  device->enable_program_cache();
  auto a = ttnn::Tensor::from_vector(std::vector<bfloat16>(k, bfloat16(0.125f)),
      tt::tt_metal::TensorSpec(ttnn::Shape({1,k}), tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16,
      tt::tt_metal::PageConfig(ttnn::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG)), device.get());
  auto b = ttnn::Tensor::from_vector(std::vector<float>(size_t(k)*n, 0.125f),
      tt::tt_metal::TensorSpec(ttnn::Shape({k,n}), tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT8_B,
      tt::tt_metal::PageConfig(ttnn::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG)), device.get());
  if (argc < 4) for (auto grid : {tt::tt_metal::CoreCoord{11,10}, {8,8}, {8,10}, {10,10}, {11,8}}) {
    for (unsigned block : {2,4,8,16,32}) {
      unsigned cores=grid.x*grid.y, nt=n/32, per_n=(nt+cores-1)/cores;
      unsigned sub=std::min(4u,per_n);while(per_n%sub) --sub;
      ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig cfg{
        .compute_with_storage_grid_size=grid, .in0_block_w=block,
        .out_subblock_h=1, .out_subblock_w=sub, .out_block_h=1,
        .out_block_w=per_n, .per_core_M=1, .per_core_N=per_n,
        .fuse_batch=true, .mcast_in0=true,
        .balance_per_core_N=nt>cores && nt<2*cores};
      auto y=ttnn::matmul(a,b,false,false,ttnn::DRAM_MEMORY_CONFIG,ttnn::DataType::BFLOAT16,cfg,std::nullopt,compute);
      auto values=y.to_vector<bfloat16>();
      double max_error=0; for(auto v:values) max_error=std::max(max_error,double(std::abs(float(v)-k/64.f)));
      if(max_error>0.001f*(k/64.f)) { std::cout<<"INVALID "<<grid.x<<' '<<grid.y<<' '<<block<<' '<<max_error<<std::endl; continue; }
      y=ttnn::matmul(a,b,false,false,ttnn::DRAM_MEMORY_CONFIG,ttnn::DataType::BFLOAT16,cfg,
          std::nullopt,compute,std::nullopt,std::nullopt,y);
      auto trace=ttnn::operations::trace::begin_trace_capture(device.get(),ttnn::QueueId(0));
      for(int i=0;i<32;i++) y=ttnn::matmul(a,b,false,false,ttnn::DRAM_MEMORY_CONFIG,ttnn::DataType::BFLOAT16,cfg,
          std::nullopt,compute,std::nullopt,std::nullopt,y);
      ttnn::operations::trace::end_trace_capture(device.get(),trace,ttnn::QueueId(0));
      ttnn::operations::trace::execute_trace(device.get(),trace,ttnn::QueueId(0),true);
      std::vector<double> samples;
      for(int i=0;i<7;i++) {
        auto start=std::chrono::steady_clock::now();
        ttnn::operations::trace::execute_trace(device.get(),trace,ttnn::QueueId(0),true);
        samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/32);
      }
      std::sort(samples.begin(),samples.end());
      std::cout<<"RESULT "<<k<<' '<<n<<' '<<grid.x<<' '<<grid.y<<' '<<block<<' '<<samples[3]<<std::endl;
      ttnn::operations::trace::release_trace(device.get(),trace);
    }
  }
  if (argc >= 4) {
    const unsigned banks = device->num_dram_channels();
    std::cout << "DRAM_BANKS " << banks << std::endl;
    auto grid = tt::tt_metal::CoreRangeSet({tt::tt_metal::CoreRange({0,0},{7,0})});
    auto dram_grid = tt::tt_metal::CoreRangeSet({tt::tt_metal::CoreRange({0,0},{banks-1,0})});
    auto amem = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::WIDTH_SHARDED,ttnn::BufferType::L1,
        tt::tt_metal::ShardSpec(grid,{32,k/8},tt::tt_metal::ShardOrientation::ROW_MAJOR));
    auto bmem = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::WIDTH_SHARDED,ttnn::BufferType::DRAM,
        tt::tt_metal::ShardSpec(dram_grid,{k,((n/32+banks-1)/banks)*32},tt::tt_metal::ShardOrientation::ROW_MAJOR));
    auto omem = ttnn::MemoryConfig(ttnn::TensorMemoryLayout::WIDTH_SHARDED,ttnn::BufferType::L1,
        tt::tt_metal::ShardSpec(grid,{32,((n/32+7)/8)*32},tt::tt_metal::ShardOrientation::ROW_MAJOR));
    auto aa=ttnn::to_memory_config(a,amem);
    auto bb=ttnn::Tensor::from_vector(std::vector<float>(size_t(k)*n,0.125f),
        tt::tt_metal::TensorSpec(ttnn::Shape({k,n}),tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT8_B,
        tt::tt_metal::PageConfig(ttnn::Layout::TILE),bmem)),device.get());
    for(unsigned block : {2,4,8,12,24}) {
      ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig cfg{
          .in0_block_w=block,.per_core_M=1,.per_core_N=(n/32+7)/8};
      auto y=ttnn::matmul(aa,bb,false,false,omem,ttnn::DataType::BFLOAT16,cfg,std::nullopt,compute);
      auto host=ttnn::to_memory_config(y,ttnn::DRAM_MEMORY_CONFIG).to_vector<bfloat16>();
      double max_error=0; for(auto v:host) max_error=std::max(max_error,double(std::abs(float(v)-k/64.f)));
      if(max_error>0.001f*(k/64.f)) { std::cout<<"DRAM_INVALID "<<block<<' '<<max_error<<std::endl; continue; }
      y=ttnn::matmul(aa,bb,false,false,omem,ttnn::DataType::BFLOAT16,cfg,
          std::nullopt,compute,std::nullopt,std::nullopt,y);
      auto trace=ttnn::operations::trace::begin_trace_capture(device.get(),ttnn::QueueId(0));
      for(int i=0;i<32;i++) y=ttnn::matmul(aa,bb,false,false,omem,ttnn::DataType::BFLOAT16,cfg,
          std::nullopt,compute,std::nullopt,std::nullopt,y);
      ttnn::operations::trace::end_trace_capture(device.get(),trace,ttnn::QueueId(0));
      ttnn::operations::trace::execute_trace(device.get(),trace,ttnn::QueueId(0),true);
      std::vector<double> samples;
      for(int i=0;i<7;i++) {
        auto start=std::chrono::steady_clock::now();
        ttnn::operations::trace::execute_trace(device.get(),trace,ttnn::QueueId(0),true);
        samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/32);
      }
      std::sort(samples.begin(),samples.end());
      std::cout<<"DRAM_RESULT "<<k<<' '<<n<<' '<<block<<' '<<samples[3]<<std::endl;
      ttnn::operations::trace::release_trace(device.get(),trace);
    }
  }
  device->close();
}
