#include <cstdint>
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

template <DataFormat Format>
void add_reduce_init() {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    add_binary_tile_init();
  } else {
    ckernel::add_int_tile_init();
  }
}

template <DataFormat Format>
void add_reduce_tile(uint32_t lhs, uint32_t rhs, uint32_t output) {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    add_binary_tile(lhs, rhs, output);
  } else if constexpr (Format == DataFormat::Int32) {
    ckernel::add_int32_tile(lhs, rhs, output);
  } else if constexpr (Format == DataFormat::UInt32) {
    ckernel::add_uint32_tile(lhs, rhs, output);
  } else if constexpr (Format == DataFormat::UInt16) {
    ckernel::add_uint16_tile(lhs, rhs, output);
  }
}

template <DataFormat Format>
void min_reduce_tile(uint32_t lhs, uint32_t rhs, uint32_t output) {
  ckernel::binary_min_tile_init();
  if constexpr (Format == DataFormat::Int32) {
    ckernel::binary_min_int32_tile(lhs, rhs, output);
  } else {
    ckernel::binary_min_tile(lhs, rhs, output);
  }
}

template <DataFormat Format>
void max_reduce_tile(uint32_t lhs, uint32_t rhs, uint32_t output) {
  ckernel::binary_max_tile_init();
  if constexpr (Format == DataFormat::Int32) {
    ckernel::binary_max_int32_tile(lhs, rhs, output);
  } else {
    ckernel::binary_max_tile(lhs, rhs, output);
  }
}

void MAIN {
  uint32_t reduce_groups = get_arg_val<uint32_t>(0);
  uint32_t reduce_count = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  constexpr uint32_t onetile = 1;

  unary_op_init_common(cb_input, cb_output);

  for (uint32_t group = 0; group < reduce_groups; ++group) {
    tile_regs_acquire();
    for (uint32_t index = 0; index < reduce_count; ++index) {
      uint32_t dst_idx = index == 0 ? 0 : 1;
      cb_wait_front(cb_input, onetile);
      copy_tile_to_dst_init_short(cb_input);
      copy_tile(cb_input, 0, dst_idx);
#if REDUCE_IS_SUM
      if (index > 0) {
        add_reduce_init<REDUCE_DATA_FORMAT>();
        add_reduce_tile<REDUCE_DATA_FORMAT>(0, dst_idx, 0);
      }
#elif REDUCE_IS_MIN
      if (index > 0) {
        min_reduce_tile<REDUCE_DATA_FORMAT>(0, dst_idx, 0);
      }
#else
      if (index > 0) {
        max_reduce_tile<REDUCE_DATA_FORMAT>(0, dst_idx, 0);
      }
#endif
      cb_pop_front(cb_input, onetile);
    }
    cb_reserve_back(cb_output, onetile);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_output);
    tile_regs_release();
    cb_push_back(cb_output, onetile);
  }
}
}  // namespace NAMESPACE
