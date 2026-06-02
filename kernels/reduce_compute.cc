#include <cstdint>
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reduce_custom.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh_dest.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

void add_reduce_init() {
  if constexpr (REDUCE_DATA_FORMAT == DataFormat::Float16 ||
                REDUCE_DATA_FORMAT == DataFormat::Float16_b ||
                REDUCE_DATA_FORMAT == DataFormat::Float32) {
    add_binary_tile_init();
  } else {
    ckernel::add_int_tile_init();
  }
}

void add_reduce_tile(uint32_t lhs, uint32_t rhs, uint32_t output) {
  if constexpr (REDUCE_DATA_FORMAT == DataFormat::Float16 ||
                REDUCE_DATA_FORMAT == DataFormat::Float16_b ||
                REDUCE_DATA_FORMAT == DataFormat::Float32) {
    add_binary_tile(lhs, rhs, output);
  } else if constexpr (REDUCE_DATA_FORMAT == DataFormat::Int32) {
    ckernel::add_int32_tile(lhs, rhs, output);
  } else if constexpr (REDUCE_DATA_FORMAT == DataFormat::UInt32) {
    ckernel::add_uint32_tile(lhs, rhs, output);
  } else if constexpr (REDUCE_DATA_FORMAT == DataFormat::UInt16) {
    ckernel::add_uint16_tile(lhs, rhs, output);
  }
}

void min_reduce_tile(uint32_t lhs, uint32_t rhs, uint32_t output) {
  ckernel::binary_min_tile_init();
  if constexpr (REDUCE_DATA_FORMAT == DataFormat::Int32) {
    ckernel::binary_min_int32_tile(lhs, rhs, output);
  } else {
    ckernel::binary_min_tile(lhs, rhs, output);
  }
}

void max_reduce_tile(uint32_t lhs, uint32_t rhs, uint32_t output) {
  ckernel::binary_max_tile_init();
  if constexpr (REDUCE_DATA_FORMAT == DataFormat::Int32) {
    ckernel::binary_max_int32_tile(lhs, rhs, output);
  } else {
    ckernel::binary_max_tile(lhs, rhs, output);
  }
}

#if !REDUCE_IS_BITWISE
void reduce_last_dim_tile(uint32_t dst_idx) {
  ckernel::transpose_wh_dest_init_short<true>();
  ckernel::transpose_wh_dest<true>(dst_idx);
  // This tt-metal snapshot's sfpu_reduce_init wrapper does not compile for
  // SUM/Float32, so keep the equivalent lower-level init and use sfpu_reduce.
  MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::reduce, true>(
      ckernel::sfpu::_init_reduce_<REDUCE_POOL_TYPE, REDUCE_DATA_FORMAT>, 1)));
  ckernel::sfpu_reduce<REDUCE_POOL_TYPE, REDUCE_DATA_FORMAT>(dst_idx);
}
#endif

void bitwise_reduce_tile(uint32_t lhs, uint32_t rhs, uint32_t output) {
  ckernel::binary_bitwise_tile_init();
#if REDUCE_IS_OR
  if constexpr (REDUCE_DATA_FORMAT == DataFormat::UInt16) {
    ckernel::bitwise_or_uint16_binary_tile(lhs, rhs, output);
  } else {
    ckernel::bitwise_or_uint32_binary_tile(lhs, rhs, output);
  }
#else
  if constexpr (REDUCE_DATA_FORMAT == DataFormat::UInt16) {
    ckernel::bitwise_and_uint16_binary_tile(lhs, rhs, output);
  } else {
    ckernel::bitwise_and_uint32_binary_tile(lhs, rhs, output);
  }
#endif
}

void combine_into_accumulator(uint32_t index, uint32_t dst_idx) {
  if (index == 0) {
    return;
  }
#if REDUCE_IS_BITWISE
  bitwise_reduce_tile(0, dst_idx, 0);
#elif REDUCE_IS_SUM
  add_reduce_init();
  add_reduce_tile(0, dst_idx, 0);
#elif REDUCE_IS_MIN
  min_reduce_tile(0, dst_idx, 0);
#else
  max_reduce_tile(0, dst_idx, 0);
#endif
}

void square_pre_reduce_tile(uint32_t dst_idx) {
#if REDUCE_PRE_SQUARE
  mul_binary_tile_init();
  mul_binary_tile(dst_idx, dst_idx, dst_idx);
#endif
}

void MAIN {
  uint32_t reduce_groups = get_arg_val<uint32_t>(0);
  uint32_t count = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  constexpr uint32_t onetile = 1;

  unary_op_init_common(cb_input, cb_output);

#if !REDUCE_IS_BITWISE
  if constexpr (REDUCE_LAST_DIM_TILED) {
    uint32_t width_tiles = count;
#if REDUCE_BLOCK_MAX_ROW
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    cb_wait_front(cb_scaler, onetile);
    ckernel::reduce_block_max_row_init<REDUCE_BLOCK_MAX_ROW_TILES>();
    for (uint32_t group = 0; group < reduce_groups; ++group) {
      tile_regs_acquire();
      uint32_t block_count =
          (width_tiles + REDUCE_BLOCK_MAX_ROW_TILES - 1) /
          REDUCE_BLOCK_MAX_ROW_TILES;
      for (uint32_t block = 0; block < block_count; ++block) {
        uint32_t dst_idx = block == 0 ? 0 : 1;
        cb_wait_front(cb_input, REDUCE_BLOCK_MAX_ROW_TILES);
        ckernel::reduce_block_max_row<REDUCE_BLOCK_MAX_ROW_TILES>(
            cb_input, cb_scaler, 0, dst_idx);
        combine_into_accumulator(block, dst_idx);
        cb_pop_front(cb_input, REDUCE_BLOCK_MAX_ROW_TILES);
      }
      cb_reserve_back(cb_output, onetile);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile(0, cb_output);
      tile_regs_release();
      cb_push_back(cb_output, onetile);
    }
    ckernel::reduce_block_max_row_uninit<DST_ACCUM_MODE>();
    cb_pop_front(cb_scaler, onetile);
    return;
#else
    for (uint32_t group = 0; group < reduce_groups; ++group) {
      tile_regs_acquire();
      for (uint32_t wt = 0; wt < width_tiles; ++wt) {
        uint32_t dst_idx = wt == 0 ? 0 : 1;
        cb_wait_front(cb_input, onetile);
        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, dst_idx);
        square_pre_reduce_tile(dst_idx);
        reduce_last_dim_tile(dst_idx);
        combine_into_accumulator(wt, dst_idx);
        cb_pop_front(cb_input, onetile);
      }
      cb_reserve_back(cb_output, onetile);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile(0, cb_output);
      tile_regs_release();
      cb_push_back(cb_output, onetile);
    }
    return;
#endif
  }
#endif

  uint32_t reduce_count = count;
  for (uint32_t group = 0; group < reduce_groups; ++group) {
    tile_regs_acquire();
    for (uint32_t index = 0; index < reduce_count; ++index) {
      uint32_t dst_idx = index == 0 ? 0 : 1;
      cb_wait_front(cb_input, onetile);
      copy_tile_to_dst_init_short(cb_input);
      copy_tile(cb_input, 0, dst_idx);
      square_pre_reduce_tile(dst_idx);
      combine_into_accumulator(index, dst_idx);
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
