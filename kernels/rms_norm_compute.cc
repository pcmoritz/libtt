#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/tile_move_copy.h"
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "compute_kernel_api/reduce.h"
#undef REDUCE_DIM
#undef REDUCE_OP

namespace NAMESPACE {

constexpr uint32_t cb_input = tt::CBIndex::c_0;
constexpr uint32_t cb_weight = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_work = tt::CBIndex::c_3;
constexpr uint32_t cb_scale = tt::CBIndex::c_4;
constexpr uint32_t cb_output = tt::CBIndex::c_16;
constexpr uint32_t width_tiles = RMS_NORM_WIDTH_TILES;
constexpr uint32_t onetile = 1;
constexpr uint32_t block_tiles = 8;

uint32_t min_u32(uint32_t lhs, uint32_t rhs) { return lhs < rhs ? lhs : rhs; }

void square_input_tiles() {
  unary_op_init_common(cb_input, cb_work);

  for (uint32_t base = 0; base < width_tiles; base += block_tiles) {
    uint32_t tiles = min_u32(block_tiles, width_tiles - base);
    cb_reserve_back(cb_work, tiles);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_input, cb_input);
    mul_binary_tile_init();
    for (uint32_t i = 0; i < tiles; ++i) {
      copy_tile(cb_input, base + i, i);
      mul_binary_tile(i, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < tiles; ++i) {
      pack_tile(i, cb_work);
    }
    tile_regs_release();
    cb_push_back(cb_work, tiles);
  }
}

void reduce_squared_tiles() {
  reconfig_data_format(cb_work, cb_scaler);
  ckernel::reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(
      cb_work, cb_scaler, cb_scale);

  tile_regs_acquire();
  for (uint32_t wt = 0; wt < width_tiles; ++wt) {
    ckernel::reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, true>(
        cb_work, cb_scaler, wt, 0, 0);
  }
  ckernel::reduce_uninit<true>();

  binop_with_scalar_tile_init();
  mul_unary_tile(0, RMS_NORM_SCALE_BITS);
  add_unary_tile(0, RMS_NORM_BIAS_BITS);
  rsqrt_tile_init();
  rsqrt_tile(0);

  tile_regs_commit();
  tile_regs_wait();
  cb_reserve_back(cb_scale, onetile);
  pack_reconfig_data_format(cb_scale);
  pack_tile(0, cb_scale);
  tile_regs_release();
  cb_push_back(cb_scale, onetile);
}

void mul_tiles_bcast_rows_dest_reuse_init_short(uint32_t icb) {
  MATH((llk_math_eltwise_binary_init_with_operands<
        ELWMUL, BroadcastType::ROW, MATH_FIDELITY,
        EltwiseBinaryReuseDestType::DEST_TO_SRCA>(icb, icb)));
  UNPACK((llk_unpack_A_init<BroadcastType::ROW, true,
                            EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
      false, false, icb)));
}

void mul_tiles_bcast_rows_dest_reuse(uint32_t icb, uint32_t itile,
                                     uint32_t idst) {
  MATH((llk_math_eltwise_binary<
        ELWMUL, BroadcastType::ROW, DST_ACCUM_MODE, MATH_FIDELITY,
        EltwiseBinaryReuseDestType::DEST_TO_SRCA>(icb, icb, idst, true)));
  UNPACK((llk_unpack_A<BroadcastType::ROW, true,
                       EltwiseBinaryReuseDestType::DEST_TO_SRCA>(icb, itile)));
}

void apply_scale_and_weight() {
  reconfig_data_format(cb_input, cb_scale);
  pack_reconfig_data_format(cb_scale, cb_output);

  for (uint32_t base = 0; base < width_tiles; base += block_tiles) {
    uint32_t tiles = min_u32(block_tiles, width_tiles - base);
    cb_reserve_back(cb_output, tiles);
    tile_regs_acquire();
    mul_bcast_cols_init_short(cb_input, cb_scale);
    for (uint32_t i = 0; i < tiles; ++i) {
      mul_tiles_bcast_cols(cb_input, cb_scale, base + i, 0, i);
    }
    mul_tiles_bcast_rows_dest_reuse_init_short(cb_weight);
    for (uint32_t i = 0; i < tiles; ++i) {
      mul_tiles_bcast_rows_dest_reuse(cb_weight, base + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < tiles; ++i) {
      pack_tile(i, cb_output);
    }
    tile_regs_release();
    cb_push_back(cb_output, tiles);
  }
}

void MAIN {
  uint32_t group_count = get_arg_val<uint32_t>(0);

  cb_wait_front(cb_scaler, onetile);

  for (uint32_t group = 0; group < group_count; ++group) {
    cb_wait_front(cb_input, width_tiles);
    square_input_tiles();

    cb_wait_front(cb_work, width_tiles);
    reduce_squared_tiles();
    cb_pop_front(cb_work, width_tiles);

    cb_wait_front(cb_scale, onetile);
    cb_wait_front(cb_weight, width_tiles);
    apply_scale_and_weight();

    cb_pop_front(cb_scale, onetile);
    cb_pop_front(cb_weight, width_tiles);
    cb_pop_front(cb_input, width_tiles);
  }

  cb_pop_front(cb_scaler, onetile);
}

}  // namespace NAMESPACE
