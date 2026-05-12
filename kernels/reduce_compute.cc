#include <cstdint>
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/transpose_wh_dest.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
  uint32_t reduce_groups = get_arg_val<uint32_t>(0);
  uint32_t width_tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  constexpr uint32_t onetile = 1;

  unary_op_init_common(cb_input, cb_output);

  for (uint32_t group = 0; group < reduce_groups; ++group) {
    tile_regs_acquire();
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      uint32_t dst_idx = wt == 0 ? 0 : 1;
      cb_wait_front(cb_input, onetile);
      copy_tile_to_dst_init_short(cb_input);
      copy_tile(cb_input, 0, dst_idx);
      ckernel::transpose_wh_dest_init_short<true>();
      ckernel::transpose_wh_dest<true>(dst_idx);
      // The local sfpu_reduce_init wrapper drops the pool type, so call the underlying callbacks directly.
      MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::reduce, true>(
          ckernel::sfpu::_init_reduce_<REDUCE_POOL_TYPE, DataFormat::Float32>, 1)));
      MATH((_llk_math_eltwise_unary_sfpu_params_<true>(
          ckernel::sfpu::_calculate_reduce_<REDUCE_POOL_TYPE, ckernel::ReduceDim::REDUCE_COL,
                                            DataFormat::Float32>,
          dst_idx, VectorMode::RC_custom, 1)));
#if REDUCE_IS_SUM
      if (wt > 0) {
        add_binary_tile_init();
        add_binary_tile(0, dst_idx, 0);
      }
#else
      if (wt > 0) {
        ckernel::binary_max_tile_init();
        ckernel::binary_max_tile(0, dst_idx, 0);
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
