#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/sub_int_sfpu.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;

  unary_op_init_common(cb_lhs, cb_out);

#if COMPARE_DIRECTION_PLACEHOLDER == 0
  sub_int_tile_init();
  eqz_tile_init();
#elif COMPARE_DIRECTION_PLACEHOLDER == 1
  sub_int_tile_init();
  nez_tile_init();
#elif COMPARE_DIRECTION_PLACEHOLDER == 2
  sub_int_tile_init();
  gez_tile_init();
#elif COMPARE_DIRECTION_PLACEHOLDER == 3
  sub_int_tile_init();
  gtz_tile_init();
#elif COMPARE_DIRECTION_PLACEHOLDER == 4
  sub_int_tile_init();
  lez_tile_init();
#elif COMPARE_DIRECTION_PLACEHOLDER == 5
  sub_int_tile_init();
  ltz_tile_init();
#endif

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_lhs, 1);
    cb_wait_front(cb_rhs, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_rhs, cb_lhs);
    copy_tile(cb_lhs, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_lhs, cb_rhs);
    copy_tile(cb_rhs, 0, 1);

#if COMPARE_DIRECTION_PLACEHOLDER == 0
    sub_int32_tile(0, 1, 0);
    eqz_tile_int32(0);
#elif COMPARE_DIRECTION_PLACEHOLDER == 1
    sub_int32_tile(0, 1, 0);
    nez_tile_int32(0);
#elif COMPARE_DIRECTION_PLACEHOLDER == 2
    sub_int32_tile(0, 1, 0);
    gez_tile_int32(0);
#elif COMPARE_DIRECTION_PLACEHOLDER == 3
    sub_int32_tile(0, 1, 0);
    gtz_tile_int32(0);
#elif COMPARE_DIRECTION_PLACEHOLDER == 4
    sub_int32_tile(0, 1, 0);
    lez_tile_int32(0);
#elif COMPARE_DIRECTION_PLACEHOLDER == 5
    sub_int32_tile(0, 1, 0);
    ltz_tile_int32(0);
#endif

    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_lhs, 1);
    cb_pop_front(cb_rhs, 1);
    cb_push_back(cb_out, 1);
  }
}
}  // namespace NAMESPACE
