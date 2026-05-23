#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);

  constexpr uint32_t cb_gate = tt::CBIndex::c_0;
  constexpr uint32_t cb_up = tt::CBIndex::c_1;
  constexpr uint32_t cb_neg = tt::CBIndex::c_2;
  constexpr uint32_t cb_exp = tt::CBIndex::c_3;
  constexpr uint32_t cb_denom = tt::CBIndex::c_4;
  constexpr uint32_t cb_silu = tt::CBIndex::c_5;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr uint32_t one_fp32 = 0x3f800000;

  unary_op_init_common(cb_gate, cb_out);
  negative_tile_init();
  exp_tile_init();
  binop_with_scalar_tile_init();
  div_binary_tile_init();
  mul_binary_tile_init();

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_gate, 1);

    cb_reserve_back(cb_neg, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_gate);
    copy_tile(cb_gate, 0, 0);
    negative_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_neg);
    tile_regs_release();
    cb_push_back(cb_neg, 1);

    cb_wait_front(cb_neg, 1);
    cb_reserve_back(cb_exp, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_neg);
    copy_tile(cb_neg, 0, 0);
    exp_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_exp);
    cb_pop_front(cb_neg, 1);
    tile_regs_release();
    cb_push_back(cb_exp, 1);

    cb_wait_front(cb_exp, 1);
    cb_reserve_back(cb_denom, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_exp);
    copy_tile(cb_exp, 0, 0);
    add_unary_tile(0, one_fp32);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_denom);
    cb_pop_front(cb_exp, 1);
    tile_regs_release();
    cb_push_back(cb_denom, 1);

    cb_wait_front(cb_denom, 1);
    cb_reserve_back(cb_silu, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_denom, cb_gate);
    copy_tile(cb_gate, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_gate, cb_denom);
    copy_tile(cb_denom, 0, 1);
    div_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_silu);
    cb_pop_front(cb_denom, 1);
    tile_regs_release();
    cb_push_back(cb_silu, 1);

    cb_wait_front(cb_silu, 1);
    cb_wait_front(cb_up, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_up, cb_silu);
    copy_tile(cb_silu, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_silu, cb_up);
    copy_tile(cb_up, 0, 1);
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    cb_pop_front(cb_silu, 1);
    cb_pop_front(cb_gate, 1);
    cb_pop_front(cb_up, 1);
    tile_regs_release();
    cb_push_back(cb_out, 1);
  }
}
}  // namespace NAMESPACE
