#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/sub_int_sfpu.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
enum class CompareDirection : uint32_t {
  Eq,
  Ne,
  Ge,
  Gt,
  Le,
  Lt,
};

template <bool Int32Input>
ALWI void compare_sub_init() {
  if constexpr (Int32Input) {
    sub_int_tile_init();
  } else {
    sub_binary_tile_init();
  }
}

template <bool Int32Input>
ALWI void compare_sub_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Int32Input) {
    sub_int32_tile(idst0, idst1, odst);
  } else {
    sub_binary_tile(idst0, idst1, odst);
  }
}

ALWI void compare_zero_init(CompareDirection direction) {
  switch (direction) {
    case CompareDirection::Eq: eqz_tile_init(); break;
    case CompareDirection::Ne: nez_tile_init(); break;
    case CompareDirection::Ge: gez_tile_init(); break;
    case CompareDirection::Gt: gtz_tile_init(); break;
    case CompareDirection::Le: lez_tile_init(); break;
    case CompareDirection::Lt: ltz_tile_init(); break;
    default: break;
  }
}

template <bool Int32Input>
ALWI void compare_zero_tile(CompareDirection direction, uint32_t idst) {
  switch (direction) {
    case CompareDirection::Eq:
      if constexpr (Int32Input) {
        eqz_tile_int32(idst);
      } else {
        eqz_tile(idst);
      }
      break;
    case CompareDirection::Ne:
      if constexpr (Int32Input) {
        nez_tile_int32(idst);
      } else {
        nez_tile(idst);
      }
      break;
    case CompareDirection::Ge:
      if constexpr (Int32Input) {
        gez_tile_int32(idst);
      } else {
        gez_tile(idst);
      }
      break;
    case CompareDirection::Gt:
      if constexpr (Int32Input) {
        gtz_tile_int32(idst);
      } else {
        gtz_tile(idst);
      }
      break;
    case CompareDirection::Le:
      if constexpr (Int32Input) {
        lez_tile_int32(idst);
      } else {
        lez_tile(idst);
      }
      break;
    case CompareDirection::Lt:
      if constexpr (Int32Input) {
        ltz_tile_int32(idst);
      } else {
        ltz_tile(idst);
      }
      break;
    default: break;
  }
}

void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr bool int32_input = COMPARE_INT32_INPUT;
  constexpr CompareDirection direction = CompareDirection::COMPARE_DIRECTION;

  unary_op_init_common(cb_lhs, cb_out);
  compare_sub_init<int32_input>();
  compare_zero_init(direction);

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_lhs, 1);
    cb_wait_front(cb_rhs, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_rhs, cb_lhs);
    copy_tile(cb_lhs, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_lhs, cb_rhs);
    copy_tile(cb_rhs, 0, 1);
    compare_sub_tile<int32_input>(0, 1, 0);
    compare_zero_tile<int32_input>(direction, 0);

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
