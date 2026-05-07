#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
enum class AddInputKind : uint32_t {
  Float,
  Int32,
  UInt32,
  UInt16,
};

template <AddInputKind Kind>
ALWI void add_input_init() {
  if constexpr (Kind == AddInputKind::Float) {
    add_binary_tile_init();
  } else {
    add_int_tile_init();
  }
}

template <AddInputKind Kind>
ALWI void add_input_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Kind == AddInputKind::Float) {
    add_binary_tile(idst0, idst1, odst);
  } else if constexpr (Kind == AddInputKind::Int32) {
    add_int32_tile(idst0, idst1, odst);
  } else if constexpr (Kind == AddInputKind::UInt32) {
    add_uint32_tile(idst0, idst1, odst);
  } else if constexpr (Kind == AddInputKind::UInt16) {
    add_uint16_tile(idst0, idst1, odst);
  }
}

void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr AddInputKind input_kind = AddInputKind::ADD_INPUT_KIND;

  unary_op_init_common(cb_lhs, cb_out);
  add_input_init<input_kind>();

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_lhs, 1);
    cb_wait_front(cb_rhs, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_rhs, cb_lhs);
    copy_tile(cb_lhs, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_lhs, cb_rhs);
    copy_tile(cb_rhs, 0, 1);
    add_input_tile<input_kind>(0, 1, 0);
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
