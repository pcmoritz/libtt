#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
template <DataFormat Format>
ALWI void add_input_init() {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    add_binary_tile_init();
  } else {
    add_int_tile_init();
  }
}

template <DataFormat Format>
ALWI void add_input_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Format == DataFormat::Float16 || Format == DataFormat::Float16_b ||
                Format == DataFormat::Float32) {
    add_binary_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::Int32) {
    add_int32_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::UInt32) {
    add_uint32_tile(idst0, idst1, odst);
  } else if constexpr (Format == DataFormat::UInt16) {
    add_uint16_tile(idst0, idst1, odst);
  }
}

constexpr DataFormat add_input_data_format(uint32_t cb_lhs, uint32_t cb_out) {
#ifdef UCK_CHLKC_PACK
  return static_cast<DataFormat>((uint)pack_src_format[cb_out]);
#else
  return static_cast<DataFormat>((uint)unpack_src_format[cb_lhs]);
#endif
}

void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_lhs = tt::CBIndex::c_0;
  constexpr uint32_t cb_rhs = tt::CBIndex::c_1;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr DataFormat input_format = add_input_data_format(cb_lhs, cb_out);

  unary_op_init_common(cb_lhs, cb_out);
  add_input_init<input_format>();

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_lhs, 1);
    cb_wait_front(cb_rhs, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(cb_rhs, cb_lhs);
    copy_tile(cb_lhs, 0, 0);
    copy_tile_to_dst_init_short_with_dt(cb_lhs, cb_rhs);
    copy_tile(cb_rhs, 0, 1);
    add_input_tile<input_format>(0, 1, 0);
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
