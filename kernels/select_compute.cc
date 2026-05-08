#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api.h"

#ifdef TRISC_MATH
#define SELECT_ITERATIONS (8)

template <bool KeepWhenPred, bool Int32Value>
inline void gate_value(const uint dst_index_pred, const uint dst_index_value, const uint dst_index_out) {
  constexpr uint dst_tile_size_sfpi = 32;
  for (int i = 0; i < SELECT_ITERATIONS; ++i) {
    vInt pred = dst_reg[dst_index_pred * dst_tile_size_sfpi];
    if constexpr (Int32Value) {
      vInt values = dst_reg[dst_index_value * dst_tile_size_sfpi];
      if constexpr (KeepWhenPred) {
        v_if (pred != 0) {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = values;
        } v_else {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = 0;
        } v_endif;
      } else {
        v_if (pred != 0) {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = 0;
        } v_else {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = values;
        } v_endif;
      }
    } else {
      vFloat values = dst_reg[dst_index_value * dst_tile_size_sfpi];
      if constexpr (KeepWhenPred) {
        v_if (pred != 0) {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = values;
        } v_else {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = 0.0f;
        } v_endif;
      } else {
        v_if (pred != 0) {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = 0.0f;
        } v_else {
          dst_reg[dst_index_out * dst_tile_size_sfpi] = values;
        } v_endif;
      }
    }
    dst_reg++;
  }
}
#endif

namespace NAMESPACE {
constexpr DataFormat value_format(uint32_t cb_value, uint32_t cb_out) {
#ifdef UCK_CHLKC_PACK
  return static_cast<DataFormat>((uint)pack_src_format[cb_out]);
#else
  return static_cast<DataFormat>((uint)unpack_src_format[cb_value]);
#endif
}

template <DataFormat Format>
ALWI void add_selected_init() {
  if constexpr (Format == DataFormat::Int32) {
    add_int_tile_init();
  } else {
    add_binary_tile_init();
  }
}

template <DataFormat Format>
ALWI void add_selected_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
  if constexpr (Format == DataFormat::Int32) {
    add_int32_tile(idst0, idst1, odst);
  } else {
    add_binary_tile(idst0, idst1, odst);
  }
}

void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);

  constexpr uint32_t cb_pred = tt::CBIndex::c_0;
  constexpr uint32_t cb_true = tt::CBIndex::c_1;
  constexpr uint32_t cb_false = tt::CBIndex::c_2;
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr DataFormat format = value_format(cb_true, cb_out);
  constexpr bool int32_value = format == DataFormat::Int32;

  unary_op_init_common(cb_true, cb_out);
  add_selected_init<format>();

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_pred, 1);
    cb_wait_front(cb_true, 1);
    cb_wait_front(cb_false, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    reconfig_data_format_srca<true>(cb_pred);
    copy_tile_to_dst_init_short(cb_pred);
    copy_tile(cb_pred, 0, 0);
    reconfig_data_format_srca<true>(cb_true);
    copy_tile_init(cb_true);
    copy_tile(cb_true, 0, 1);
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(gate_value<true, int32_value>, 0, 1, 1, VectorMode::RC);)
    copy_tile_init(cb_false);
    copy_tile(cb_false, 0, 2);
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(gate_value<false, int32_value>, 0, 2, 2, VectorMode::RC);)
    add_selected_tile<format>(1, 2, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    cb_pop_front(cb_pred, 1);
    cb_pop_front(cb_true, 1);
    cb_pop_front(cb_false, 1);
    cb_push_back(cb_out, 1);
  }
}
}  // namespace NAMESPACE
