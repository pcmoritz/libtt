#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C +
         col_in_face;
}

void fill_padded_columns(uint32_t tile_l1_addr, uint32_t valid_cols, uint32_t identity_bits) {
  volatile tt_l1_ptr uint32_t *tile =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_l1_addr);
  for (uint32_t row = 0; row < TILE_R; ++row) {
    for (uint32_t col = valid_cols; col < TILE_C; ++col) {
      tile[tile_element_index(row, col)] = identity_bits;
    }
  }
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t group_offset = get_arg_val<uint32_t>(1);
  uint32_t reduce_groups = get_arg_val<uint32_t>(2);
  uint32_t width_tiles = get_arg_val<uint32_t>(3);
  uint32_t valid_last_width = get_arg_val<uint32_t>(4);
  uint32_t padding_identity_bits = get_arg_val<uint32_t>(5);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  for (uint32_t group = 0; group < reduce_groups; ++group) {
    uint32_t tile_base = (group_offset + group) * width_tiles;
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      cb_reserve_back(cb_input, 1);
      uint32_t tile_l1_addr = get_write_ptr(cb_input);
      noc_async_read_tile(tile_base + wt, input, tile_l1_addr);
      noc_async_read_barrier();
      if (wt == width_tiles - 1 && valid_last_width < TILE_C) {
        fill_padded_columns(tile_l1_addr, valid_last_width, padding_identity_bits);
      }
      cb_push_back(cb_input, 1);
    }
  }
}
