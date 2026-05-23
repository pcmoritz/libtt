#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t WIDTH_TILES = ROPE_WIDTH_TILES;
constexpr uint32_t LOGICAL_ROWS = ROPE_LOGICAL_ROWS;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C +
         col_in_face;
}

void zero_invalid_rows(uint32_t l1_addr) {
  if constexpr (LOGICAL_ROWS >= TILE_R) {
    return;
  }
  volatile tt_l1_ptr uint16_t *tile =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_addr);
  for (uint32_t row = LOGICAL_ROWS; row < TILE_R; ++row) {
    for (uint32_t col = 0; col < TILE_C; ++col) {
      tile[tile_element_index(row, col)] = 0;
    }
  }
}

void read_activation_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                          uint32_t cb) {
  cb_reserve_back(cb, 1);
  uint32_t l1_addr = get_write_ptr(cb);
  noc_async_read_tile(tile_id, input, l1_addr);
  noc_async_read_barrier();
  zero_invalid_rows(l1_addr);
  cb_push_back(cb, 1);
}

void read_broadcast_row_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                             uint32_t cb) {
  cb_reserve_back(cb, 1);
  uint32_t l1_addr = get_write_ptr(cb);
  noc_async_read_tile(tile_id, input, l1_addr);
  noc_async_read_barrier();

  volatile tt_l1_ptr uint16_t *tile =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_addr);
  for (uint32_t row = 1; row < TILE_R; ++row) {
    for (uint32_t col = 0; col < TILE_C; ++col) {
      tile[tile_element_index(row, col)] = tile[tile_element_index(0, col)];
    }
  }
  cb_push_back(cb, 1);
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t cos_addr = get_arg_val<uint32_t>(1);
  uint32_t sin_addr = get_arg_val<uint32_t>(2);
  uint32_t offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_rotated = tt::CBIndex::c_1;
  constexpr uint32_t cb_cos = tt::CBIndex::c_2;
  constexpr uint32_t cb_sin = tt::CBIndex::c_3;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> cos = {
      .bank_base_address = cos_addr,
      .page_size = get_tile_size(cb_cos),
      .data_format = get_dataformat(cb_cos),
  };
  const InterleavedAddrGenFast<true> sin = {
      .bank_base_address = sin_addr,
      .page_size = get_tile_size(cb_sin),
      .data_format = get_dataformat(cb_sin),
  };

  constexpr uint32_t half_width_tiles = WIDTH_TILES / 2;
  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t tile_id = offset + i;
    uint32_t batch = tile_id / WIDTH_TILES;
    uint32_t tile_col = tile_id % WIDTH_TILES;
    uint32_t rotated_tile_col =
        tile_col < half_width_tiles ? tile_col + half_width_tiles : tile_col - half_width_tiles;
    uint32_t rotated_tile_id = batch * WIDTH_TILES + rotated_tile_col;
    uint32_t rope_tile_id = batch * WIDTH_TILES + tile_col;

    read_activation_tile(input, tile_id, cb_input);
    read_activation_tile(input, rotated_tile_id, cb_rotated);
    read_broadcast_row_tile(cos, rope_tile_id, cb_cos);
    read_broadcast_row_tile(sin, rope_tile_id, cb_sin);
  }
}
