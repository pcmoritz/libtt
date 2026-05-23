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
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

}  // namespace

void kernel_main() {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t partial_tile_id = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_reduced = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_reduced),
      .data_format = get_dataformat(cb_reduced),
  };

  cb_wait_front(cb_reduced, 1);
  uint32_t reduced_l1_addr = get_read_ptr(cb_reduced);
  volatile tt_l1_ptr uint32_t *tile =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(reduced_l1_addr);
  uint32_t values[TILE_R];
  for (uint32_t row = 0; row < TILE_R; ++row) {
    values[row] = tile[tile_element_index(0, row)];
  }
  for (uint32_t row = 0; row < TILE_R; ++row) {
    for (uint32_t col = 0; col < TILE_C; ++col) {
      tile[tile_element_index(row, col)] = 0;
    }
    tile[tile_element_index(row, 0)] = values[row];
  }
  noc_async_write_tile(partial_tile_id, output, reduced_l1_addr);
  noc_async_write_barrier();
  cb_pop_front(cb_reduced, 1);
}
