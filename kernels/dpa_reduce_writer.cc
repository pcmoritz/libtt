#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;

#define A(n) get_arg_val<uint32_t>(n)

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  constexpr uint32_t words = TILE_R * TILE_C;
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

}  // namespace

void kernel_main() {
  constexpr uint32_t cb_reduced = tt::CBIndex::c_16;
  constexpr uint32_t cb_output = tt::CBIndex::c_17;
  constexpr uint32_t one_tile = 1;

  const uint32_t output_addr = A(0);
  const uint32_t output_tile_offset = A(1);
  const uint32_t output_tile_count = A(2);
  const uint32_t query_tokens = A(3);
  const uint32_t kv_heads = A(4);
  const uint32_t output_tiles_per_row = A(5);

  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  for (uint32_t local_tile = 0; local_tile < output_tile_count; ++local_tile) {
    const uint32_t output_tile = output_tile_offset + local_tile;
    const uint32_t t_tile = output_tile % output_tiles_per_row;

    cb_reserve_back(cb_output, one_tile);
    zero_tile(cb_output);
    volatile tt_l1_ptr uint32_t *packed_output =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb_output));

    for (uint32_t row = 0; row < TILE_R; ++row) {
      cb_wait_front(cb_reduced, one_tile);
      volatile tt_l1_ptr uint32_t *reduced =
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_read_ptr(cb_reduced));
      const uint32_t kv_head = row;
      if (kv_head < kv_heads) {
        for (uint32_t col = 0; col < TILE_C; ++col) {
          const uint32_t query_token = t_tile * TILE_C + col;
          if (query_token >= query_tokens) {
            break;
          }
          const uint32_t src_index = tile_element_index(0, col);
          const uint32_t dst_index = tile_element_index(row, col);
          packed_output[dst_index] = reduced[src_index];
        }
      }
      cb_pop_front(cb_reduced, one_tile);
    }

    cb_push_back(cb_output, one_tile);
    cb_wait_front(cb_output, one_tile);
    noc_async_write_tile(output_tile, output, get_read_ptr(cb_output));
    noc_async_write_barrier();
    cb_pop_front(cb_output, one_tile);
  }
}
