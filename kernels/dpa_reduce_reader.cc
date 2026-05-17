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

void fill_tile(uint32_t tile_l1_addr, uint32_t value_bits) {
  volatile tt_l1_ptr uint32_t *tile =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_l1_addr);
  constexpr uint32_t words = TILE_R * TILE_C;
  for (uint32_t i = 0; i < words; ++i) {
    tile[i] = value_bits;
  }
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
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t one_tile = 1;

  const uint32_t input_addr = A(0);
  const uint32_t output_tile_offset = A(1);
  const uint32_t output_tile_count = A(2);
  const uint32_t query_tokens = A(3);
  const uint32_t batch_count = A(4);
  const uint32_t kv_heads = A(5);
  const uint32_t input_tiles_per_row = A(6);
  const uint32_t output_tiles_per_row = A(7);
  const uint32_t valid_last_width = A(8);
  const uint32_t padding_identity_bits = A(9);

  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  for (uint32_t local_tile = 0; local_tile < output_tile_count; ++local_tile) {
    const uint32_t output_tile = output_tile_offset + local_tile;
    const uint32_t t_tile = output_tile % output_tiles_per_row;
    const uint32_t prefix = output_tile / output_tiles_per_row;
    const uint32_t batch = prefix % batch_count;
    const uint32_t group = prefix / batch_count;

    for (uint32_t row = 0; row < TILE_R; ++row) {
      const uint32_t kv_head = row;
      const bool valid_kv_head = kv_head < kv_heads;
      for (uint32_t width_tile = 0; width_tile < input_tiles_per_row; ++width_tile) {
        cb_reserve_back(cb_input, one_tile);
        const uint32_t tile_l1_addr = get_write_ptr(cb_input);
        if (valid_kv_head) {
          const uint32_t input_prefix = (group * batch_count + batch) * kv_heads + kv_head;
          const uint32_t input_tile =
              (input_prefix * output_tiles_per_row + t_tile) * input_tiles_per_row + width_tile;
          noc_async_read_tile(input_tile, input, tile_l1_addr);
          noc_async_read_barrier();
          if (width_tile == input_tiles_per_row - 1 && valid_last_width < TILE_C) {
            fill_padded_columns(tile_l1_addr, valid_last_width, padding_identity_bits);
          }
        } else {
          fill_tile(tile_l1_addr, padding_identity_bits);
        }
        cb_push_back(cb_input, one_tile);
      }
    }
  }
}
