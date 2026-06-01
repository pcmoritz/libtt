#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t HEADS = ROTARY_HEADS;
constexpr uint32_t HALF_DIM = ROTARY_HALF_DIM;
constexpr uint32_t INPUT_TILES_PER_ROW = (HALF_DIM * 2) / TILE_C;
constexpr uint32_t SCALE_TILES_PER_ROW = HALF_DIM / TILE_C;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void read_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
               uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
}

void flip_bf16_sign(uint32_t cb) {
  volatile tt_l1_ptr uint16_t *tile =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_write_ptr(cb));
  for (uint32_t row = 0; row < TILE_R; ++row) {
    for (uint32_t col = 0; col < TILE_C; ++col) {
      tile[tile_element_index(row, col)] ^= 0x8000u;
    }
  }
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t cos_addr = get_arg_val<uint32_t>(1);
  uint32_t sin_addr = get_arg_val<uint32_t>(2);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(3);
  uint32_t output_tile_count = get_arg_val<uint32_t>(4);

  constexpr uint32_t cb_x_a = tt::CBIndex::c_0;
  constexpr uint32_t cb_cos = tt::CBIndex::c_1;
  constexpr uint32_t cb_x_b = tt::CBIndex::c_2;
  constexpr uint32_t cb_sin = tt::CBIndex::c_3;

  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_x_a),
      .data_format = get_dataformat(cb_x_a),
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

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t row_tile = output_tile_id / INPUT_TILES_PER_ROW;
    uint32_t col_tile = output_tile_id - row_tile * INPUT_TILES_PER_ROW;
    bool first_half = col_tile < SCALE_TILES_PER_ROW;
    uint32_t scale_col_tile = first_half ? col_tile : col_tile - SCALE_TILES_PER_ROW;
    uint32_t first_col_tile = scale_col_tile;
    uint32_t second_col_tile = scale_col_tile + SCALE_TILES_PER_ROW;
    uint32_t x_a_col_tile = first_half ? first_col_tile : second_col_tile;
    uint32_t x_b_col_tile = first_half ? second_col_tile : first_col_tile;
    uint32_t input_row_tile_base = row_tile * INPUT_TILES_PER_ROW;
    uint32_t scale_row_tile_base = row_tile * SCALE_TILES_PER_ROW;

    read_tile(input, input_row_tile_base + x_a_col_tile, cb_x_a);
    read_tile(cos, scale_row_tile_base + scale_col_tile, cb_cos);
    read_tile(input, input_row_tile_base + x_b_col_tile, cb_x_b);

    cb_reserve_back(cb_sin, 1);
    noc_async_read_tile(scale_row_tile_base + scale_col_tile, sin, get_write_ptr(cb_sin));
    noc_async_read_barrier();
    if (first_half) {
      flip_bf16_sign(cb_sin);
    }
    cb_push_back(cb_sin, 1);
  }
}
