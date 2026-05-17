#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INPUT_RANK = BROADCAST_INPUT_RANK;
constexpr uint32_t OUTPUT_RANK = BROADCAST_OUTPUT_RANK;
constexpr uint32_t OUTPUT_COORD_COUNT = OUTPUT_RANK == 0 ? 1 : OUTPUT_RANK;
constexpr uint32_t OUTPUT_SHAPE[OUTPUT_COORD_COUNT] = BROADCAST_OUTPUT_SHAPE;
constexpr uint32_t INPUT_TILE_ROWS = BROADCAST_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = BROADCAST_INPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = BROADCAST_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = BROADCAST_OUTPUT_TILES_PER_ROW;
using Element = BROADCAST_ELEMENT_TYPE;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

uint32_t tile_extent(uint32_t logical_dim, uint32_t base, uint32_t tile_dim) {
  if (base >= logical_dim) {
    return 0;
  }
  uint32_t remaining = logical_dim - base;
  return remaining < tile_dim ? remaining : tile_dim;
}

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

}  // namespace

void kernel_main() {
  const uint32_t input_addr = get_arg_val<uint32_t>(0);
  const uint32_t output_tile_offset = get_arg_val<uint32_t>(1);
  const uint32_t output_tile_count = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  constexpr uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    const uint32_t output_tile_id = output_tile_offset + tile;
    const uint32_t output_batch = output_tile_id / output_matrix_tiles;
    const uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    const uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    const uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    const uint32_t output_row_base = output_tile_row * TILE_R;
    const uint32_t output_col_base = output_tile_col * TILE_C;
    const uint32_t row_count =
        tile_extent(OUTPUT_SHAPE[OUTPUT_RANK - 2], output_row_base, TILE_R);
    const uint32_t col_count =
        tile_extent(OUTPUT_SHAPE[OUTPUT_RANK - 1], output_col_base, TILE_C);
    const uint32_t input_tile =
        (output_batch * INPUT_TILE_ROWS + output_tile_row) * INPUT_TILES_PER_ROW;

    cb_reserve_back(cb_input, 1);
    noc_async_read_tile(input_tile, input, get_write_ptr(cb_input));
    noc_async_read_barrier();
    cb_push_back(cb_input, 1);
    cb_wait_front(cb_input, 1);

    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);
    volatile tt_l1_ptr Element *source =
        reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
    volatile tt_l1_ptr Element *output =
        reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
    for (uint32_t row = 0; row < row_count; ++row) {
      const Element value = source[tile_element_index(row, 0)];
      for (uint32_t col = 0; col < col_count; ++col) {
        output[tile_element_index(row, col)] = value;
      }
    }
    cb_pop_front(cb_input, 1);
    cb_push_back(cb_output, 1);
  }
}
