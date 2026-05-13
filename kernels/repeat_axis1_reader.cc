#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INPUT_ROWS = REPEAT_INPUT_ROWS;
constexpr uint32_t OUTPUT_ROWS = REPEAT_OUTPUT_ROWS;
constexpr uint32_t COLS = REPEAT_COLS;
constexpr uint32_t REPEATS = REPEAT_FACTOR;
constexpr uint32_t INPUT_TILE_ROWS = REPEAT_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = REPEAT_INPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = REPEAT_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = REPEAT_OUTPUT_TILES_PER_ROW;
using Element = REPEAT_ELEMENT_TYPE;

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

void copy_element(uint32_t cb_input, uint32_t cb_output, uint32_t source_row,
                  uint32_t source_col, uint32_t output_row, uint32_t output_col) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  output[tile_element_index(output_row, output_col)] =
      source[tile_element_index(source_row, source_col)];
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(1);
  uint32_t output_tile_count = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
    uint32_t batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t row_count = tile_extent(OUTPUT_ROWS, output_row_base, TILE_R);
    uint32_t col_count = tile_extent(COLS, output_col_base, TILE_C);

    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

    uint32_t loaded_input_tile = 0xffffffffu;
    for (uint32_t row = 0; row < row_count; ++row) {
      uint32_t output_row = output_row_base + row;
      uint32_t input_row = output_row / REPEATS;
      uint32_t input_tile_row = input_row / TILE_R;
      uint32_t source_row = input_row % TILE_R;
      for (uint32_t col = 0; col < col_count; ++col) {
        uint32_t output_col = output_col_base + col;
        uint32_t input_tile_col = output_col / TILE_C;
        uint32_t input_tile =
            (batch * INPUT_TILE_ROWS + input_tile_row) * INPUT_TILES_PER_ROW +
            input_tile_col;
        if (input_tile != loaded_input_tile) {
          if (loaded_input_tile != 0xffffffffu) {
            cb_pop_front(cb_input, 1);
          }
          cb_reserve_back(cb_input, 1);
          noc_async_read_tile(input_tile, input, get_write_ptr(cb_input));
          noc_async_read_barrier();
          cb_push_back(cb_input, 1);
          cb_wait_front(cb_input, 1);
          loaded_input_tile = input_tile;
        }
        copy_element(cb_input, cb_output, source_row, output_col % TILE_C, row, col);
      }
    }

    if (loaded_input_tile != 0xffffffffu) {
      cb_pop_front(cb_input, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
