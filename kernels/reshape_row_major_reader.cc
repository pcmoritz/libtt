#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
using Element = RESHAPE_ELEMENT_TYPE;

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
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

void read_input_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id, uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
  cb_wait_front(cb, 1);
}

void copy_row(uint32_t cb_input, uint32_t cb_output, uint32_t source_row,
              uint32_t output_row) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  uint32_t source_offset = tile_element_index(source_row, 0);
  uint32_t output_offset = tile_element_index(output_row, 0);
  for (uint32_t col = 0; col < FACE_C; ++col) {
    output[output_offset + col] = source[source_offset + col];
  }
  source_offset = tile_element_index(source_row, FACE_C);
  output_offset = tile_element_index(output_row, FACE_C);
  for (uint32_t col = 0; col < FACE_C; ++col) {
    output[output_offset + col] = source[source_offset + col];
  }
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(1);
  uint32_t output_tile_count = get_arg_val<uint32_t>(2);
  uint32_t input_rows = get_arg_val<uint32_t>(3);
  uint32_t input_cols = get_arg_val<uint32_t>(4);
  uint32_t input_tile_rows = get_arg_val<uint32_t>(5);
  uint32_t input_tiles_per_row = get_arg_val<uint32_t>(6);
  uint32_t output_rows = get_arg_val<uint32_t>(7);
  uint32_t output_cols = get_arg_val<uint32_t>(8);
  uint32_t output_tile_rows = get_arg_val<uint32_t>(9);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(10);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  uint32_t input_matrix_tiles = input_tile_rows * input_tiles_per_row;
  uint32_t output_matrix_tiles = output_tile_rows * output_tiles_per_row;

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / output_tiles_per_row;
    uint32_t output_tile_col = output_matrix_tile % output_tiles_per_row;

    cb_reserve_back(cb_output, 1);

#if RESHAPE_ROW_MAJOR_MODE_PACK
    uint32_t output_row_base = output_tile_row * TILE_R;
    if (output_row_base + TILE_R > output_rows) {
      zero_tile(cb_output);
    }
    for (uint32_t row = 0; row < TILE_R; ++row) {
      uint32_t output_row = output_row_base + row;
      if (output_row >= output_rows) {
        break;
      }
      uint32_t input_tile_col = output_row * output_tiles_per_row + output_tile_col;
      uint32_t input_tile = output_batch * input_tiles_per_row + input_tile_col;
      read_input_tile(input, input_tile, cb_input);
      copy_row(cb_input, cb_output, 0, row);
      cb_pop_front(cb_input, 1);
    }
#else
    zero_tile(cb_output);
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t input_row = output_col_base / input_cols;
    uint32_t input_tile_col = (output_col_base - input_row * input_cols) / TILE_C;
    uint32_t input_tile =
        output_batch * input_matrix_tiles + (input_row / TILE_R) * input_tiles_per_row +
        input_tile_col;
    read_input_tile(input, input_tile, cb_input);
    copy_row(cb_input, cb_output, input_row % TILE_R, 0);
    cb_pop_front(cb_input, 1);
#endif

    cb_push_back(cb_output, 1);
  }
}
