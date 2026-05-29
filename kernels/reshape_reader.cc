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

uint32_t min_u32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? lhs : rhs;
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
  uint32_t logical_volume = get_arg_val<uint32_t>(3);
  uint32_t input_rows = get_arg_val<uint32_t>(4);
  uint32_t input_cols = get_arg_val<uint32_t>(5);
  uint32_t input_tile_rows = get_arg_val<uint32_t>(6);
  uint32_t input_tiles_per_row = get_arg_val<uint32_t>(7);
  uint32_t output_rows = get_arg_val<uint32_t>(8);
  uint32_t output_cols = get_arg_val<uint32_t>(9);
  uint32_t output_tile_rows = get_arg_val<uint32_t>(10);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(11);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };

  uint32_t output_matrix_tiles = output_tile_rows * output_tiles_per_row;

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / output_tiles_per_row;
    uint32_t output_tile_col = output_matrix_tile % output_tiles_per_row;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t loaded_input_tile = 0xffffffffu;

    uint32_t valid_rows = output_row_base < output_rows
                              ? min_u32(TILE_R, output_rows - output_row_base)
                              : 0;
    uint32_t valid_cols = output_col_base < output_cols
                              ? min_u32(TILE_C, output_cols - output_col_base)
                              : 0;
    uint32_t tile_first_index =
        (output_batch * output_rows + output_row_base) * output_cols + output_col_base;
    uint32_t tile_last_index =
        tile_first_index + (TILE_R - 1) * output_cols + (TILE_C - 1);
    bool tile_fully_written =
        valid_rows == TILE_R && valid_cols == TILE_C && tile_last_index < logical_volume;

    cb_reserve_back(cb_output, 1);
    if (!tile_fully_written) {
      zero_tile(cb_output);
    }

    for (uint32_t row = 0; row < valid_rows; ++row) {
      uint32_t output_row = output_row_base + row;
      uint32_t flat_index =
          (output_batch * output_rows + output_row) * output_cols + output_col_base;
      if (flat_index >= logical_volume) {
        continue;
      }

      uint32_t row_cols = min_u32(valid_cols, logical_volume - flat_index);
      uint32_t input_col = flat_index % input_cols;
      uint32_t input_row_major = flat_index / input_cols;
      uint32_t input_row = input_row_major % input_rows;
      uint32_t input_batch = input_row_major / input_rows;

      for (uint32_t col = 0; col < row_cols; ++col) {
        uint32_t input_tile_row = input_row / TILE_R;
        uint32_t input_tile_col = input_col / TILE_C;
        uint32_t input_tile =
            (input_batch * input_tile_rows + input_tile_row) * input_tiles_per_row +
            input_tile_col;

        if (input_tile != loaded_input_tile) {
          if (loaded_input_tile != 0xffffffffu) {
            cb_pop_front(cb_input, 1);
          }
          read_input_tile(input, input_tile, cb_input);
          loaded_input_tile = input_tile;
        }

        copy_element(cb_input, cb_output, input_row % TILE_R, input_col % TILE_C, row, col);
        ++input_col;
        if (input_col == input_cols) {
          input_col = 0;
          ++input_row;
          if (input_row == input_rows) {
            input_row = 0;
            ++input_batch;
          }
        }
      }
    }

    if (loaded_input_tile != 0xffffffffu) {
      cb_pop_front(cb_input, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
