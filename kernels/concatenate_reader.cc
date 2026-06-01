#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INPUT_COUNT = CONCAT_INPUT_COUNT;
constexpr bool CONCAT_COLS = CONCAT_AXIS_COLS;
using Element = CONCAT_ELEMENT_TYPE;

uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb));
  uint32_t elements = get_tile_size(cb) / sizeof(Element);
  for (uint32_t i = 0; i < elements; ++i) {
    ptr[i] = 0;
  }
}

void read_input_tile(uint32_t input_addr, uint32_t tile_id, uint32_t cb) {
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb),
      .data_format = get_dataformat(cb),
  };
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

void copy_row(uint32_t cb_input, uint32_t cb_output, uint32_t source_row,
              uint32_t source_col, uint32_t output_row, uint32_t output_col,
              uint32_t count) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  if (count == TILE_C && source_col == 0 && output_col == 0) {
    uint32_t source_face0 = tile_element_index(source_row, 0);
    uint32_t source_face1 = tile_element_index(source_row, FACE_C);
    uint32_t output_face0 = tile_element_index(output_row, 0);
    uint32_t output_face1 = tile_element_index(output_row, FACE_C);
    for (uint32_t col = 0; col < FACE_C; ++col) {
      output[output_face0 + col] = source[source_face0 + col];
      output[output_face1 + col] = source[source_face1 + col];
    }
    return;
  }
  for (uint32_t col = 0; col < count; ++col) {
    output[tile_element_index(output_row, output_col + col)] =
        source[tile_element_index(source_row, source_col + col)];
  }
}

uint32_t input_tile_id(uint32_t batch, uint32_t tile_row, uint32_t tile_col,
                       uint32_t input_tile_rows, uint32_t input_tiles_per_row) {
  return (batch * input_tile_rows + tile_row) * input_tiles_per_row + tile_col;
}

}  // namespace

void kernel_main() {
  uint32_t arg = 0;
  uint32_t input_addr[INPUT_COUNT];
  for (uint32_t i = 0; i < INPUT_COUNT; ++i) {
    input_addr[i] = get_arg_val<uint32_t>(arg++);
  }

  uint32_t output_tile_offset = get_arg_val<uint32_t>(arg++);
  uint32_t output_tile_count = get_arg_val<uint32_t>(arg++);
  uint32_t output_rows = get_arg_val<uint32_t>(arg++);
  uint32_t output_cols = get_arg_val<uint32_t>(arg++);
  uint32_t output_tile_rows = get_arg_val<uint32_t>(arg++);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(arg++);

  uint32_t input_rows[INPUT_COUNT];
  uint32_t input_cols[INPUT_COUNT];
  uint32_t input_tile_rows[INPUT_COUNT];
  uint32_t input_tiles_per_row[INPUT_COUNT];
  uint32_t concat_offsets[INPUT_COUNT];
  for (uint32_t i = 0; i < INPUT_COUNT; ++i) {
    input_rows[i] = get_arg_val<uint32_t>(arg++);
    input_cols[i] = get_arg_val<uint32_t>(arg++);
    input_tile_rows[i] = get_arg_val<uint32_t>(arg++);
    input_tiles_per_row[i] = get_arg_val<uint32_t>(arg++);
    concat_offsets[i] = get_arg_val<uint32_t>(arg++);
  }

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  uint32_t output_matrix_tiles = output_tile_rows * output_tiles_per_row;

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t batch = output_tile_id / output_matrix_tiles;
    uint32_t matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = matrix_tile / output_tiles_per_row;
    uint32_t output_tile_col = matrix_tile % output_tiles_per_row;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;

    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

    for (uint32_t input_index = 0; input_index < INPUT_COUNT; ++input_index) {
      if (CONCAT_COLS) {
        uint32_t row_begin = output_row_base;
        uint32_t row_end = min_u32(output_row_base + TILE_R, output_rows);
        uint32_t col_begin =
            max_u32(output_col_base, concat_offsets[input_index]);
        uint32_t col_end = min_u32(
            min_u32(output_col_base + TILE_C, output_cols),
            concat_offsets[input_index] + input_cols[input_index]);
        if (row_begin >= row_end || col_begin >= col_end) {
          continue;
        }

        uint32_t source_tile_row = output_tile_row;
        uint32_t source_col_begin = col_begin - concat_offsets[input_index];
        uint32_t source_col_end = col_end - concat_offsets[input_index];
        uint32_t source_tile_col_begin = source_col_begin / TILE_C;
        uint32_t source_tile_col_end = (source_col_end - 1) / TILE_C;
        for (uint32_t source_tile_col = source_tile_col_begin;
             source_tile_col <= source_tile_col_end; ++source_tile_col) {
          uint32_t source_tile_col_base = source_tile_col * TILE_C;
          uint32_t copy_col_begin =
              max_u32(col_begin, concat_offsets[input_index] + source_tile_col_base);
          uint32_t copy_col_end = min_u32(
              col_end, concat_offsets[input_index] + source_tile_col_base + TILE_C);
          uint32_t tile_id =
              input_tile_id(batch, source_tile_row, source_tile_col,
                            input_tile_rows[input_index],
                            input_tiles_per_row[input_index]);

          read_input_tile(input_addr[input_index], tile_id, cb_input);
          for (uint32_t row = row_begin; row < row_end; ++row) {
            copy_row(cb_input, cb_output, row - output_row_base,
                     copy_col_begin - concat_offsets[input_index] - source_tile_col_base,
                     row - output_row_base, copy_col_begin - output_col_base,
                     copy_col_end - copy_col_begin);
          }
          cb_pop_front(cb_input, 1);
        }
      } else {
        uint32_t row_begin =
            max_u32(output_row_base, concat_offsets[input_index]);
        uint32_t row_end = min_u32(
            min_u32(output_row_base + TILE_R, output_rows),
            concat_offsets[input_index] + input_rows[input_index]);
        uint32_t col_begin = output_col_base;
        uint32_t col_end = min_u32(output_col_base + TILE_C, output_cols);
        if (row_begin >= row_end || col_begin >= col_end) {
          continue;
        }

        uint32_t source_col_begin = col_begin;
        uint32_t source_col_end = col_end;
        uint32_t source_tile_col = output_tile_col;
        uint32_t source_tile_row_begin =
            (row_begin - concat_offsets[input_index]) / TILE_R;
        uint32_t source_tile_row_end =
            (row_end - concat_offsets[input_index] - 1) / TILE_R;
        for (uint32_t source_tile_row = source_tile_row_begin;
             source_tile_row <= source_tile_row_end; ++source_tile_row) {
          uint32_t source_tile_row_base = source_tile_row * TILE_R;
          uint32_t copy_row_begin =
              max_u32(row_begin, concat_offsets[input_index] + source_tile_row_base);
          uint32_t copy_row_end = min_u32(
              row_end, concat_offsets[input_index] + source_tile_row_base + TILE_R);
          uint32_t tile_id =
              input_tile_id(batch, source_tile_row, source_tile_col,
                            input_tile_rows[input_index],
                            input_tiles_per_row[input_index]);

          read_input_tile(input_addr[input_index], tile_id, cb_input);
          for (uint32_t row = copy_row_begin; row < copy_row_end; ++row) {
            copy_row(cb_input, cb_output,
                     row - concat_offsets[input_index] - source_tile_row_base,
                     source_col_begin - output_col_base, row - output_row_base,
                     source_col_begin - output_col_base,
                     source_col_end - source_col_begin);
          }
          cb_pop_front(cb_input, 1);
        }
      }
    }

    cb_push_back(cb_output, 1);
  }
}
