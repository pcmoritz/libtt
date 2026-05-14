#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INPUT_SHAPE[2] = STACK_INPUT_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[STACK_FLATTEN ? 2 : 3] = STACK_OUTPUT_SHAPE;
constexpr uint32_t INPUT_TILE_ROWS = STACK_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = STACK_INPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = STACK_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = STACK_OUTPUT_TILES_PER_ROW;
constexpr uint32_t HEAD_COUNT = STACK_HEAD_COUNT;
constexpr bool FLATTEN = STACK_FLATTEN != 0;
using Element = STACK_ELEMENT_TYPE;

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

void read_input_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                     uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
  cb_wait_front(cb, 1);
}

void copy_row_segment(uint32_t cb_input, uint32_t cb_output, uint32_t source_row,
                      uint32_t source_col, uint32_t output_row, uint32_t output_col,
                      uint32_t count) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));

  uint32_t copied = 0;
  while (copied < count) {
    uint32_t source_col_offset = source_col + copied;
    uint32_t output_col_offset = output_col + copied;
    uint32_t source_run = FACE_C - (source_col_offset % FACE_C);
    uint32_t output_run = FACE_C - (output_col_offset % FACE_C);
    uint32_t remaining = count - copied;
    uint32_t run = source_run < output_run ? source_run : output_run;
    run = run < remaining ? run : remaining;

    uint32_t source_index = tile_element_index(source_row, source_col_offset);
    uint32_t output_index = tile_element_index(output_row, output_col_offset);
    for (uint32_t i = 0; i < run; ++i) {
      output[output_index + i] = source[source_index + i];
    }
    copied += run;
  }
}

}  // namespace

void kernel_main() {
  uint32_t arg = 0;
  uint32_t input_addr[HEAD_COUNT];
  for (uint32_t head = 0; head < HEAD_COUNT; ++head) {
    input_addr[head] = get_arg_val<uint32_t>(arg++);
  }
  uint32_t output_tile_offset = get_arg_val<uint32_t>(arg++);
  uint32_t output_tile_count = get_arg_val<uint32_t>(arg++);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

    if constexpr (FLATTEN) {
      uint32_t output_tile_row = output_tile_id / OUTPUT_TILES_PER_ROW;
      uint32_t output_tile_col = output_tile_id % OUTPUT_TILES_PER_ROW;
      uint32_t output_row_base = output_tile_row * TILE_R;
      uint32_t output_col_base = output_tile_col * TILE_C;
      uint32_t row_count = tile_extent(INPUT_SHAPE[0], output_row_base, TILE_R);
      uint32_t head = output_col_base / INPUT_SHAPE[1];
      uint32_t input_col_base = output_col_base - head * INPUT_SHAPE[1];
      uint32_t col_count = tile_extent(INPUT_SHAPE[1], input_col_base, TILE_C);
      uint32_t input_tile_row = output_tile_row;
      uint32_t input_tile_col = input_col_base / TILE_C;
      uint32_t input_tile = input_tile_row * INPUT_TILES_PER_ROW + input_tile_col;

      const InterleavedAddrGenFast<true> input = {
          .bank_base_address = input_addr[head],
          .page_size = get_tile_size(cb_input),
          .data_format = get_dataformat(cb_input),
      };
      read_input_tile(input, input_tile, cb_input);
      for (uint32_t row = 0; row < row_count; ++row) {
        copy_row_segment(cb_input, cb_output, row, input_col_base % TILE_C, row, 0,
                         col_count);
      }
      cb_pop_front(cb_input, 1);
    } else {
      uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
      uint32_t batch = output_tile_id / output_matrix_tiles;
      uint32_t matrix_tile = output_tile_id % output_matrix_tiles;
      uint32_t output_tile_col = matrix_tile % OUTPUT_TILES_PER_ROW;
      uint32_t output_col_base = output_tile_col * TILE_C;
      uint32_t col_count = tile_extent(INPUT_SHAPE[1], output_col_base, TILE_C);
      uint32_t input_tile_row = batch / TILE_R;
      uint32_t input_row = batch % TILE_R;
      uint32_t input_tile_col = output_col_base / TILE_C;
      uint32_t input_tile = input_tile_row * INPUT_TILES_PER_ROW + input_tile_col;

      for (uint32_t head = 0; head < HEAD_COUNT; ++head) {
        const InterleavedAddrGenFast<true> input = {
            .bank_base_address = input_addr[head],
            .page_size = get_tile_size(cb_input),
            .data_format = get_dataformat(cb_input),
        };
        read_input_tile(input, input_tile, cb_input);
        copy_row_segment(cb_input, cb_output, input_row, output_col_base % TILE_C, head, 0,
                         col_count);
        cb_pop_front(cb_input, 1);
      }
    }

    cb_push_back(cb_output, 1);
  }
}
