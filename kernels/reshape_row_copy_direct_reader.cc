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

void write_row_segment(const InterleavedAddrGenFast<true> &output, uint32_t output_tile,
                       uint32_t output_row, uint32_t input_l1_addr,
                       uint32_t input_row, uint32_t element_bytes) {
  uint32_t source_offset = tile_element_index(input_row, 0) * element_bytes;
  uint32_t output_offset = tile_element_index(output_row, 0) * element_bytes;
  noc_async_write(input_l1_addr + source_offset, get_noc_addr(output_tile, output, output_offset),
                  FACE_C * element_bytes);
  source_offset = tile_element_index(input_row, FACE_C) * element_bytes;
  output_offset = tile_element_index(output_row, FACE_C) * element_bytes;
  noc_async_write(input_l1_addr + source_offset, get_noc_addr(output_tile, output, output_offset),
                  FACE_C * element_bytes);
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t output_addr = get_arg_val<uint32_t>(1);
  uint32_t fragment_offset = get_arg_val<uint32_t>(2);
  uint32_t fragment_count = get_arg_val<uint32_t>(3);
  uint32_t input_rows = get_arg_val<uint32_t>(4);
  uint32_t input_tile_rows = get_arg_val<uint32_t>(5);
  uint32_t input_tiles_per_row = get_arg_val<uint32_t>(6);
  uint32_t output_rows = get_arg_val<uint32_t>(7);
  uint32_t output_tile_rows = get_arg_val<uint32_t>(8);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(9);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  const uint32_t tile_bytes = get_tile_size(cb_input);
  const uint32_t element_bytes = tile_bytes / (TILE_R * TILE_C);
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = tile_bytes,
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = tile_bytes,
      .data_format = get_dataformat(cb_input),
  };

  const uint32_t input_matrix_tiles = input_tile_rows * input_tiles_per_row;
  const uint32_t output_matrix_tiles = output_tile_rows * output_tiles_per_row;

  for (uint32_t i = 0; i < fragment_count; ++i) {
    uint32_t fragment = fragment_offset + i;
    uint32_t global_row = fragment / output_tiles_per_row;
    uint32_t tile_col = fragment - global_row * output_tiles_per_row;

    uint32_t input_batch = global_row / input_rows;
    uint32_t input_row = global_row - input_batch * input_rows;
    uint32_t input_tile = input_batch * input_matrix_tiles +
                          (input_row / TILE_R) * input_tiles_per_row + tile_col;

    uint32_t output_batch = global_row / output_rows;
    uint32_t output_row = global_row - output_batch * output_rows;
    uint32_t output_tile = output_batch * output_matrix_tiles +
                           (output_row / TILE_R) * output_tiles_per_row + tile_col;

    cb_reserve_back(cb_input, 1);
    uint32_t input_l1_addr = get_write_ptr(cb_input);
    noc_async_read_tile(input_tile, input, input_l1_addr);
    noc_async_read_barrier();
    write_row_segment(output, output_tile, output_row % TILE_R, input_l1_addr,
                      input_row % TILE_R, element_bytes);
    noc_async_write_barrier();
    cb_push_back(cb_input, 1);
    cb_pop_front(cb_input, 1);
  }
}
