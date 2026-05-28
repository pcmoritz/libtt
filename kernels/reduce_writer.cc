#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
using Element = REDUCE_ELEMENT_TYPE;

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void zero_tiles(uint32_t base_l1_addr, uint32_t tile_size, uint32_t tile_count) {
  for (uint32_t tile = 0; tile < tile_count; ++tile) {
    volatile tt_l1_ptr uint32_t *ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(base_l1_addr + tile * tile_size);
    uint32_t words = tile_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < words; ++i) {
      ptr[i] = 0;
    }
  }
}

void copy_reduced_tile(uint32_t reduced_l1_addr, uint32_t output_base_l1_addr,
                       uint32_t output_tile_size, uint32_t global_group,
                       uint32_t inner_output_tiles, uint32_t output_tile_offset,
                       uint32_t output_dim0, uint32_t output_dim1,
                       uint32_t output_tile_rows_per_prefix) {
  volatile tt_l1_ptr Element *reduced =
      reinterpret_cast<volatile tt_l1_ptr Element *>(reduced_l1_addr);
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(output_base_l1_addr);
  uint32_t output_tile_elements = output_tile_size / sizeof(Element);
  uint32_t output_col_base = (global_group % inner_output_tiles) * TILE_C;

  uint32_t row_group = global_group / inner_output_tiles;
  uint32_t prefix = row_group / output_dim0;
  uint32_t output_row = row_group % output_dim0;
  uint32_t output_tile_row = output_row / TILE_R;
  uint32_t output_tile_col = output_col_base / TILE_C;
  uint32_t output_tile =
      (prefix * output_tile_rows_per_prefix + output_tile_row) * inner_output_tiles + output_tile_col;
  uint32_t local_output_tile = output_tile - output_tile_offset;

  for (uint32_t col = 0; col < TILE_C; ++col) {
    uint32_t output_col = output_col_base + col;
    if (output_col >= output_dim1) {
      continue;
    }

    uint32_t output_index =
        local_output_tile * output_tile_elements + tile_element_index(output_row % TILE_R, output_col % TILE_C);
    uint32_t reduced_index = REDUCE_BLOCK_MAX_ROW ? tile_element_index(col, 0)
                                                  : tile_element_index(0, col);
    output[output_index] = reduced[reduced_index];
  }
}

}  // namespace

void kernel_main() {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t group_offset = get_arg_val<uint32_t>(1);
  uint32_t reduce_groups = get_arg_val<uint32_t>(2);
  uint32_t inner_output_tiles = get_arg_val<uint32_t>(3);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(4);
  uint32_t output_tiles = get_arg_val<uint32_t>(5);
  uint32_t output_dim0 = get_arg_val<uint32_t>(6);
  uint32_t output_dim1 = get_arg_val<uint32_t>(7);
  uint32_t output_tile_rows_per_prefix = get_arg_val<uint32_t>(8);

  constexpr uint32_t cb_reduced = tt::CBIndex::c_16;
  constexpr uint32_t cb_output = tt::CBIndex::c_17;
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  cb_reserve_back(cb_output, output_tiles);
  uint32_t output_base_l1_addr = get_write_ptr(cb_output);
  uint32_t output_tile_size = get_tile_size(cb_output);
  zero_tiles(output_base_l1_addr, output_tile_size, output_tiles);

  for (uint32_t group = 0; group < reduce_groups; ++group) {
    cb_wait_front(cb_reduced, 1);
    copy_reduced_tile(get_read_ptr(cb_reduced), output_base_l1_addr, output_tile_size,
                      group_offset + group, inner_output_tiles, output_tile_offset, output_dim0,
                      output_dim1, output_tile_rows_per_prefix);
    cb_pop_front(cb_reduced, 1);
  }

  for (uint32_t tile = 0; tile < output_tiles; ++tile) {
    noc_async_write_tile(output_tile_offset + tile, output, output_base_l1_addr + tile * output_tile_size);
    noc_async_write_barrier();
  }
  cb_push_back(cb_output, output_tiles);
  cb_pop_front(cb_output, output_tiles);
}
