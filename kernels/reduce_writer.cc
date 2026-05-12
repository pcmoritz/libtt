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
                       uint32_t output_tile_size, uint32_t group, uint32_t input_row_tiles,
                       uint32_t output_tiles_per_row, uint32_t output_rank, uint32_t output_dim0,
                       uint32_t output_dim1) {
  volatile tt_l1_ptr uint32_t *reduced =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(reduced_l1_addr);
  volatile tt_l1_ptr uint32_t *output =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(output_base_l1_addr);
  uint32_t output_tile_elements = output_tile_size / sizeof(uint32_t);
  uint32_t outer = group / input_row_tiles;
  uint32_t row_tile = group % input_row_tiles;

  for (uint32_t row = 0; row < TILE_R; ++row) {
    uint32_t output_row = 0;
    uint32_t output_col = 0;
    if (output_rank == 1) {
      output_col = group * TILE_R + row;
      if (output_col >= output_dim1) {
        continue;
      }
    } else {
      output_row = outer;
      output_col = row_tile * TILE_R + row;
      if (output_row >= output_dim0 || output_col >= output_dim1) {
        continue;
      }
    }

    uint32_t output_tile_row = output_row / TILE_R;
    uint32_t output_tile_col = output_col / TILE_C;
    uint32_t output_tile = output_tile_row * output_tiles_per_row + output_tile_col;
    uint32_t output_index =
        output_tile * output_tile_elements + tile_element_index(output_row % TILE_R, output_col % TILE_C);
    output[output_index] = reduced[tile_element_index(0, row)];
  }
}

}  // namespace

void kernel_main() {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t reduce_groups = get_arg_val<uint32_t>(1);
  uint32_t input_row_tiles = get_arg_val<uint32_t>(2);
  uint32_t output_tiles = get_arg_val<uint32_t>(3);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(4);
  uint32_t output_rank = get_arg_val<uint32_t>(5);
  uint32_t output_dim0 = get_arg_val<uint32_t>(6);
  uint32_t output_dim1 = get_arg_val<uint32_t>(7);

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
    copy_reduced_tile(get_read_ptr(cb_reduced), output_base_l1_addr, output_tile_size, group,
                      input_row_tiles, output_tiles_per_row, output_rank, output_dim0, output_dim1);
    cb_pop_front(cb_reduced, 1);
  }

  for (uint32_t tile = 0; tile < output_tiles; ++tile) {
    noc_async_write_tile(tile, output, output_base_l1_addr + tile * output_tile_size);
    noc_async_write_barrier();
  }
  cb_push_back(cb_output, output_tiles);
  cb_pop_front(cb_output, output_tiles);
}
