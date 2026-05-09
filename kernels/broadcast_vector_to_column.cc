#include <cstdint>

namespace {

constexpr uint32_t kTileRows = 32;
constexpr uint32_t kTileCols = 32;

uint32_t tile_element_offset(uint32_t row, uint32_t col) {
  uint32_t face_row = row >> 4;
  uint32_t face_col = col >> 4;
  uint32_t local_row = row & 0xf;
  uint32_t local_col = col & 0xf;
  return ((face_row * 2 + face_col) * 256) + local_row * 16 + local_col;
}

void zero_tile(uint32_t l1_addr, uint32_t tile_bytes) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(l1_addr);
  for (uint32_t i = 0; i < tile_bytes / sizeof(uint32_t); ++i) {
    ptr[i] = 0;
  }
}

void copy_element(uint32_t src_l1_addr, uint32_t dst_l1_addr,
                  uint32_t src_element, uint32_t dst_element,
                  uint32_t element_bytes) {
  if (element_bytes == sizeof(uint32_t)) {
    volatile tt_l1_ptr uint32_t *src =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(src_l1_addr);
    volatile tt_l1_ptr uint32_t *dst =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dst_l1_addr);
    dst[dst_element] = src[src_element];
    return;
  }
  if (element_bytes == sizeof(uint16_t)) {
    volatile tt_l1_ptr uint16_t *src =
        reinterpret_cast<volatile tt_l1_ptr uint16_t *>(src_l1_addr);
    volatile tt_l1_ptr uint16_t *dst =
        reinterpret_cast<volatile tt_l1_ptr uint16_t *>(dst_l1_addr);
    dst[dst_element] = src[src_element];
    return;
  }
  volatile tt_l1_ptr uint8_t *src =
      reinterpret_cast<volatile tt_l1_ptr uint8_t *>(src_l1_addr);
  volatile tt_l1_ptr uint8_t *dst =
      reinterpret_cast<volatile tt_l1_ptr uint8_t *>(dst_l1_addr);
  dst[dst_element] = src[src_element];
}

uint32_t input_coord(uint32_t input_size, uint32_t output_coord) {
  return input_size == 1 ? 0 : output_coord;
}

uint32_t mapped_output_coord(uint32_t physical_dim, uint32_t output_row,
                             uint32_t output_col) {
  return physical_dim == 0 ? output_row : output_col;
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t output_addr = get_arg_val<uint32_t>(1);
  uint32_t offset = get_arg_val<uint32_t>(2);
  uint32_t n_tiles = get_arg_val<uint32_t>(3);
  uint32_t input_rank = get_arg_val<uint32_t>(4);
  uint32_t input_rows = get_arg_val<uint32_t>(5);
  uint32_t input_cols = get_arg_val<uint32_t>(6);
  uint32_t output_rows = get_arg_val<uint32_t>(7);
  uint32_t output_cols = get_arg_val<uint32_t>(8);
  uint32_t input_tiles_per_row = get_arg_val<uint32_t>(9);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(10);
  uint32_t dim0 = get_arg_val<uint32_t>(11);
  uint32_t dim1 = get_arg_val<uint32_t>(12);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> output = {
      .bank_base_address = output_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  uint32_t tile_bytes = get_tile_size(cb_input);
  uint32_t element_bytes = tile_bytes / (kTileRows * kTileCols);

  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t output_tile_id = offset + i;
    uint32_t output_tile_row = output_tile_id / output_tiles_per_row;
    uint32_t output_tile_col = output_tile_id % output_tiles_per_row;

    uint32_t base_output_row = output_tile_row * kTileRows;
    uint32_t base_output_col = output_tile_col * kTileCols;
    uint32_t source_row = 0;
    uint32_t source_col = 0;
    if (input_rank == 1) {
      uint32_t coord = mapped_output_coord(dim0, base_output_row, base_output_col);
      source_col = input_coord(input_cols, coord);
    } else if (input_rank == 2) {
      uint32_t coord0 = mapped_output_coord(dim0, base_output_row, base_output_col);
      uint32_t coord1 = mapped_output_coord(dim1, base_output_row, base_output_col);
      source_row = input_coord(input_rows, coord0);
      source_col = input_coord(input_cols, coord1);
    }

    uint32_t input_tile_id =
        (source_row / kTileRows) * input_tiles_per_row + (source_col / kTileCols);

    cb_reserve_back(cb_input, 1);
    cb_reserve_back(cb_output, 1);

    uint32_t input_l1 = get_write_ptr(cb_input);
    uint32_t output_l1 = get_write_ptr(cb_output);

    noc_async_read_tile(input_tile_id, input, input_l1);
    noc_async_read_barrier();
    zero_tile(output_l1, get_tile_size(cb_output));

    for (uint32_t row = 0; row < kTileRows; ++row) {
      uint32_t output_row = base_output_row + row;
      if (output_row >= output_rows) {
        continue;
      }
      for (uint32_t col = 0; col < kTileCols; ++col) {
        uint32_t output_col = base_output_col + col;
        if (output_col >= output_cols) {
          continue;
        }

        uint32_t src_row = 0;
        uint32_t src_col = 0;
        if (input_rank == 1) {
          uint32_t coord = mapped_output_coord(dim0, output_row, output_col);
          src_col = input_coord(input_cols, coord) & (kTileCols - 1);
        } else if (input_rank == 2) {
          uint32_t coord0 = mapped_output_coord(dim0, output_row, output_col);
          uint32_t coord1 = mapped_output_coord(dim1, output_row, output_col);
          src_row = input_coord(input_rows, coord0) & (kTileRows - 1);
          src_col = input_coord(input_cols, coord1) & (kTileCols - 1);
        }

        uint32_t src_element = tile_element_offset(src_row, src_col);
        uint32_t dst_element = tile_element_offset(row, col);
        copy_element(input_l1, output_l1, src_element, dst_element, element_bytes);
      }
    }

    noc_async_write_tile(output_tile_id, output, output_l1);
    noc_async_write_barrier();

    cb_push_back(cb_input, 1);
    cb_pop_front(cb_input, 1);
    cb_push_back(cb_output, 1);
    cb_pop_front(cb_output, 1);
  }
}
