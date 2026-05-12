#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INPUT_RANK = BROADCAST_INPUT_RANK;
constexpr uint32_t OUTPUT_RANK = BROADCAST_OUTPUT_RANK;
constexpr uint32_t INPUT_COORD_COUNT = INPUT_RANK == 0 ? 1 : INPUT_RANK;
constexpr uint32_t OUTPUT_COORD_COUNT = OUTPUT_RANK == 0 ? 1 : OUTPUT_RANK;
constexpr uint32_t INPUT_SHAPE[INPUT_COORD_COUNT] = BROADCAST_INPUT_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[OUTPUT_COORD_COUNT] = BROADCAST_OUTPUT_SHAPE;
constexpr uint32_t BROADCAST_DIMS[INPUT_COORD_COUNT] = BROADCAST_DIMENSIONS;
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

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb));
  uint32_t elements = get_tile_size(cb) / sizeof(Element);
  for (uint32_t i = 0; i < elements; ++i) {
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

bool output_logical_coords(uint32_t output_tile_id, uint32_t row, uint32_t col,
                           uint32_t *coords) {
  constexpr uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
  uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
  uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
  uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
  uint32_t output_row = output_tile_row * TILE_R + row;
  uint32_t output_col = output_tile_col * TILE_C + col;

  if constexpr (OUTPUT_RANK == 0) {
    return output_tile_id == 0 && row == 0 && col == 0;
  } else if constexpr (OUTPUT_RANK == 1) {
    if (row != 0 || output_col >= OUTPUT_SHAPE[0]) {
      return false;
    }
    coords[0] = output_col;
    return true;
  } else {
    if (output_row >= OUTPUT_SHAPE[OUTPUT_RANK - 2] ||
        output_col >= OUTPUT_SHAPE[OUTPUT_RANK - 1]) {
      return false;
    }

    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    for (uint32_t dim = OUTPUT_RANK - 2; dim > 0; --dim) {
      uint32_t index = dim - 1;
      coords[index] = output_batch % OUTPUT_SHAPE[index];
      output_batch /= OUTPUT_SHAPE[index];
    }
    coords[OUTPUT_RANK - 2] = output_row;
    coords[OUTPUT_RANK - 1] = output_col;
    return true;
  }
}

void input_physical_location(const uint32_t *output_coords, uint32_t &tile_id,
                             uint32_t &source_row, uint32_t &source_col) {
  uint32_t input_coords[INPUT_COORD_COUNT];
  for (uint32_t dim = 0; dim < INPUT_RANK; ++dim) {
    uint32_t output_dim = BROADCAST_DIMS[dim];
    uint32_t output_coord = output_coords[output_dim];
    input_coords[dim] = INPUT_SHAPE[dim] == 1 ? 0 : output_coord;
  }

  uint32_t batch = 0;
  uint32_t row = 0;
  uint32_t col = 0;
  if constexpr (INPUT_RANK == 0) {
    row = 0;
    col = 0;
  } else if constexpr (INPUT_RANK == 1) {
    row = 0;
    col = input_coords[0];
  } else {
    for (uint32_t dim = 0; dim < INPUT_RANK - 2; ++dim) {
      batch = batch * INPUT_SHAPE[dim] + input_coords[dim];
    }
    row = input_coords[INPUT_RANK - 2];
    col = input_coords[INPUT_RANK - 1];
  }

  tile_id = (batch * INPUT_TILE_ROWS + row / TILE_R) * INPUT_TILES_PER_ROW + col / TILE_C;
  source_row = row % TILE_R;
  source_col = col % TILE_C;
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
    uint32_t loaded_input_tile = 0xffffffffu;

    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

    for (uint32_t row = 0; row < TILE_R; ++row) {
      for (uint32_t col = 0; col < TILE_C; ++col) {
        uint32_t output_coords[OUTPUT_COORD_COUNT];
        if (!output_logical_coords(output_tile_id, row, col, output_coords)) {
          continue;
        }

        uint32_t input_tile = 0;
        uint32_t source_row = 0;
        uint32_t source_col = 0;
        input_physical_location(output_coords, input_tile, source_row, source_col);

        if (input_tile != loaded_input_tile) {
          if (loaded_input_tile != 0xffffffffu) {
            cb_pop_front(cb_input, 1);
          }
          read_input_tile(input, input_tile, cb_input);
          loaded_input_tile = input_tile;
        }

        copy_element(cb_input, cb_output, source_row, source_col, row, col);
      }
    }

    if (loaded_input_tile != 0xffffffffu) {
      cb_pop_front(cb_input, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
