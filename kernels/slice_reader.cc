#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t RANK = SLICE_RANK;
constexpr uint32_t COORD_COUNT = RANK == 0 ? 1 : RANK;
constexpr uint32_t INPUT_SHAPE[COORD_COUNT] = SLICE_INPUT_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[COORD_COUNT] = SLICE_OUTPUT_SHAPE;
constexpr uint32_t START_INDICES[COORD_COUNT] = SLICE_START_INDICES;
constexpr uint32_t STRIDES[COORD_COUNT] = SLICE_STRIDES;
constexpr uint32_t INPUT_TILE_ROWS = SLICE_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = SLICE_INPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = SLICE_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = SLICE_OUTPUT_TILES_PER_ROW;
constexpr bool DIRECT_COPY = SLICE_DIRECT_COPY != 0;
using Element = SLICE_ELEMENT_TYPE;

struct Location {
  uint32_t tile;
  uint32_t row;
  uint32_t col;
};

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

void read_output_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                      uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
}

constexpr bool last_dim_tile_copy() {
  if constexpr (RANK < 2) {
    return false;
  }
  for (uint32_t dim = 0; dim < RANK - 1; ++dim) {
    if (START_INDICES[dim] != 0 || STRIDES[dim] != 1 ||
        INPUT_SHAPE[dim] != OUTPUT_SHAPE[dim]) {
      return false;
    }
  }
  return STRIDES[RANK - 1] == 1 && START_INDICES[RANK - 1] % TILE_C == 0 &&
         OUTPUT_SHAPE[RANK - 1] % TILE_C == 0;
}

constexpr bool LAST_DIM_TILE_COPY = last_dim_tile_copy();

void copy_element(uint32_t cb_input, uint32_t cb_output, uint32_t source_row,
                  uint32_t source_col, uint32_t output_row, uint32_t output_col) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  output[tile_element_index(output_row, output_col)] =
      source[tile_element_index(source_row, source_col)];
}

void decode_output_batch(uint32_t output_batch, uint32_t output_coords[COORD_COUNT]) {
  for (uint32_t dim = 0; dim < RANK; ++dim) {
    output_coords[dim] = 0;
  }
  if constexpr (RANK >= 3) {
    for (uint32_t index = 0; index < RANK - 2; ++index) {
      uint32_t dim = RANK - 3 - index;
      output_coords[dim] = output_batch % OUTPUT_SHAPE[dim];
      output_batch /= OUTPUT_SHAPE[dim];
    }
  }
}

uint32_t output_coord(uint32_t dim, const uint32_t base_output_coords[COORD_COUNT],
                      uint32_t output_row, uint32_t output_col) {
  if constexpr (RANK == 0) {
    return 0;
  } else if constexpr (RANK == 1) {
    return output_col;
  } else {
    if (dim == RANK - 1) {
      return output_col;
    }
    if (dim == RANK - 2) {
      return output_row;
    }
    return base_output_coords[dim];
  }
}

Location input_location(const uint32_t base_output_coords[COORD_COUNT],
                        uint32_t output_row, uint32_t output_col) {
  if constexpr (RANK == 0) {
    return Location{0, 0, 0};
  } else if constexpr (RANK == 1) {
    uint32_t input_col = START_INDICES[0] + output_col * STRIDES[0];
    return Location{input_col / TILE_C, 0, input_col % TILE_C};
  } else {
    uint32_t input_batch = 0;
    for (uint32_t dim = 0; dim < RANK - 2; ++dim) {
      uint32_t coord =
          START_INDICES[dim] +
          output_coord(dim, base_output_coords, output_row, output_col) * STRIDES[dim];
      input_batch = input_batch * INPUT_SHAPE[dim] + coord;
    }
    uint32_t input_row = START_INDICES[RANK - 2] + output_row * STRIDES[RANK - 2];
    uint32_t input_col = START_INDICES[RANK - 1] + output_col * STRIDES[RANK - 1];
    uint32_t input_tile_row = input_row / TILE_R;
    uint32_t input_tile_col = input_col / TILE_C;
    uint32_t input_tile =
        (input_batch * INPUT_TILE_ROWS + input_tile_row) * INPUT_TILES_PER_ROW +
        input_tile_col;
    return Location{input_tile, input_row % TILE_R, input_col % TILE_C};
  }
}

void ensure_input_tile(const InterleavedAddrGenFast<true> &input, uint32_t requested_tile,
                       uint32_t *loaded_tile) {
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_input, 1);
  }
  read_input_tile(input, requested_tile, cb_input);
  *loaded_tile = requested_tile;
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
    if constexpr (DIRECT_COPY) {
      read_output_tile(input, output_tile_id, cb_output);
      continue;
    }

    uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t row_count = 1;
    uint32_t col_count = 1;

    if constexpr (LAST_DIM_TILE_COPY) {
      uint32_t input_tile_col = START_INDICES[RANK - 1] / TILE_C + output_tile_col;
      uint32_t input_tile =
          (output_batch * INPUT_TILE_ROWS + output_tile_row) * INPUT_TILES_PER_ROW +
          input_tile_col;
      read_output_tile(input, input_tile, cb_output);
      continue;
    }

    if constexpr (RANK == 1) {
      row_count = 1;
      col_count = tile_extent(OUTPUT_SHAPE[0], output_col_base, TILE_C);
    } else if constexpr (RANK >= 2) {
      row_count = tile_extent(OUTPUT_SHAPE[RANK - 2], output_row_base, TILE_R);
      col_count = tile_extent(OUTPUT_SHAPE[RANK - 1], output_col_base, TILE_C);
    }

    uint32_t base_output_coords[COORD_COUNT];
    decode_output_batch(output_batch, base_output_coords);

    uint32_t loaded_input_tile = INVALID_TILE;
    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

    for (uint32_t row = 0; row < row_count; ++row) {
      uint32_t output_row = output_row_base + row;
      uint32_t col = 0;
      while (col < col_count) {
        uint32_t output_col = output_col_base + col;
        Location source = input_location(base_output_coords, output_row, output_col);
        ensure_input_tile(input, source.tile, &loaded_input_tile);

        uint32_t run = 1;
        bool contiguous_cols = false;
        if (col + 1 < col_count) {
          Location next_source = input_location(base_output_coords, output_row, output_col + 1);
          if (next_source.tile == source.tile && next_source.row == source.row &&
              next_source.col == source.col + 1) {
            contiguous_cols = true;
            uint32_t source_cols_remaining = TILE_C - source.col;
            uint32_t output_cols_remaining = col_count - col;
            run = source_cols_remaining < output_cols_remaining ? source_cols_remaining
                                                                : output_cols_remaining;
          }
        }

        for (uint32_t i = 0; i < run; ++i) {
          copy_element(cb_input, cb_output, source.row,
                       contiguous_cols ? source.col + i : source.col, row, col + i);
        }
        col += run;
      }
    }

    if (loaded_input_tile != INVALID_TILE) {
      cb_pop_front(cb_input, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
