#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
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

void fill_row(uint32_t cb_output, uint32_t output_row, uint32_t output_col,
              uint32_t count, Element value) {
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  if (count == TILE_C && output_col == 0) {
    uint32_t output_face0 = tile_element_index(output_row, 0);
    uint32_t output_face1 = tile_element_index(output_row, FACE_C);
    for (uint32_t col = 0; col < FACE_C; ++col) {
      output[output_face0 + col] = value;
      output[output_face1 + col] = value;
    }
    return;
  }
  for (uint32_t col = 0; col < count; ++col) {
    output[tile_element_index(output_row, output_col + col)] = value;
  }
}

void decode_output_batch(uint32_t output_batch, uint32_t output_coords[OUTPUT_COORD_COUNT]) {
  for (uint32_t dim = 0; dim < OUTPUT_RANK; ++dim) {
    output_coords[dim] = 0;
  }
  if constexpr (OUTPUT_RANK >= 3) {
    for (uint32_t index = 0; index < OUTPUT_RANK - 2; ++index) {
      uint32_t dim = OUTPUT_RANK - 3 - index;
      output_coords[dim] = output_batch % OUTPUT_SHAPE[dim];
      output_batch /= OUTPUT_SHAPE[dim];
    }
  }
}

constexpr bool column_fill_broadcast() {
  if constexpr (INPUT_RANK < 2 || OUTPUT_RANK < 2 || INPUT_RANK != OUTPUT_RANK) {
    return false;
  }
  for (uint32_t dim = 0; dim < INPUT_RANK; ++dim) {
    if (BROADCAST_DIMS[dim] != dim) {
      return false;
    }
  }
  for (uint32_t dim = 0; dim < INPUT_RANK - 1; ++dim) {
    if (INPUT_SHAPE[dim] != OUTPUT_SHAPE[dim]) {
      return false;
    }
  }
  return INPUT_SHAPE[INPUT_RANK - 1] == 1 && OUTPUT_SHAPE[OUTPUT_RANK - 1] > 1;
}

constexpr bool COLUMN_FILL = column_fill_broadcast();

constexpr bool direct_full_tile_mapping() {
  if constexpr (INPUT_RANK < 2 || OUTPUT_RANK < 2) {
    return false;
  }
  return BROADCAST_DIMS[INPUT_RANK - 2] == OUTPUT_RANK - 2 &&
         BROADCAST_DIMS[INPUT_RANK - 1] == OUTPUT_RANK - 1 &&
         INPUT_SHAPE[INPUT_RANK - 2] == OUTPUT_SHAPE[OUTPUT_RANK - 2] &&
         INPUT_SHAPE[INPUT_RANK - 1] == OUTPUT_SHAPE[OUTPUT_RANK - 1];
}

constexpr bool DIRECT_FULL_TILE = direct_full_tile_mapping();

uint32_t output_coord(uint32_t dim, const uint32_t base_output_coords[OUTPUT_COORD_COUNT],
                      uint32_t output_row, uint32_t output_col) {
  if constexpr (OUTPUT_RANK == 0) {
    return 0;
  } else if constexpr (OUTPUT_RANK == 1) {
    return output_col;
  } else {
    if (dim == OUTPUT_RANK - 1) {
      return output_col;
    }
    if (dim == OUTPUT_RANK - 2) {
      return output_row;
    }
    return base_output_coords[dim];
  }
}

uint32_t input_coord(uint32_t dim, const uint32_t base_output_coords[OUTPUT_COORD_COUNT],
                     uint32_t output_row, uint32_t output_col) {
  if (INPUT_SHAPE[dim] == 1) {
    return 0;
  }
  return output_coord(BROADCAST_DIMS[dim], base_output_coords, output_row, output_col);
}

Location input_location(const uint32_t base_output_coords[OUTPUT_COORD_COUNT],
                        uint32_t output_row, uint32_t output_col) {
  if constexpr (INPUT_RANK == 0) {
    return Location{0, 0, 0};
  } else if constexpr (INPUT_RANK == 1) {
    uint32_t input_col = input_coord(0, base_output_coords, output_row, output_col);
    return Location{input_col / TILE_C, 0, input_col % TILE_C};
  } else {
    uint32_t input_batch = 0;
    for (uint32_t dim = 0; dim < INPUT_RANK - 2; ++dim) {
      uint32_t coord = input_coord(dim, base_output_coords, output_row, output_col);
      input_batch = input_batch * INPUT_SHAPE[dim] + coord;
    }
    uint32_t input_row =
        input_coord(INPUT_RANK - 2, base_output_coords, output_row, output_col);
    uint32_t input_col =
        input_coord(INPUT_RANK - 1, base_output_coords, output_row, output_col);
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
    uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t row_count = 1;
    uint32_t col_count = 1;

    if constexpr (OUTPUT_RANK == 1) {
      row_count = 1;
      col_count = tile_extent(OUTPUT_SHAPE[0], output_col_base, TILE_C);
    } else if constexpr (OUTPUT_RANK >= 2) {
      row_count = tile_extent(OUTPUT_SHAPE[OUTPUT_RANK - 2], output_row_base, TILE_R);
      col_count = tile_extent(OUTPUT_SHAPE[OUTPUT_RANK - 1], output_col_base, TILE_C);
    }

    if constexpr (COLUMN_FILL) {
      uint32_t input_tile =
          (output_batch * INPUT_TILE_ROWS + output_tile_row) * INPUT_TILES_PER_ROW;
      read_input_tile(input, input_tile, cb_input);

      cb_reserve_back(cb_output, 1);
      zero_tile(cb_output);
      volatile tt_l1_ptr Element *source =
          reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
      for (uint32_t row = 0; row < row_count; ++row) {
        const Element value = source[tile_element_index(row, 0)];
        fill_row(cb_output, row, 0, col_count, value);
      }
      cb_pop_front(cb_input, 1);
      cb_push_back(cb_output, 1);
      continue;
    }

    uint32_t base_output_coords[OUTPUT_COORD_COUNT];
    decode_output_batch(output_batch, base_output_coords);

    if constexpr (DIRECT_FULL_TILE) {
      Location source = input_location(base_output_coords, output_row_base, output_col_base);
      if (source.row == 0 && source.col == 0) {
        read_output_tile(input, source.tile, cb_output);
        continue;
      }
    }

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
          if (next_source.tile == source.tile && next_source.row == source.row) {
            if (next_source.col == source.col + 1) {
              contiguous_cols = true;
              uint32_t source_cols_remaining = TILE_C - source.col;
              uint32_t output_cols_remaining = col_count - col;
              run = source_cols_remaining < output_cols_remaining ? source_cols_remaining
                                                                  : output_cols_remaining;
            } else if (next_source.col == source.col) {
              run = col_count - col;
            }
          }
        }

        if (contiguous_cols) {
          copy_row(cb_input, cb_output, source.row, source.col, row, col, run);
        } else if (run > 1) {
          volatile tt_l1_ptr Element *source_ptr =
              reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
          Element value = source_ptr[tile_element_index(source.row, source.col)];
          fill_row(cb_output, row, col, run, value);
        } else {
          copy_element(cb_input, cb_output, source.row, source.col, row, col);
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
