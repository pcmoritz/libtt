#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t INPUT_RANK = SLICE_INPUT_RANK;
constexpr uint32_t OUTPUT_RANK = SLICE_OUTPUT_RANK;
constexpr uint32_t INPUT_COORD_COUNT = INPUT_RANK < 3 ? 3 : INPUT_RANK;
constexpr uint32_t OUTPUT_COORD_COUNT = OUTPUT_RANK < 3 ? 3 : OUTPUT_RANK;
constexpr uint32_t INPUT_SHAPE[INPUT_COORD_COUNT] = SLICE_INPUT_SHAPE;
constexpr uint32_t SLICE_SHAPE[INPUT_COORD_COUNT] = SLICE_SLICE_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[OUTPUT_COORD_COUNT] = SLICE_OUTPUT_SHAPE;
constexpr uint32_t START_INDICES[INPUT_COORD_COUNT] = SLICE_START_INDICES;
constexpr uint32_t STRIDES[INPUT_COORD_COUNT] = SLICE_STRIDES;
constexpr uint32_t INPUT_TILE_ROWS = SLICE_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = SLICE_INPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = SLICE_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = SLICE_OUTPUT_TILES_PER_ROW;
constexpr bool TRANSPOSE_RANK2 = SLICE_TRANSPOSE_RANK2 != 0;
constexpr bool HEAD_SLICE_RESHAPE =
    !TRANSPOSE_RANK2 && INPUT_RANK == 3 && OUTPUT_RANK == 2 && SLICE_SHAPE[1] == 1 &&
    OUTPUT_SHAPE[0] == SLICE_SHAPE[0] && OUTPUT_SHAPE[1] == SLICE_SHAPE[2] &&
    STRIDES[0] == 1 && STRIDES[1] == 1 && STRIDES[2] == 1;
constexpr bool HEAD_SLICE_RESHAPE_TRANSPOSE =
    TRANSPOSE_RANK2 && INPUT_RANK == 3 && OUTPUT_RANK == 2 && SLICE_SHAPE[1] == 1 &&
    OUTPUT_SHAPE[0] == SLICE_SHAPE[2] && OUTPUT_SHAPE[1] == SLICE_SHAPE[0] &&
    STRIDES[0] == 1 && STRIDES[1] == 1 && STRIDES[2] == 1;
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

void copy_element(uint32_t cb_input, uint32_t cb_output, uint32_t source_row,
                  uint32_t source_col, uint32_t output_row, uint32_t output_col) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  output[tile_element_index(output_row, output_col)] =
      source[tile_element_index(source_row, source_col)];
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

void decode_output_batch(uint32_t output_batch,
                         uint32_t output_coords[OUTPUT_COORD_COUNT]) {
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

uint32_t output_coord(uint32_t dim,
                      const uint32_t base_output_coords[OUTPUT_COORD_COUNT],
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

uint32_t output_flat_index(const uint32_t base_output_coords[OUTPUT_COORD_COUNT],
                           uint32_t output_row, uint32_t output_col) {
  if constexpr (OUTPUT_RANK == 0) {
    return 0;
  } else {
    uint32_t flat = 0;
    for (uint32_t dim = 0; dim < OUTPUT_RANK; ++dim) {
      flat = flat * OUTPUT_SHAPE[dim] +
             output_coord(dim, base_output_coords, output_row, output_col);
    }
    return flat;
  }
}

void decode_slice_coords(uint32_t flat, uint32_t slice_coords[INPUT_COORD_COUNT]) {
  for (uint32_t dim = 0; dim < INPUT_RANK; ++dim) {
    slice_coords[dim] = 0;
  }
  if constexpr (INPUT_RANK > 0) {
    for (uint32_t index = 0; index < INPUT_RANK; ++index) {
      uint32_t dim = INPUT_RANK - 1 - index;
      slice_coords[dim] = flat % SLICE_SHAPE[dim];
      flat /= SLICE_SHAPE[dim];
    }
  }
}

Location input_location(uint32_t flat) {
  if constexpr (INPUT_RANK == 0) {
    return Location{0, 0, 0};
  } else {
    uint32_t slice_coords[INPUT_COORD_COUNT];
    decode_slice_coords(flat, slice_coords);

    if constexpr (INPUT_RANK == 1) {
      uint32_t input_col = START_INDICES[0] + slice_coords[0] * STRIDES[0];
      return Location{input_col / TILE_C, 0, input_col % TILE_C};
    } else {
      uint32_t input_batch = 0;
      for (uint32_t dim = 0; dim < INPUT_RANK - 2; ++dim) {
        uint32_t coord = START_INDICES[dim] + slice_coords[dim] * STRIDES[dim];
        input_batch = input_batch * INPUT_SHAPE[dim] + coord;
      }
      uint32_t input_row =
          START_INDICES[INPUT_RANK - 2] +
          slice_coords[INPUT_RANK - 2] * STRIDES[INPUT_RANK - 2];
      uint32_t input_col =
          START_INDICES[INPUT_RANK - 1] +
          slice_coords[INPUT_RANK - 1] * STRIDES[INPUT_RANK - 1];
      uint32_t input_tile_row = input_row / TILE_R;
      uint32_t input_tile_col = input_col / TILE_C;
      uint32_t input_tile =
          (input_batch * INPUT_TILE_ROWS + input_tile_row) * INPUT_TILES_PER_ROW +
          input_tile_col;
      return Location{input_tile, input_row % TILE_R, input_col % TILE_C};
    }
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

    if constexpr (HEAD_SLICE_RESHAPE) {
      uint32_t loaded_input_tile = INVALID_TILE;
      cb_reserve_back(cb_output, 1);
      zero_tile(cb_output);

      for (uint32_t row = 0; row < row_count; ++row) {
        uint32_t output_row = output_row_base + row;
        uint32_t input_batch = START_INDICES[0] + output_row;
        uint32_t input_row = START_INDICES[1];
        uint32_t input_tile_row = input_row / TILE_R;
        uint32_t source_row = input_row % TILE_R;
        uint32_t col = 0;
        while (col < col_count) {
          uint32_t input_col = START_INDICES[2] + output_col_base + col;
          uint32_t input_tile_col = input_col / TILE_C;
          uint32_t input_tile =
              (input_batch * INPUT_TILE_ROWS + input_tile_row) * INPUT_TILES_PER_ROW +
              input_tile_col;
          ensure_input_tile(input, input_tile, &loaded_input_tile);
          uint32_t run = TILE_C - (input_col % TILE_C);
          uint32_t remaining = col_count - col;
          run = run < remaining ? run : remaining;
          copy_row_segment(cb_input, cb_output, source_row, input_col % TILE_C, row, col, run);
          col += run;
        }
      }

      if (loaded_input_tile != INVALID_TILE) {
        cb_pop_front(cb_input, 1);
      }
      cb_push_back(cb_output, 1);
      continue;
    }

    if constexpr (HEAD_SLICE_RESHAPE_TRANSPOSE) {
      uint32_t loaded_input_tile = INVALID_TILE;
      cb_reserve_back(cb_output, 1);
      zero_tile(cb_output);

      uint32_t input_tile_col = (START_INDICES[2] + output_row_base) / TILE_C;
      uint32_t input_row = START_INDICES[1];
      uint32_t input_tile_row = input_row / TILE_R;
      uint32_t source_row = input_row % TILE_R;

      for (uint32_t col = 0; col < col_count; ++col) {
        uint32_t output_col = output_col_base + col;
        uint32_t input_batch = START_INDICES[0] + output_col;
        uint32_t input_tile =
            (input_batch * INPUT_TILE_ROWS + input_tile_row) * INPUT_TILES_PER_ROW +
            input_tile_col;
        ensure_input_tile(input, input_tile, &loaded_input_tile);
        for (uint32_t row = 0; row < row_count; ++row) {
          uint32_t input_col = START_INDICES[2] + output_row_base + row;
          copy_element(cb_input, cb_output, source_row, input_col % TILE_C, row, col);
        }
      }

      if (loaded_input_tile != INVALID_TILE) {
        cb_pop_front(cb_input, 1);
      }
      cb_push_back(cb_output, 1);
      continue;
    }

    uint32_t base_output_coords[OUTPUT_COORD_COUNT];
    decode_output_batch(output_batch, base_output_coords);

    uint32_t loaded_input_tile = INVALID_TILE;
    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

    for (uint32_t row = 0; row < row_count; ++row) {
      uint32_t output_row = output_row_base + row;
      for (uint32_t col = 0; col < col_count; ++col) {
        uint32_t output_col = output_col_base + col;
        uint32_t flat = output_flat_index(base_output_coords, output_row, output_col);
        Location source = input_location(flat);
        ensure_input_tile(input, source.tile, &loaded_input_tile);
        copy_element(cb_input, cb_output, source.row, source.col, row, col);
      }
    }

    if (loaded_input_tile != INVALID_TILE) {
      cb_pop_front(cb_input, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
