#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t RANK = BITWISE_REDUCE_RANK;
constexpr uint32_t OUT_RANK = RANK - 1;
constexpr uint32_t COORD_COUNT = RANK == 0 ? 1 : RANK;
constexpr uint32_t OUT_COORD_COUNT = OUT_RANK == 0 ? 1 : OUT_RANK;
constexpr uint32_t INPUT_SHAPE[COORD_COUNT] = BITWISE_REDUCE_INPUT_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[OUT_COORD_COUNT] = BITWISE_REDUCE_OUTPUT_SHAPE;
constexpr uint32_t INPUT_TILE_ROWS = BITWISE_REDUCE_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = BITWISE_REDUCE_INPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = BITWISE_REDUCE_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = BITWISE_REDUCE_OUTPUT_TILES_PER_ROW;
constexpr uint32_t OP_AND = 0;
constexpr uint32_t OP_OR = 1;
constexpr uint32_t OP = BITWISE_REDUCE_OP;
using Element = BITWISE_REDUCE_ELEMENT_TYPE;
constexpr Element IDENTITY = static_cast<Element>(BITWISE_REDUCE_IDENTITY);

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

Element apply(Element lhs, Element rhs) {
  if constexpr (OP == OP_AND) {
    return lhs & rhs;
  } else {
    return lhs | rhs;
  }
}

uint32_t tile_extent(uint32_t logical_dim, uint32_t base, uint32_t tile_dim) {
  if (base >= logical_dim) {
    return 0;
  }
  uint32_t remaining = logical_dim - base;
  return remaining < tile_dim ? remaining : tile_dim;
}

void fill_tile(uint32_t cb, Element value) {
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb));
  uint32_t elements = get_tile_size(cb) / sizeof(Element);
  for (uint32_t i = 0; i < elements; ++i) {
    ptr[i] = value;
  }
}

void read_tile_to_cb(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                     uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
  cb_wait_front(cb, 1);
}

void decode_output_batch(uint32_t batch, uint32_t coords[OUT_COORD_COUNT]) {
  for (uint32_t dim = 0; dim < OUT_RANK; ++dim) {
    coords[dim] = 0;
  }
  if constexpr (OUT_RANK >= 3) {
    for (uint32_t index = 0; index < OUT_RANK - 2; ++index) {
      uint32_t dim = OUT_RANK - 3 - index;
      coords[dim] = batch % OUTPUT_SHAPE[dim];
      batch /= OUTPUT_SHAPE[dim];
    }
  }
}

uint32_t output_coord(uint32_t dim, const uint32_t base_coords[OUT_COORD_COUNT],
                      uint32_t output_row, uint32_t output_col) {
  if constexpr (OUT_RANK == 1) {
    return output_col;
  } else {
    if (dim == OUT_RANK - 1) {
      return output_col;
    }
    if (dim == OUT_RANK - 2) {
      return output_row;
    }
    return base_coords[dim];
  }
}

Location input_location(const uint32_t coords[COORD_COUNT]) {
  if constexpr (RANK == 1) {
    return Location{coords[0] / TILE_C, 0, coords[0] % TILE_C};
  } else {
    uint32_t batch = 0;
    if constexpr (RANK >= 3) {
      for (uint32_t dim = 0; dim < RANK - 2; ++dim) {
        batch = batch * INPUT_SHAPE[dim] + coords[dim];
      }
    }
    uint32_t row = coords[RANK - 2];
    uint32_t col = coords[RANK - 1];
    uint32_t tile_row = row / TILE_R;
    uint32_t tile_col = col / TILE_C;
    uint32_t tile = (batch * INPUT_TILE_ROWS + tile_row) * INPUT_TILES_PER_ROW + tile_col;
    return Location{tile, row % TILE_R, col % TILE_C};
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
  read_tile_to_cb(input, requested_tile, cb_input);
  *loaded_tile = requested_tile;
}

Element read_input_element(const InterleavedAddrGenFast<true> &input,
                           const uint32_t coords[COORD_COUNT],
                           uint32_t *loaded_input_tile) {
  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  Location source = input_location(coords);
  ensure_input_tile(input, source.tile, loaded_input_tile);
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_input));
  return ptr[tile_element_index(source.row, source.col)];
}

void write_output_element(uint32_t row, uint32_t col, Element value) {
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  ptr[tile_element_index(row, col)] = value;
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

    if constexpr (OUT_RANK == 1) {
      col_count = tile_extent(OUTPUT_SHAPE[0], output_col_base, TILE_C);
    } else {
      row_count = tile_extent(OUTPUT_SHAPE[OUT_RANK - 2], output_row_base, TILE_R);
      col_count = tile_extent(OUTPUT_SHAPE[OUT_RANK - 1], output_col_base, TILE_C);
    }

    uint32_t base_output_coords[OUT_COORD_COUNT];
    decode_output_batch(output_batch, base_output_coords);

    cb_reserve_back(cb_output, 1);
    fill_tile(cb_output, IDENTITY);

    uint32_t loaded_input_tile = INVALID_TILE;
    for (uint32_t row = 0; row < row_count; ++row) {
      uint32_t output_row = output_row_base + row;
      for (uint32_t col = 0; col < col_count; ++col) {
        uint32_t output_col = output_col_base + col;
        uint32_t input_coords[COORD_COUNT];
        for (uint32_t dim = 0; dim < OUT_RANK; ++dim) {
          input_coords[dim] = output_coord(dim, base_output_coords, output_row, output_col);
        }

        Element value = IDENTITY;
        for (uint32_t reduce_index = 0; reduce_index < INPUT_SHAPE[RANK - 1]; ++reduce_index) {
          input_coords[RANK - 1] = reduce_index;
          value = apply(value, read_input_element(input, input_coords, &loaded_input_tile));
        }
        write_output_element(row, col, value);
      }
    }

    if (loaded_input_tile != INVALID_TILE) {
      cb_pop_front(cb_input, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
