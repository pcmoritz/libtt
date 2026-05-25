#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t RANK = GATHER_DIM0_RANK;
constexpr uint32_t AXIS = GATHER_DIM0_AXIS;
constexpr uint32_t COORD_COUNT = RANK == 0 ? 1 : RANK;
constexpr uint32_t OPERAND_SHAPE[COORD_COUNT] = GATHER_DIM0_OPERAND_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[COORD_COUNT] = GATHER_DIM0_OUTPUT_SHAPE;
constexpr uint32_t OPERAND_TILE_ROWS = GATHER_DIM0_OPERAND_TILE_ROWS;
constexpr uint32_t OPERAND_TILES_PER_ROW = GATHER_DIM0_OPERAND_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = GATHER_DIM0_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = GATHER_DIM0_OUTPUT_TILES_PER_ROW;
using Element = GATHER_DIM0_ELEMENT_TYPE;

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

void decode_batch(uint32_t batch, const uint32_t shape[COORD_COUNT],
                  uint32_t coords[COORD_COUNT]) {
  for (uint32_t dim = 0; dim < RANK; ++dim) {
    coords[dim] = 0;
  }
  if constexpr (RANK >= 3) {
    for (uint32_t index = 0; index < RANK - 2; ++index) {
      uint32_t dim = RANK - 3 - index;
      coords[dim] = batch % shape[dim];
      batch /= shape[dim];
    }
  }
}

uint32_t output_coord(uint32_t dim, const uint32_t base_coords[COORD_COUNT],
                      uint32_t output_row, uint32_t output_col) {
  if constexpr (RANK == 1) {
    return output_col;
  } else {
    if (dim == RANK - 1) {
      return output_col;
    }
    if (dim == RANK - 2) {
      return output_row;
    }
    return base_coords[dim];
  }
}

Location tensor_location(const uint32_t shape[COORD_COUNT], uint32_t tile_rows,
                         uint32_t tiles_per_row,
                         const uint32_t coords[COORD_COUNT]) {
  if constexpr (RANK == 1) {
    return Location{coords[0] / TILE_C, 0, coords[0] % TILE_C};
  } else {
    uint32_t batch = 0;
    if constexpr (RANK >= 3) {
      for (uint32_t dim = 0; dim < RANK - 2; ++dim) {
        batch = batch * shape[dim] + coords[dim];
      }
    }
    uint32_t row = coords[RANK - 2];
    uint32_t col = coords[RANK - 1];
    uint32_t tile_row = row / TILE_R;
    uint32_t tile_col = col / TILE_C;
    uint32_t tile = (batch * tile_rows + tile_row) * tiles_per_row + tile_col;
    return Location{tile, row % TILE_R, col % TILE_C};
  }
}

void ensure_index_tile(const InterleavedAddrGenFast<true> &indices,
                       uint32_t requested_tile, uint32_t *loaded_tile) {
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_indices, 1);
  }
  read_tile_to_cb(indices, requested_tile, cb_indices);
  *loaded_tile = requested_tile;
}

int32_t read_gather_index(const InterleavedAddrGenFast<true> &indices,
                          uint32_t output_index, uint32_t *loaded_index_tile) {
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  uint32_t tile = output_index / TILE_R;
  ensure_index_tile(indices, tile, loaded_index_tile);
  volatile tt_l1_ptr int32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_read_ptr(cb_indices));
  return ptr[tile_element_index(output_index % TILE_R, 0)];
}

void ensure_operand_tile(const InterleavedAddrGenFast<true> &operand,
                         uint32_t requested_tile, uint32_t *loaded_tile) {
  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_operand, 1);
  }
  read_tile_to_cb(operand, requested_tile, cb_operand);
  *loaded_tile = requested_tile;
}

void copy_operand_element(uint32_t source_row, uint32_t source_col,
                          uint32_t output_row, uint32_t output_col) {
  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_operand));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  output[tile_element_index(output_row, output_col)] =
      source[tile_element_index(source_row, source_col)];
}

}  // namespace

void kernel_main() {
  uint32_t operand_addr = get_arg_val<uint32_t>(0);
  uint32_t start_indices_addr = get_arg_val<uint32_t>(1);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_tile_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  const InterleavedAddrGenFast<true> operand = {
      .bank_base_address = operand_addr,
      .page_size = get_tile_size(cb_operand),
      .data_format = get_dataformat(cb_operand),
  };
  const InterleavedAddrGenFast<true> indices = {
      .bank_base_address = start_indices_addr,
      .page_size = get_tile_size(cb_indices),
      .data_format = get_dataformat(cb_indices),
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

    if constexpr (RANK == 1) {
      col_count = tile_extent(OUTPUT_SHAPE[0], output_col_base, TILE_C);
    } else {
      row_count = tile_extent(OUTPUT_SHAPE[RANK - 2], output_row_base, TILE_R);
      col_count = tile_extent(OUTPUT_SHAPE[RANK - 1], output_col_base, TILE_C);
    }

    uint32_t base_coords[COORD_COUNT];
    decode_batch(output_batch, OUTPUT_SHAPE, base_coords);

    uint32_t loaded_index_tile = INVALID_TILE;
    uint32_t loaded_operand_tile = INVALID_TILE;
    cb_reserve_back(cb_output, 1);
    fill_tile(cb_output, 0);

    for (uint32_t row = 0; row < row_count; ++row) {
      uint32_t output_row = output_row_base + row;
      for (uint32_t col = 0; col < col_count; ++col) {
        uint32_t output_col = output_col_base + col;
        uint32_t gather_output_index =
            output_coord(AXIS, base_coords, output_row, output_col);
        int32_t gather_index =
            read_gather_index(indices, gather_output_index, &loaded_index_tile);
        if (gather_index < 0 ||
            static_cast<uint32_t>(gather_index) >= OPERAND_SHAPE[AXIS]) {
          continue;
        }

        uint32_t operand_coords[COORD_COUNT];
        for (uint32_t dim = 0; dim < RANK; ++dim) {
          operand_coords[dim] =
              dim == AXIS
                  ? static_cast<uint32_t>(gather_index)
                  : output_coord(dim, base_coords, output_row, output_col);
        }

        Location source =
            tensor_location(OPERAND_SHAPE, OPERAND_TILE_ROWS, OPERAND_TILES_PER_ROW,
                            operand_coords);
        ensure_operand_tile(operand, source.tile, &loaded_operand_tile);
        copy_operand_element(source.row, source.col, row, col);
      }
    }

    if (loaded_index_tile != INVALID_TILE) {
      cb_pop_front(cb_indices, 1);
    }
    if (loaded_operand_tile != INVALID_TILE) {
      cb_pop_front(cb_operand, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
