#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr bool BF16_ROWS = GATHER_BF16_ROWS != 0;
constexpr uint32_t RANK = GATHER_RANK;
constexpr uint32_t AXIS = GATHER_AXIS;
static_assert(RANK >= 2, "gather reader expects shapes padded to at least rank 2");
constexpr uint32_t COORD_COUNT = RANK;
constexpr uint32_t OPERAND_SHAPE[COORD_COUNT] = GATHER_OPERAND_SHAPE;
constexpr uint32_t OPERAND_PHYSICAL_SHAPE[COORD_COUNT] =
    GATHER_OPERAND_PHYSICAL_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[COORD_COUNT] = GATHER_OUTPUT_SHAPE;
constexpr uint32_t OPERAND_DIM_STRIDES[COORD_COUNT] =
    GATHER_OPERAND_DIM_STRIDES;
constexpr uint32_t OPERAND_DIM_OFFSETS[COORD_COUNT] =
    GATHER_OPERAND_DIM_OFFSETS;
constexpr bool OPERAND_RESHAPE_VIEW = GATHER_OPERAND_RESHAPE_VIEW != 0;
constexpr bool OPERAND_HAS_DIM_TRANSFORM = [] {
  for (uint32_t dim = 0; dim < COORD_COUNT; ++dim) {
    if (OPERAND_DIM_STRIDES[dim] != 1 || OPERAND_DIM_OFFSETS[dim] != 0) {
      return true;
    }
  }
  return false;
}();
constexpr uint32_t SOURCE_ROWS = GATHER_SOURCE_ROWS;
constexpr uint32_t SOURCE_COLS = GATHER_SOURCE_COLS;
constexpr uint32_t SOURCE_TILE_ROWS = GATHER_SOURCE_TILE_ROWS;
constexpr uint32_t SOURCE_TILES_PER_ROW = GATHER_SOURCE_TILES_PER_ROW;
constexpr uint32_t OPERAND_TILE_ROWS = GATHER_OPERAND_TILE_ROWS;
constexpr uint32_t OPERAND_TILES_PER_ROW = GATHER_OPERAND_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = GATHER_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = GATHER_OUTPUT_TILES_PER_ROW;
using Element = GATHER_ELEMENT_TYPE;

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

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

uint32_t tile_extent(uint32_t logical_dim, uint32_t base, uint32_t tile_dim) {
  if (base >= logical_dim) {
    return 0;
  }
  uint32_t remaining = logical_dim - base;
  return remaining < tile_dim ? remaining : tile_dim;
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
  for (uint32_t index = 0; index < RANK - 2; ++index) {
    uint32_t dim = RANK - 3 - index;
    coords[dim] = batch % shape[dim];
    batch /= shape[dim];
  }
}

uint32_t output_coord(uint32_t dim, const uint32_t base_coords[COORD_COUNT],
                      uint32_t output_row, uint32_t output_col) {
  if (dim == RANK - 1) {
    return output_col;
  }
  if (dim == RANK - 2) {
    return output_row;
  }
  return base_coords[dim];
}

Location tensor_location(const uint32_t shape[COORD_COUNT], uint32_t tile_rows,
                         uint32_t tiles_per_row,
                         const uint32_t coords[COORD_COUNT]) {
  uint32_t batch = 0;
  for (uint32_t dim = 0; dim < RANK - 2; ++dim) {
    batch = batch * shape[dim] + coords[dim];
  }
  uint32_t row = coords[RANK - 2];
  uint32_t col = coords[RANK - 1];
  uint32_t tile_row = row / TILE_R;
  uint32_t tile_col = col / TILE_C;
  uint32_t tile = (batch * tile_rows + tile_row) * tiles_per_row + tile_col;
  return Location{tile, row % TILE_R, col % TILE_C};
}

Location reshape_source_location(uint32_t flat_index) {
  uint32_t col = flat_index % SOURCE_COLS;
  uint32_t row_major = flat_index / SOURCE_COLS;
  uint32_t row = row_major % SOURCE_ROWS;
  uint32_t batch = row_major / SOURCE_ROWS;
  uint32_t tile_row = row / TILE_R;
  uint32_t tile_col = col / TILE_C;
  uint32_t tile = (batch * SOURCE_TILE_ROWS + tile_row) * SOURCE_TILES_PER_ROW + tile_col;
  return Location{tile, row % TILE_R, col % TILE_C};
}

Location operand_location(const uint32_t coords[COORD_COUNT]) {
  uint32_t physical_coords[COORD_COUNT];
  for (uint32_t dim = 0; dim < RANK; ++dim) {
    physical_coords[dim] =
        coords[dim] * OPERAND_DIM_STRIDES[dim] + OPERAND_DIM_OFFSETS[dim];
  }
  if constexpr (OPERAND_RESHAPE_VIEW) {
    uint32_t flat = 0;
    for (uint32_t dim = 0; dim < RANK; ++dim) {
      flat = flat * OPERAND_PHYSICAL_SHAPE[dim] + physical_coords[dim];
    }
    return reshape_source_location(flat);
  } else {
    return tensor_location(
        OPERAND_PHYSICAL_SHAPE, OPERAND_TILE_ROWS, OPERAND_TILES_PER_ROW,
        physical_coords);
  }
}

void ensure_tile(const InterleavedAddrGenFast<true> &input, uint32_t requested_tile,
                 uint32_t cb, uint32_t *loaded_tile) {
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb, 1);
  }
  read_tile_to_cb(input, requested_tile, cb);
  *loaded_tile = requested_tile;
}

int32_t read_gather_index(const InterleavedAddrGenFast<true> &indices,
                          uint32_t output_index, uint32_t *loaded_index_tile) {
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  uint32_t tile = output_index / TILE_R;
  ensure_tile(indices, tile, cb_indices, loaded_index_tile);
  volatile tt_l1_ptr int32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_read_ptr(cb_indices));
  return ptr[tile_element_index(output_index % TILE_R, 0)];
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

void copy_operand_row(uint32_t source_l1_addr, uint32_t output_l1_addr,
                      uint32_t source_row, uint32_t source_col,
                      uint32_t output_row, uint32_t output_col,
                      uint32_t count) {
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(source_l1_addr);
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(output_l1_addr);
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

void copy_bf16_row(uint32_t source_l1_addr, uint32_t output_l1_addr, uint32_t source_row,
                   uint32_t output_row) {
  copy_operand_row(source_l1_addr, output_l1_addr, source_row, 0, output_row, 0, TILE_C);
}

void run_reshape_row_axis_gather(const InterleavedAddrGenFast<true> &operand,
                                 const InterleavedAddrGenFast<true> &indices) {
  uint32_t output_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_tile_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  uint32_t loaded_index_tile = INVALID_TILE;
  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t row_count = tile_extent(OUTPUT_SHAPE[RANK - 2], output_row_base, TILE_R);
    uint32_t col_count = tile_extent(OUTPUT_SHAPE[RANK - 1], output_col_base, TILE_C);

    uint32_t loaded_operand_tile = INVALID_TILE;
    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);
    uint32_t output_l1_addr = get_write_ptr(cb_output);

    for (uint32_t row = 0; row < row_count; ++row) {
      uint32_t output_row = output_row_base + row;
      int32_t gather_index = read_gather_index(indices, output_row, &loaded_index_tile);
      if (gather_index < 0 ||
          static_cast<uint32_t>(gather_index) >= OPERAND_SHAPE[RANK - 2]) {
        continue;
      }

      uint32_t compact_row =
          output_batch * OPERAND_SHAPE[RANK - 2] + static_cast<uint32_t>(gather_index);
      uint32_t source_row = compact_row % SOURCE_ROWS;
      uint32_t source_batch = compact_row / SOURCE_ROWS;
      uint32_t source_tile_row = source_row / TILE_R;
      uint32_t source_tile_col = output_col_base / TILE_C;
      uint32_t source_tile =
          (source_batch * SOURCE_TILE_ROWS + source_tile_row) * SOURCE_TILES_PER_ROW +
          source_tile_col;

      ensure_tile(operand, source_tile, cb_operand, &loaded_operand_tile);
      copy_operand_row(get_read_ptr(cb_operand), output_l1_addr, source_row % TILE_R,
                       output_col_base % TILE_C, row, 0, col_count);
    }

    if (loaded_operand_tile != INVALID_TILE) {
      cb_pop_front(cb_operand, 1);
    }
    cb_push_back(cb_output, 1);
  }
  if (loaded_index_tile != INVALID_TILE) {
    cb_pop_front(cb_indices, 1);
  }
}

// Embedding lookup is a hot LLM path: rank-2 BF16 row gathers can copy a whole
// tile row from one operand tile instead of doing per-element coordinate decode.
void run_bf16_rows_gather(const InterleavedAddrGenFast<true> &operand,
                          const InterleavedAddrGenFast<true> &start_indices) {
  uint32_t output_row_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_row_tile_count = get_arg_val<uint32_t>(3);
  uint32_t logical_output_rows = get_arg_val<uint32_t>(4);
  uint32_t operand_tiles_per_row = get_arg_val<uint32_t>(5);
  uint32_t output_tiles_per_row = get_arg_val<uint32_t>(6);
  uint32_t logical_operand_rows = get_arg_val<uint32_t>(7);

  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  for (uint32_t row_tile = 0; row_tile < output_row_tile_count; ++row_tile) {
    uint32_t output_row_tile = output_row_tile_offset + row_tile;

    read_tile_to_cb(start_indices, output_row_tile, cb_indices);
    uint32_t indices_l1_addr = get_read_ptr(cb_indices);
    volatile tt_l1_ptr int32_t *indices =
        reinterpret_cast<volatile tt_l1_ptr int32_t *>(indices_l1_addr);

    for (uint32_t tile_col = 0; tile_col < output_tiles_per_row; ++tile_col) {
      cb_reserve_back(cb_output, 1);
      zero_tile(cb_output);
      uint32_t output_l1_addr = get_write_ptr(cb_output);

      for (uint32_t row = 0; row < TILE_R; ++row) {
        uint32_t logical_row = output_row_tile * TILE_R + row;
        if (logical_row >= logical_output_rows) {
          continue;
        }

        int32_t token = indices[tile_element_index(row, 0)];
        if (token < 0 || static_cast<uint32_t>(token) >= logical_operand_rows) {
          continue;
        }

        uint32_t token_row = static_cast<uint32_t>(token);
        uint32_t operand_tile_row = token_row / TILE_R;
        uint32_t operand_row = token_row % TILE_R;
        uint32_t operand_tile_id = operand_tile_row * operand_tiles_per_row + tile_col;

        read_tile_to_cb(operand, operand_tile_id, cb_operand);
        copy_bf16_row(get_read_ptr(cb_operand), output_l1_addr, operand_row, row);
        cb_pop_front(cb_operand, 1);
      }

      cb_push_back(cb_output, 1);
    }

    cb_pop_front(cb_indices, 1);
  }
}

void run_axis_gather(const InterleavedAddrGenFast<true> &operand,
                     const InterleavedAddrGenFast<true> &indices) {
  uint32_t output_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_tile_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t row_count = tile_extent(OUTPUT_SHAPE[RANK - 2], output_row_base, TILE_R);
    uint32_t col_count = tile_extent(OUTPUT_SHAPE[RANK - 1], output_col_base, TILE_C);

    uint32_t base_coords[COORD_COUNT];
    decode_batch(output_batch, OUTPUT_SHAPE, base_coords);

    uint32_t loaded_index_tile = INVALID_TILE;
    uint32_t loaded_operand_tile = INVALID_TILE;
    cb_reserve_back(cb_output, 1);
    zero_tile(cb_output);

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

        Location source = operand_location(operand_coords);
        ensure_tile(operand, source.tile, cb_operand, &loaded_operand_tile);
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

void run_prefix_axis_gather(const InterleavedAddrGenFast<true> &operand,
                            const InterleavedAddrGenFast<true> &indices) {
  uint32_t output_tile_offset = get_arg_val<uint32_t>(2);
  uint32_t output_tile_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_output = tt::CBIndex::c_16;

  uint32_t loaded_index_tile = INVALID_TILE;
  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;
    uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t row_count = tile_extent(OUTPUT_SHAPE[RANK - 2], output_row_base, TILE_R);
    uint32_t col_count = tile_extent(OUTPUT_SHAPE[RANK - 1], output_col_base, TILE_C);

    uint32_t base_coords[COORD_COUNT];
    decode_batch(output_batch, OUTPUT_SHAPE, base_coords);
    int32_t gather_index =
        read_gather_index(indices, base_coords[AXIS], &loaded_index_tile);

    cb_reserve_back(cb_output, 1);
    if (gather_index < 0 || static_cast<uint32_t>(gather_index) >= OPERAND_SHAPE[AXIS]) {
      zero_tile(cb_output);
    } else if constexpr (!OPERAND_RESHAPE_VIEW && !OPERAND_HAS_DIM_TRANSFORM) {
      uint32_t operand_coords[COORD_COUNT];
      for (uint32_t dim = 0; dim < RANK; ++dim) {
        operand_coords[dim] = base_coords[dim];
      }
      operand_coords[AXIS] = static_cast<uint32_t>(gather_index);
      operand_coords[RANK - 2] = output_row_base;
      operand_coords[RANK - 1] = output_col_base;
      Location source = tensor_location(
          OPERAND_SHAPE, OPERAND_TILE_ROWS, OPERAND_TILES_PER_ROW, operand_coords);
      noc_async_read_tile(source.tile, operand, get_write_ptr(cb_output));
      noc_async_read_barrier();
    } else {
      zero_tile(cb_output);
      uint32_t output_l1_addr = get_write_ptr(cb_output);
      uint32_t loaded_operand_tile = INVALID_TILE;
      uint32_t operand_coords[COORD_COUNT];
      for (uint32_t dim = 0; dim < RANK; ++dim) {
        operand_coords[dim] = base_coords[dim];
      }
      operand_coords[AXIS] = static_cast<uint32_t>(gather_index);
      for (uint32_t row = 0; row < row_count; ++row) {
        operand_coords[RANK - 2] = output_row_base + row;
        operand_coords[RANK - 1] = output_col_base;
        Location source = operand_location(operand_coords);
        ensure_tile(operand, source.tile, cb_operand, &loaded_operand_tile);
        copy_operand_row(
            get_read_ptr(cb_operand), output_l1_addr, source.row, source.col,
            row, 0, col_count);
      }
      if (loaded_operand_tile != INVALID_TILE) {
        cb_pop_front(cb_operand, 1);
      }
    }
    cb_push_back(cb_output, 1);
  }
  if (loaded_index_tile != INVALID_TILE) {
    cb_pop_front(cb_indices, 1);
  }
}

}  // namespace

void kernel_main() {
  uint32_t operand_addr = get_arg_val<uint32_t>(0);
  uint32_t start_indices_addr = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;

  const InterleavedAddrGenFast<true> operand = {
      .bank_base_address = operand_addr,
      .page_size = get_tile_size(cb_operand),
      .data_format = get_dataformat(cb_operand),
  };
  const InterleavedAddrGenFast<true> start_indices = {
      .bank_base_address = start_indices_addr,
      .page_size = get_tile_size(cb_indices),
      .data_format = get_dataformat(cb_indices),
  };
  if constexpr (BF16_ROWS) {
    run_bf16_rows_gather(operand, start_indices);
  } else if constexpr (RANK >= 3 && AXIS + 2 < RANK) {
    run_prefix_axis_gather(operand, start_indices);
  } else if constexpr (
      OPERAND_RESHAPE_VIEW && !OPERAND_HAS_DIM_TRANSFORM && RANK >= 3 &&
      AXIS == RANK - 2) {
    run_reshape_row_axis_gather(operand, start_indices);
  } else {
    run_axis_gather(operand, start_indices);
  }
}
