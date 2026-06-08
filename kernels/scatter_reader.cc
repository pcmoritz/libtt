#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t RANK = SCATTER_RANK;
constexpr uint32_t COORD_COUNT = RANK == 0 ? 1 : RANK;
constexpr uint32_t OPERAND_SHAPE[COORD_COUNT] = SCATTER_OPERAND_SHAPE;
constexpr uint32_t UPDATE_SHAPE[COORD_COUNT] = SCATTER_UPDATE_SHAPE;
constexpr uint32_t SCATTER_DIM = SCATTER_DIM_ARG;
constexpr uint32_t UPDATE_COUNT = SCATTER_UPDATE_COUNT;
constexpr bool OPERAND_RESHAPE_VIEW = SCATTER_OPERAND_RESHAPE_VIEW != 0;
constexpr uint32_t SOURCE_ROWS = SCATTER_SOURCE_ROWS;
constexpr uint32_t SOURCE_COLS = SCATTER_SOURCE_COLS;
constexpr uint32_t SOURCE_TILE_ROWS = SCATTER_SOURCE_TILE_ROWS;
constexpr uint32_t SOURCE_TILES_PER_ROW = SCATTER_SOURCE_TILES_PER_ROW;
constexpr bool UPDATE_RESHAPE_VIEW = SCATTER_UPDATE_RESHAPE_VIEW != 0;
constexpr uint32_t UPDATE_SOURCE_ROWS = SCATTER_UPDATE_SOURCE_ROWS;
constexpr uint32_t UPDATE_SOURCE_COLS = SCATTER_UPDATE_SOURCE_COLS;
constexpr uint32_t UPDATE_SOURCE_TILE_ROWS = SCATTER_UPDATE_SOURCE_TILE_ROWS;
constexpr uint32_t UPDATE_SOURCE_TILES_PER_ROW = SCATTER_UPDATE_SOURCE_TILES_PER_ROW;
constexpr uint32_t OPERAND_TILE_ROWS = SCATTER_OPERAND_TILE_ROWS;
constexpr uint32_t OPERAND_TILES_PER_ROW = SCATTER_OPERAND_TILES_PER_ROW;
constexpr uint32_t UPDATE_TILE_ROWS = SCATTER_UPDATE_TILE_ROWS;
constexpr uint32_t UPDATE_TILES_PER_ROW = SCATTER_UPDATE_TILES_PER_ROW;
using Element = SCATTER_ELEMENT_TYPE;

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

void read_tile_to_cb(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                     uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
  cb_wait_front(cb, 1);
}

void read_tile_to_output(const InterleavedAddrGenFast<true> &input,
                         uint32_t tile_id, uint32_t cb_output) {
  noc_async_read_tile(tile_id, input, get_write_ptr(cb_output));
  noc_async_read_barrier();
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

Location reshape_source_location(uint32_t flat_index, uint32_t rows, uint32_t cols,
                                 uint32_t tile_rows, uint32_t tiles_per_row) {
  uint32_t col = flat_index % cols;
  uint32_t row_major = flat_index / cols;
  uint32_t row = row_major % rows;
  uint32_t batch = row_major / rows;
  uint32_t tile_row = row / TILE_R;
  uint32_t tile_col = col / TILE_C;
  uint32_t tile = (batch * tile_rows + tile_row) * tiles_per_row + tile_col;
  return Location{tile, row % TILE_R, col % TILE_C};
}

Location operand_location(const uint32_t coords[COORD_COUNT]) {
  if constexpr (OPERAND_RESHAPE_VIEW) {
    uint32_t flat = 0;
    for (uint32_t dim = 0; dim < RANK; ++dim) {
      flat = flat * OPERAND_SHAPE[dim] + coords[dim];
    }
    return reshape_source_location(flat, SOURCE_ROWS, SOURCE_COLS, SOURCE_TILE_ROWS,
                                   SOURCE_TILES_PER_ROW);
  } else {
    return tensor_location(OPERAND_SHAPE, OPERAND_TILE_ROWS, OPERAND_TILES_PER_ROW,
                           coords);
  }
}

Location update_location(const uint32_t coords[COORD_COUNT]) {
  if constexpr (UPDATE_RESHAPE_VIEW) {
    uint32_t flat = 0;
    for (uint32_t dim = 0; dim < RANK; ++dim) {
      flat = flat * UPDATE_SHAPE[dim] + coords[dim];
    }
    return reshape_source_location(flat, UPDATE_SOURCE_ROWS, UPDATE_SOURCE_COLS,
                                   UPDATE_SOURCE_TILE_ROWS,
                                   UPDATE_SOURCE_TILES_PER_ROW);
  } else {
    return tensor_location(UPDATE_SHAPE, UPDATE_TILE_ROWS, UPDATE_TILES_PER_ROW,
                           coords);
  }
}

uint32_t direct_prefix_update_tile(uint32_t update_index,
                                   const uint32_t base_coords[COORD_COUNT],
                                   uint32_t output_tile_row,
                                   uint32_t output_tile_col) {
  uint32_t update_batch = 0;
  if constexpr (RANK >= 3) {
    for (uint32_t dim = 0; dim < RANK - 2; ++dim) {
      uint32_t coord = dim == SCATTER_DIM ? update_index : base_coords[dim];
      update_batch = update_batch * UPDATE_SHAPE[dim] + coord;
    }
  }
  return (update_batch * UPDATE_TILE_ROWS + output_tile_row) * UPDATE_TILES_PER_ROW +
         output_tile_col;
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

int32_t read_scatter_index(const InterleavedAddrGenFast<true> &indices,
                           uint32_t update_index, uint32_t *loaded_index_tile) {
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  uint32_t tile = update_index / TILE_R;
  ensure_index_tile(indices, tile, loaded_index_tile);
  volatile tt_l1_ptr int32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_read_ptr(cb_indices));
  return ptr[tile_element_index(update_index % TILE_R, 0)];
}

void ensure_update_tile(const InterleavedAddrGenFast<true> &updates,
                        uint32_t requested_tile, uint32_t *loaded_tile) {
  constexpr uint32_t cb_updates = tt::CBIndex::c_2;
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_updates, 1);
  }
  read_tile_to_cb(updates, requested_tile, cb_updates);
  *loaded_tile = requested_tile;
}

void copy_element(uint32_t source_cb, uint32_t source_row, uint32_t source_col,
                  uint32_t output_row, uint32_t output_col) {
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(source_cb));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  output[tile_element_index(output_row, output_col)] =
      source[tile_element_index(source_row, source_col)];
}

void copy_row_partial(uint32_t source_cb, uint32_t source_row, uint32_t source_col,
                      uint32_t output_row, uint32_t output_col, uint32_t count) {
  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(source_cb));
  volatile tt_l1_ptr Element *output =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_output));
  for (uint32_t col = 0; col < count; ++col) {
    output[tile_element_index(output_row, output_col + col)] =
        source[tile_element_index(source_row, source_col + col)];
  }
}

void scatter_prefix(const InterleavedAddrGenFast<true> &indices,
                    const InterleavedAddrGenFast<true> &updates,
                    const uint32_t base_coords[COORD_COUNT],
                    uint32_t output_tile_row, uint32_t output_tile_col,
                    uint32_t output_row_base, uint32_t output_col_base,
                    uint32_t row_count, uint32_t col_count,
                    uint32_t *loaded_index_tile,
                    uint32_t *loaded_update_tile) {
  if constexpr (RANK >= 3 && SCATTER_DIM < RANK - 2) {
    constexpr uint32_t cb_updates = tt::CBIndex::c_2;
    for (uint32_t update_index = 0; update_index < UPDATE_COUNT; ++update_index) {
      int32_t target = read_scatter_index(indices, update_index, loaded_index_tile);
      if (target < 0 || static_cast<uint32_t>(target) >= OPERAND_SHAPE[SCATTER_DIM] ||
          base_coords[SCATTER_DIM] != static_cast<uint32_t>(target)) {
        continue;
      }
      if constexpr (UPDATE_RESHAPE_VIEW) {
        for (uint32_t row = 0; row < row_count; ++row) {
          uint32_t output_row = output_row_base + row;
          uint32_t update_coords[COORD_COUNT];
          for (uint32_t dim = 0; dim < RANK; ++dim) {
            update_coords[dim] =
                dim == SCATTER_DIM
                    ? update_index
                    : output_coord(dim, base_coords, output_row, output_col_base);
          }
          Location source = update_location(update_coords);
          ensure_update_tile(updates, source.tile, loaded_update_tile);
          if (source.col + col_count <= TILE_C && source.col + col_count <= UPDATE_SOURCE_COLS) {
            copy_row_partial(cb_updates, source.row, source.col, row, 0, col_count);
          } else {
            for (uint32_t col = 0; col < col_count; ++col) {
              update_coords[RANK - 1] = output_col_base + col;
              source = update_location(update_coords);
              ensure_update_tile(updates, source.tile, loaded_update_tile);
              copy_element(cb_updates, source.row, source.col, row, col);
            }
          }
        }
      } else {
        read_tile_to_output(
            updates,
            direct_prefix_update_tile(update_index, base_coords, output_tile_row,
                                      output_tile_col),
            tt::CBIndex::c_16);
      }
    }
  }
}

void scatter_generic(const InterleavedAddrGenFast<true> &indices,
                     const InterleavedAddrGenFast<true> &updates,
                     const uint32_t base_coords[COORD_COUNT],
                     uint32_t output_row_base, uint32_t output_col_base,
                     uint32_t row_count, uint32_t col_count,
                     uint32_t *loaded_index_tile,
                     uint32_t *loaded_update_tile) {
  constexpr uint32_t cb_updates = tt::CBIndex::c_2;
  for (uint32_t update_index = 0; update_index < UPDATE_COUNT; ++update_index) {
    int32_t target = read_scatter_index(indices, update_index, loaded_index_tile);
    if (target < 0 || static_cast<uint32_t>(target) >= OPERAND_SHAPE[SCATTER_DIM]) {
      continue;
    }

    for (uint32_t row = 0; row < row_count; ++row) {
      uint32_t output_row = output_row_base + row;
      for (uint32_t col = 0; col < col_count; ++col) {
        uint32_t output_col = output_col_base + col;
        if (output_coord(SCATTER_DIM, base_coords, output_row, output_col) !=
            static_cast<uint32_t>(target)) {
          continue;
        }

        uint32_t update_coords[COORD_COUNT];
        for (uint32_t dim = 0; dim < RANK; ++dim) {
          update_coords[dim] = dim == SCATTER_DIM
                                   ? update_index
                                   : output_coord(dim, base_coords, output_row, output_col);
        }

        Location source = update_location(update_coords);
        ensure_update_tile(updates, source.tile, loaded_update_tile);
        copy_element(cb_updates, source.row, source.col, row, col);
      }
    }
  }
}

}  // namespace

void kernel_main() {
  uint32_t operand_addr = get_arg_val<uint32_t>(0);
  uint32_t start_indices_addr = get_arg_val<uint32_t>(1);
  uint32_t updates_addr = get_arg_val<uint32_t>(2);
  uint32_t output_tile_offset = get_arg_val<uint32_t>(3);
  uint32_t output_tile_count = get_arg_val<uint32_t>(4);

  constexpr uint32_t cb_operand = tt::CBIndex::c_0;
  constexpr uint32_t cb_indices = tt::CBIndex::c_1;
  constexpr uint32_t cb_updates = tt::CBIndex::c_2;
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
  const InterleavedAddrGenFast<true> updates = {
      .bank_base_address = updates_addr,
      .page_size = get_tile_size(cb_updates),
      .data_format = get_dataformat(cb_updates),
  };

  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    uint32_t output_tile_id = output_tile_offset + tile;

    uint32_t output_matrix_tiles = OPERAND_TILE_ROWS * OPERAND_TILES_PER_ROW;
    uint32_t output_batch = output_tile_id / output_matrix_tiles;
    uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    uint32_t output_tile_row = output_matrix_tile / OPERAND_TILES_PER_ROW;
    uint32_t output_tile_col = output_matrix_tile % OPERAND_TILES_PER_ROW;
    uint32_t output_row_base = output_tile_row * TILE_R;
    uint32_t output_col_base = output_tile_col * TILE_C;
    uint32_t row_count = 1;
    uint32_t col_count = 1;

    if constexpr (RANK == 1) {
      col_count = tile_extent(OPERAND_SHAPE[0], output_col_base, TILE_C);
    } else {
      row_count = tile_extent(OPERAND_SHAPE[RANK - 2], output_row_base, TILE_R);
      col_count = tile_extent(OPERAND_SHAPE[RANK - 1], output_col_base, TILE_C);
    }

    uint32_t base_coords[COORD_COUNT];
    decode_batch(output_batch, OPERAND_SHAPE, base_coords);

    cb_reserve_back(cb_output, 1);
    if constexpr (OPERAND_RESHAPE_VIEW) {
      zero_tile(cb_output);
      uint32_t loaded_operand_tile = INVALID_TILE;
      for (uint32_t row = 0; row < row_count; ++row) {
        uint32_t output_row = output_row_base + row;
        for (uint32_t col = 0; col < col_count; ++col) {
          uint32_t output_col = output_col_base + col;
          uint32_t operand_coords[COORD_COUNT];
          for (uint32_t dim = 0; dim < RANK; ++dim) {
            operand_coords[dim] = output_coord(dim, base_coords, output_row, output_col);
          }
          Location source = operand_location(operand_coords);
          ensure_operand_tile(operand, source.tile, &loaded_operand_tile);
          copy_element(cb_operand, source.row, source.col, row, col);
        }
      }
      if (loaded_operand_tile != INVALID_TILE) {
        cb_pop_front(cb_operand, 1);
      }
    } else {
      read_tile_to_output(operand, output_tile_id, cb_output);
    }

    uint32_t loaded_index_tile = INVALID_TILE;
    uint32_t loaded_update_tile = INVALID_TILE;
    if constexpr (RANK >= 3 && SCATTER_DIM < RANK - 2) {
      scatter_prefix(indices, updates, base_coords, output_tile_row, output_tile_col,
                     output_row_base, output_col_base, row_count, col_count,
                     &loaded_index_tile, &loaded_update_tile);
    } else {
      scatter_generic(indices, updates, base_coords, output_row_base, output_col_base,
                      row_count, col_count, &loaded_index_tile, &loaded_update_tile);
    }

    if (loaded_index_tile != INVALID_TILE) {
      cb_pop_front(cb_indices, 1);
    }
    if (loaded_update_tile != INVALID_TILE) {
      cb_pop_front(cb_updates, 1);
    }
    cb_push_back(cb_output, 1);
  }
}
