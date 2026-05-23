#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

namespace {
constexpr uint32_t TILE_R = 32, TILE_C = 32;
constexpr uint32_t FACE_R = 16, FACE_C = 16;
constexpr uint32_t MAX_RANK = 8;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t VIEW_ARG_COUNT = 9 + 4 * MAX_RANK;
constexpr uint32_t VIEW_CONTIGUOUS = 0;
constexpr uint32_t VIEW_TILED_INDEX_MAP = 4;

struct View {
  uint32_t kind, rank, batch_rank, row_rank, col_rank;
  uint32_t logical_rows, logical_cols, tile_rows, tiles_per_row;
  uint32_t shape[MAX_RANK];
  uint32_t batch_dims[MAX_RANK];
  uint32_t row_dims[MAX_RANK];
  uint32_t col_dims[MAX_RANK];
};

void load_array(uint32_t base, uint32_t *target) {
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    target[i] = A(base + i);
  }
}

View load_view(uint32_t arg_view_kind) {
  const uint32_t arg_view_shape = arg_view_kind + 9;
  const uint32_t arg_view_batch_dims = arg_view_shape + MAX_RANK;
  const uint32_t arg_view_row_dims = arg_view_batch_dims + MAX_RANK;
  const uint32_t arg_view_col_dims = arg_view_row_dims + MAX_RANK;
  View view;
  view.kind = A(arg_view_kind);
  view.rank = A(arg_view_kind + 1);
  view.batch_rank = A(arg_view_kind + 2);
  view.row_rank = A(arg_view_kind + 3);
  view.col_rank = A(arg_view_kind + 4);
  view.logical_rows = A(arg_view_kind + 5);
  view.logical_cols = A(arg_view_kind + 6);
  view.tile_rows = A(arg_view_kind + 7);
  view.tiles_per_row = A(arg_view_kind + 8);
  load_array(arg_view_shape, view.shape);
  load_array(arg_view_batch_dims, view.batch_dims);
  load_array(arg_view_row_dims, view.row_dims);
  load_array(arg_view_col_dims, view.col_dims);
  return view;
}

uint32_t tile_element_index(uint32_t row, uint32_t col) {
  uint32_t face_row = row / FACE_R;
  uint32_t face_col = col / FACE_C;
  uint32_t row_in_face = row % FACE_R;
  uint32_t col_in_face = col % FACE_C;
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C + col_in_face;
}

void zero_tile_at(uint32_t tile_addr, uint32_t tile_bytes) {
  volatile tt_l1_ptr uint32_t *ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_addr);
  uint32_t words = tile_bytes / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

void decompose_into_dims(uint32_t flat, const uint32_t *dims, uint32_t dim_count,
                         const uint32_t *shape, uint32_t *indices) {
  for (int32_t i = static_cast<int32_t>(dim_count) - 1; i >= 0; --i) {
    uint32_t dim = dims[i];
    uint32_t extent = shape[dim];
    indices[dim] = flat % extent;
    flat /= extent;
  }
}

uint32_t tile_id_for_indices(const View &view, const uint32_t *indices,
                             uint32_t *row_in_tile, uint32_t *col_in_tile) {
  uint32_t prefix = 0;
  for (uint32_t dim = 0; dim + 2 < view.rank; ++dim) {
    prefix = prefix * view.shape[dim] + indices[dim];
  }
  uint32_t row = indices[view.rank - 2];
  uint32_t col = indices[view.rank - 1];
  *row_in_tile = row % TILE_R;
  *col_in_tile = col % TILE_C;
  return (prefix * view.tile_rows + row / TILE_R) * view.tiles_per_row + col / TILE_C;
}

void read_source_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                      uint32_t cb_source) {
  cb_reserve_back(cb_source, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb_source));
  noc_async_read_barrier();
  cb_push_back(cb_source, 1);
  cb_wait_front(cb_source, 1);
}

void ensure_source_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                        uint32_t cb_source, uint32_t *loaded_tile) {
  if (*loaded_tile == tile_id) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
  read_source_tile(input, tile_id, cb_source);
  *loaded_tile = tile_id;
}

void copy_element_from_source(uint32_t cb_source, uint32_t dst_addr, uint32_t source_row,
                              uint32_t source_col, uint32_t dst_row, uint32_t dst_col) {
  volatile tt_l1_ptr uint16_t *source =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_read_ptr(cb_source));
  volatile tt_l1_ptr uint16_t *dst = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(dst_addr);
  dst[tile_element_index(dst_row, dst_col)] =
      source[tile_element_index(source_row, source_col)];
}

bool is_grouped_row_view(const View &view) {
  return view.rank == 5 && view.batch_rank == 2 && view.row_rank == 2 &&
         view.col_rank == 1 && view.batch_dims[0] == 0 &&
         view.batch_dims[1] == 2 && view.row_dims[0] == 1 &&
         view.row_dims[1] == 3 && view.col_dims[0] == 4 &&
         view.shape[3] > 0 && view.shape[3] <= TILE_R &&
         TILE_R % view.shape[3] == 0;
}

void fill_grouped_row_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                           uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                           uint32_t dst_addr, uint32_t tile_bytes,
                           uint32_t cb_source) {
  zero_tile_at(dst_addr, tile_bytes);
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    return;
  }

  const uint32_t group_rows = view.shape[3];
  const uint32_t valid_cols =
      (view.logical_cols - col_base) < TILE_C ? (view.logical_cols - col_base) : TILE_C;
  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  indices[4] = col_base;

  for (uint32_t row_group = 0; row_group < TILE_R; row_group += group_rows) {
    const uint32_t logical_row = row_base + row_group;
    if (logical_row >= view.logical_rows) {
      break;
    }
    indices[1] = logical_row / group_rows;
    indices[3] = 0;
    uint32_t source_row = 0;
    uint32_t source_col = 0;
    uint32_t source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
    read_source_tile(input, source_tile, cb_source);
    for (uint32_t inner = 0; inner < group_rows; ++inner) {
      if (logical_row + inner >= view.logical_rows) {
        break;
      }
      for (uint32_t col = 0; col < valid_cols; ++col) {
        copy_element_from_source(cb_source, dst_addr, inner, source_col + col,
                                 row_group + inner, col);
      }
    }
    cb_pop_front(cb_source, 1);
  }
}

bool is_key_by_grouped_col_view(const View &view) {
  return view.rank == 5 && view.batch_rank == 2 && view.row_rank == 1 &&
         view.col_rank == 2 && view.batch_dims[0] == 1 &&
         view.batch_dims[1] == 2 && view.row_dims[0] == 4 &&
         view.col_dims[0] == 0 && view.col_dims[1] == 3 &&
         view.shape[0] > 0 && view.shape[0] <= TILE_C &&
         view.shape[3] > 0 && view.shape[3] <= TILE_R;
}

void fill_key_by_grouped_col_tile(const InterleavedAddrGenFast<true> &input,
                                  const View &view, uint32_t batch,
                                  uint32_t row_tile, uint32_t col_tile,
                                  uint32_t dst_addr, uint32_t tile_bytes,
                                  uint32_t cb_source) {
  zero_tile_at(dst_addr, tile_bytes);
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    return;
  }

  const uint32_t query_count = view.shape[3];
  const uint32_t valid_rows =
      (view.logical_rows - row_base) < TILE_R ? (view.logical_rows - row_base) : TILE_R;
  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  indices[3] = 0;
  indices[4] = row_base;

  uint32_t first_repeat = col_base / query_count;
  uint32_t last_logical_col = col_base + TILE_C - 1;
  if (last_logical_col >= view.logical_cols) {
    last_logical_col = view.logical_cols - 1;
  }
  uint32_t last_repeat = last_logical_col / query_count;
  if (last_repeat >= view.shape[0]) {
    last_repeat = view.shape[0] - 1;
  }

  for (uint32_t repeat = first_repeat; repeat <= last_repeat; ++repeat) {
    indices[0] = repeat;
    uint32_t source_row = 0;
    uint32_t source_col = 0;
    uint32_t source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
    read_source_tile(input, source_tile, cb_source);
    for (uint32_t col = 0; col < TILE_C; ++col) {
      uint32_t logical_col = col_base + col;
      if (logical_col >= view.logical_cols ||
          logical_col / query_count != repeat) {
        continue;
      }
      uint32_t query = logical_col - repeat * query_count;
      for (uint32_t row = 0; row < valid_rows; ++row) {
        copy_element_from_source(cb_source, dst_addr, query, source_col + row,
                                 row, col);
      }
    }
    cb_pop_front(cb_source, 1);
  }
}

void fill_generic_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                       uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                       uint32_t dst_addr, uint32_t tile_bytes, uint32_t cb_source) {
  if (is_grouped_row_view(view)) {
    fill_grouped_row_tile(input, view, batch, row_tile, col_tile, dst_addr,
                          tile_bytes, cb_source);
    return;
  }
  if (is_key_by_grouped_col_view(view)) {
    fill_key_by_grouped_col_tile(input, view, batch, row_tile, col_tile,
                                 dst_addr, tile_bytes, cb_source);
    return;
  }
  zero_tile_at(dst_addr, tile_bytes);
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    return;
  }

  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  uint32_t batch_indices[MAX_RANK] = {};
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    batch_indices[i] = indices[i];
  }
  uint32_t col_indices[TILE_C][MAX_RANK] = {};
  bool valid_cols[TILE_C] = {};
  for (uint32_t col = 0; col < TILE_C; ++col) {
    uint32_t logical_col = col_base + col;
    if (logical_col >= view.logical_cols) {
      continue;
    }
    valid_cols[col] = true;
    decompose_into_dims(logical_col, view.col_dims, view.col_rank, view.shape, col_indices[col]);
  }

  uint32_t loaded_tile = INVALID_TILE;
  for (uint32_t row = 0; row < TILE_R; ++row) {
    uint32_t logical_row = row_base + row;
    if (logical_row >= view.logical_rows) {
      continue;
    }
    for (uint32_t i = 0; i < MAX_RANK; ++i) {
      indices[i] = batch_indices[i];
    }
    decompose_into_dims(logical_row, view.row_dims, view.row_rank, view.shape, indices);
    for (uint32_t col = 0; col < TILE_C; ++col) {
      if (!valid_cols[col]) {
        continue;
      }
      for (uint32_t i = 0; i < view.col_rank; ++i) {
        uint32_t dim = view.col_dims[i];
        indices[dim] = col_indices[col][dim];
      }
      uint32_t source_row = 0;
      uint32_t source_col = 0;
      uint32_t source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
      ensure_source_tile(input, source_tile, cb_source, &loaded_tile);
      copy_element_from_source(cb_source, dst_addr, source_row, source_col, row, col);
    }
  }
  if (loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
}

// Specialized tiled index map for views where the matmul row dimension is the
// source tensor's innermost physical dimension and the matmul column dimension
// is a prefix dimension. Example: [batch, token, head, dim] is viewed as
// [batch, head, dim, token], so one output column/token maps to one source tile
// and output rows/dim walk contiguous columns inside that source tile.
struct TiledIndexMap {
  uint32_t source_row_dim;
  uint32_t source_col_dim;
};

void fill_tiled_index_map_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                               uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                               uint32_t dst_addr, uint32_t tile_bytes,
                               uint32_t cb_source) {
  zero_tile_at(dst_addr, tile_bytes);
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    return;
  }

  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  TiledIndexMap map = {view.row_dims[0], view.col_dims[0]};
  indices[map.source_row_dim] = row_base;

  for (uint32_t col = 0; col < TILE_C; ++col) {
    uint32_t logical_col = col_base + col;
    if (logical_col >= view.logical_cols) {
      continue;
    }
    indices[map.source_col_dim] = logical_col;
    uint32_t source_row = 0;
    uint32_t source_col = 0;
    uint32_t source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
    read_source_tile(input, source_tile, cb_source);
    for (uint32_t row = 0; row < TILE_R; ++row) {
      if (row_base + row >= view.logical_rows) {
        continue;
      }
      copy_element_from_source(cb_source, dst_addr, source_row, source_col + row, row, col);
    }
    cb_pop_front(cb_source, 1);
  }
}
}  // namespace
