#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

namespace {
constexpr uint32_t TILE_R = 32, TILE_C = 32;
constexpr uint32_t FACE_R = 16, FACE_C = 16;
constexpr uint32_t MAX_RANK = 8;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t VIEW_ARG_COUNT = 11 + 5 * MAX_RANK;
constexpr uint32_t VIEW_CONTIGUOUS = 0;
constexpr uint32_t VIEW_TILED_INDEX_MAP = 4;
constexpr uint32_t VIEW_TILE_TRANSPOSE = 5;
constexpr uint32_t GROUPED_DIM_NONE = 0xffffffffu;

struct View {
  uint32_t kind, rank, batch_rank, row_rank, col_rank;
  uint32_t logical_rows, logical_cols, tile_rows, tiles_per_row;
  uint32_t grouped_dim, group_size;
  uint32_t shape[MAX_RANK];
  uint32_t physical_shape[MAX_RANK];
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
  const uint32_t arg_view_shape = arg_view_kind + 11;
  const uint32_t arg_view_physical_shape = arg_view_shape + MAX_RANK;
  const uint32_t arg_view_batch_dims = arg_view_physical_shape + MAX_RANK;
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
  view.grouped_dim = A(arg_view_kind + 9);
  view.group_size = A(arg_view_kind + 10);
  load_array(arg_view_shape, view.shape);
  load_array(arg_view_physical_shape, view.physical_shape);
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

uint32_t min_u32(uint32_t lhs, uint32_t rhs) {
  return lhs < rhs ? lhs : rhs;
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
  auto physical_index = [&](uint32_t dim) {
    uint32_t index = indices[dim];
    if (dim == view.grouped_dim) {
      index /= view.group_size;
    }
    return index;
  };
  if (view.rank == 1) {
    uint32_t col = physical_index(0);
    *row_in_tile = 0;
    *col_in_tile = col % TILE_C;
    return col / TILE_C;
  }
  uint32_t prefix = 0;
  for (uint32_t dim = 0; dim + 2 < view.rank; ++dim) {
    prefix = prefix * view.physical_shape[dim] + physical_index(dim);
  }
  uint32_t row = physical_index(view.rank - 2);
  uint32_t col = physical_index(view.rank - 1);
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

template <uint32_t DatumBytes>
struct ElementForDatumBytes;

template <>
struct ElementForDatumBytes<sizeof(uint16_t)> {
  using Type = uint16_t;
};

template <>
struct ElementForDatumBytes<sizeof(uint32_t)> {
  using Type = uint32_t;
};

template <uint32_t DatumBytes>
using ElementForDatumBytesT = typename ElementForDatumBytes<DatumBytes>::Type;

template <uint32_t DatumBytes>
void copy_element_from_source(uint32_t cb_source, uint32_t dst_addr, uint32_t source_row,
                              uint32_t source_col, uint32_t dst_row, uint32_t dst_col) {
  using Element = ElementForDatumBytesT<DatumBytes>;
  const uint32_t dst_index = tile_element_index(dst_row, dst_col);
  const uint32_t source_index = tile_element_index(source_row, source_col);
  volatile tt_l1_ptr Element *source =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_source));
  volatile tt_l1_ptr Element *dst =
      reinterpret_cast<volatile tt_l1_ptr Element *>(dst_addr);
  dst[dst_index] = source[source_index];
}

template <uint32_t DatumBytes>
void copy_source_row_to_dst_row(uint32_t cb_source, uint32_t dst_addr,
                                uint32_t source_row, uint32_t source_col,
                                uint32_t dst_row, uint32_t dst_col, uint32_t count) {
  if (count == TILE_C && source_col == 0 && dst_col == 0) {
    using Element = ElementForDatumBytesT<DatumBytes>;
    const uint32_t source_face0 = tile_element_index(source_row, 0);
    const uint32_t source_face1 = tile_element_index(source_row, FACE_C);
    const uint32_t dst_face0 = tile_element_index(dst_row, 0);
    const uint32_t dst_face1 = tile_element_index(dst_row, FACE_C);
    volatile tt_l1_ptr Element *source =
        reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_source));
    volatile tt_l1_ptr Element *dst =
        reinterpret_cast<volatile tt_l1_ptr Element *>(dst_addr);
    for (uint32_t col = 0; col < FACE_C; ++col) {
      dst[dst_face0 + col] = source[source_face0 + col];
      dst[dst_face1 + col] = source[source_face1 + col];
    }
    return;
  }
  for (uint32_t col = 0; col < count; ++col) {
    copy_element_from_source<DatumBytes>(
        cb_source, dst_addr, source_row, source_col + col, dst_row, dst_col + col);
  }
}

template <uint32_t DatumBytes>
void copy_source_row_to_dst_col(uint32_t cb_source, uint32_t dst_addr,
                                uint32_t source_row, uint32_t source_col,
                                uint32_t dst_row, uint32_t dst_col, uint32_t count) {
  if (count == TILE_R && source_col == 0 && dst_row == 0) {
    using Element = ElementForDatumBytesT<DatumBytes>;
    const uint32_t source_face0 = tile_element_index(source_row, 0);
    const uint32_t source_face1 = tile_element_index(source_row, FACE_C);
    const uint32_t dst_face0 = tile_element_index(0, dst_col);
    const uint32_t dst_face1 = tile_element_index(FACE_R, dst_col);
    volatile tt_l1_ptr Element *source =
        reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_source));
    volatile tt_l1_ptr Element *dst =
        reinterpret_cast<volatile tt_l1_ptr Element *>(dst_addr);
    for (uint32_t row = 0; row < FACE_R; ++row) {
      dst[dst_face0 + row * FACE_C] = source[source_face0 + row];
      dst[dst_face1 + row * FACE_C] = source[source_face1 + row];
    }
    return;
  }
  for (uint32_t row = 0; row < count; ++row) {
    copy_element_from_source<DatumBytes>(
        cb_source, dst_addr, source_row, source_col + row, dst_row + row, dst_col);
  }
}

bool contains_dim(const uint32_t *dims, uint32_t dim_count, uint32_t dim) {
  for (uint32_t i = 0; i < dim_count; ++i) {
    if (dims[i] == dim) {
      return true;
    }
  }
  return false;
}

bool is_prefix_row_inner_col_view(const View &view) {
  // GQA KV-cache score matmuls read logical rows from a prefix dimension while
  // columns stay in the innermost physical dimension.
  if (view.rank < 3 || view.row_rank != 1 || view.col_rank != 1 ||
      view.col_dims[0] != view.rank - 1 || view.row_dims[0] + 2 >= view.rank) {
    return false;
  }
  if (view.grouped_dim == GROUPED_DIM_NONE || view.group_size <= 1) {
    return false;
  }
  for (uint32_t dim = 0; dim + 2 < view.rank; ++dim) {
    if (dim != view.row_dims[0] &&
        !contains_dim(view.batch_dims, view.batch_rank, dim)) {
      return false;
    }
  }
  return true;
}

template <uint32_t DatumBytes>
void fill_prefix_row_inner_col_tile_impl(const InterleavedAddrGenFast<true> &input,
                                         const View &view, uint32_t batch,
                                         uint32_t row_tile, uint32_t col_tile,
                                         uint32_t dst_addr, uint32_t tile_bytes,
                                         uint32_t cb_source) {
  const uint32_t row_base = row_tile * TILE_R;
  const uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    zero_tile_at(dst_addr, tile_bytes);
    return;
  }

  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  indices[view.col_dims[0]] = col_base;
  const uint32_t valid_rows = min_u32(TILE_R, view.logical_rows - row_base);
  const uint32_t valid_cols = min_u32(TILE_C, view.logical_cols - col_base);
  if (valid_rows != TILE_R || valid_cols != TILE_C) {
    zero_tile_at(dst_addr, tile_bytes);
  }

  const uint32_t row_dim = view.row_dims[0];
  const bool can_stride_source_tiles = row_dim != view.grouped_dim;
  uint32_t source_tile_base = 0;
  uint32_t source_tile_stride = 0;
  uint32_t strided_source_row = 0;
  uint32_t strided_source_col = 0;
  if (can_stride_source_tiles) {
    indices[row_dim] = row_base;
    source_tile_base = tile_id_for_indices(
        view, indices, &strided_source_row, &strided_source_col);
    source_tile_stride = view.tile_rows * view.tiles_per_row;
    for (uint32_t dim = row_dim + 1; dim + 2 < view.rank; ++dim) {
      source_tile_stride *= view.physical_shape[dim];
    }
  }

  for (uint32_t row = 0; row < valid_rows; ++row) {
    uint32_t source_row = strided_source_row;
    uint32_t source_col = strided_source_col;
    uint32_t source_tile = source_tile_base + row * source_tile_stride;
    if (!can_stride_source_tiles) {
      indices[row_dim] = row_base + row;
      source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
    }
    read_source_tile(input, source_tile, cb_source);
    copy_source_row_to_dst_row<DatumBytes>(
        cb_source, dst_addr, source_row, source_col, row, 0, valid_cols);
    cb_pop_front(cb_source, 1);
  }
}

template <uint32_t DatumBytes>
void fill_generic_tile_impl(const InterleavedAddrGenFast<true> &input, const View &view,
                            uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                            uint32_t dst_addr, uint32_t tile_bytes, uint32_t cb_source) {
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    zero_tile_at(dst_addr, tile_bytes);
    return;
  }
  if (row_base + TILE_R > view.logical_rows || col_base + TILE_C > view.logical_cols) {
    zero_tile_at(dst_addr, tile_bytes);
  }

  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);

  uint32_t loaded_tile = INVALID_TILE;
  for (uint32_t row = 0; row < TILE_R; ++row) {
    uint32_t logical_row = row_base + row;
    if (logical_row >= view.logical_rows) {
      continue;
    }
    for (uint32_t col = 0; col < TILE_C; ++col) {
      uint32_t logical_col = col_base + col;
      if (logical_col >= view.logical_cols) {
        continue;
      }
      decompose_into_dims(logical_row, view.row_dims, view.row_rank, view.shape, indices);
      decompose_into_dims(logical_col, view.col_dims, view.col_rank, view.shape, indices);
      uint32_t source_row = 0;
      uint32_t source_col = 0;
      uint32_t source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
      ensure_source_tile(input, source_tile, cb_source, &loaded_tile);
      copy_element_from_source<DatumBytes>(cb_source, dst_addr, source_row, source_col, row,
                                           col);
    }
  }
  if (loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
}

void fill_generic_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                       uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                       uint32_t dst_addr, uint32_t tile_bytes, uint32_t cb_source) {
  const bool prefix_row_inner_col = is_prefix_row_inner_col_view(view);
  if (tile_bytes == sizeof(uint32_t) * TILE_R * TILE_C) {
    if (prefix_row_inner_col) {
      fill_prefix_row_inner_col_tile_impl<sizeof(uint32_t)>(
          input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source);
    } else {
      fill_generic_tile_impl<sizeof(uint32_t)>(
          input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source);
    }
  } else {
    if (prefix_row_inner_col) {
      fill_prefix_row_inner_col_tile_impl<sizeof(uint16_t)>(
          input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source);
    } else {
      fill_generic_tile_impl<sizeof(uint16_t)>(
          input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source);
    }
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

template <uint32_t DatumBytes>
void fill_tiled_index_map_tile_impl(const InterleavedAddrGenFast<true> &input,
                                    const View &view, uint32_t batch, uint32_t row_tile,
                                    uint32_t col_tile, uint32_t dst_addr,
                                    uint32_t tile_bytes, uint32_t cb_source) {
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    zero_tile_at(dst_addr, tile_bytes);
    return;
  }
  const uint32_t valid_rows = min_u32(TILE_R, view.logical_rows - row_base);
  const uint32_t valid_cols = min_u32(TILE_C, view.logical_cols - col_base);
  if (valid_rows != TILE_R || valid_cols != TILE_C) {
    zero_tile_at(dst_addr, tile_bytes);
  }

  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  if (view.grouped_dim != GROUPED_DIM_NONE) {
    indices[view.grouped_dim] /= view.group_size;
  }

  TiledIndexMap map = {view.row_dims[0], view.col_dims[0]};
  indices[map.source_row_dim] = row_base;
  const uint32_t source_row_base = indices[view.rank - 2];
  const uint32_t source_col_base = row_base;
  const uint32_t source_row_in_tile = source_row_base % TILE_R;
  const uint32_t source_col_in_tile = source_col_base % TILE_C;
  const uint32_t source_tile_row = source_row_base / TILE_R;
  const uint32_t source_tile_col = source_col_base / TILE_C;

  for (uint32_t col = 0; col < valid_cols; ++col) {
    uint32_t logical_col = col_base + col;
    uint32_t prefix = 0;
    for (uint32_t dim = 0; dim + 2 < view.rank; ++dim) {
      uint32_t index = dim == map.source_col_dim ? logical_col : indices[dim];
      prefix = prefix * view.physical_shape[dim] + index;
    }
    uint32_t source_tile =
        (prefix * view.tile_rows + source_tile_row) * view.tiles_per_row + source_tile_col;
    read_source_tile(input, source_tile, cb_source);
    copy_source_row_to_dst_col<DatumBytes>(
        cb_source, dst_addr, source_row_in_tile, source_col_in_tile, 0, col, valid_rows);
    cb_pop_front(cb_source, 1);
  }
}

void fill_tiled_index_map_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                               uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                               uint32_t dst_addr, uint32_t tile_bytes,
                               uint32_t cb_source) {
  if (tile_bytes == sizeof(uint32_t) * TILE_R * TILE_C) {
    fill_tiled_index_map_tile_impl<sizeof(uint32_t)>(
        input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source);
  } else {
    fill_tiled_index_map_tile_impl<sizeof(uint16_t)>(
        input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source);
  }
}

template <uint32_t DatumBytes>
void fill_tile_transpose_tile_impl(const InterleavedAddrGenFast<true> &input,
                                   const View &view, uint32_t batch, uint32_t row_tile,
                                   uint32_t col_tile, uint32_t dst_addr,
                                   uint32_t tile_bytes, uint32_t cb_source) {
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    zero_tile_at(dst_addr, tile_bytes);
    return;
  }

  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  indices[view.col_dims[0]] = col_base;
  indices[view.row_dims[0]] = row_base;

  uint32_t source_row_base = 0;
  uint32_t source_col_base = 0;
  uint32_t source_tile =
      tile_id_for_indices(view, indices, &source_row_base, &source_col_base);
  if (row_base + TILE_R <= view.logical_rows && col_base + TILE_C <= view.logical_cols) {
    noc_async_read_tile(source_tile, input, dst_addr);
    noc_async_read_barrier();
    return;
  }

  zero_tile_at(dst_addr, tile_bytes);
  read_source_tile(input, source_tile, cb_source);

  for (uint32_t row = 0; row < TILE_R && col_base + row < view.logical_cols; ++row) {
    for (uint32_t col = 0; col < TILE_C && row_base + col < view.logical_rows; ++col) {
      copy_element_from_source<DatumBytes>(
          cb_source, dst_addr, source_row_base + row, source_col_base + col, row, col);
    }
  }
  cb_pop_front(cb_source, 1);
}

void fill_tile_transpose_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                              uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                              uint32_t dst_addr, uint32_t tile_bytes,
                              uint32_t cb_source) {
  if (tile_bytes == sizeof(uint32_t) * TILE_R * TILE_C) {
    fill_tile_transpose_tile_impl<sizeof(uint32_t)>(
        input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source);
  } else {
    fill_tile_transpose_tile_impl<sizeof(uint16_t)>(
        input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source);
  }
}
}  // namespace
