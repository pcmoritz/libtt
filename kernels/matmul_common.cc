#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)
#define SEM(n) reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_semaphore(A(n)))

namespace {
constexpr uint32_t TILE_R = 32, TILE_C = 32;
constexpr uint32_t FACE_R = 16, FACE_C = 16;
constexpr uint32_t MAX_RANK = 8;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t VIEW_ARG_COUNT = 17 + 8 * MAX_RANK;
constexpr uint32_t VIEW_CONTIGUOUS = 0;
constexpr uint32_t VIEW_TILED_INDEX_MAP = 4;
constexpr uint32_t VIEW_TILE_TRANSPOSE = 5;
constexpr uint32_t GROUPED_DIM_NONE = 0xffffffffu;

struct View {
  uint32_t kind, rank, batch_rank, row_rank, col_rank;
  uint32_t logical_rows, logical_cols, tile_rows, tiles_per_row;
  uint32_t grouped_dim, group_size, gather_dim;
  uint32_t reshape_source, source_rows, source_cols, source_tile_rows, source_tiles_per_row;
  uint32_t shape[MAX_RANK];
  uint32_t physical_shape[MAX_RANK];
  uint32_t reshape_shape[MAX_RANK];
  uint32_t dim_strides[MAX_RANK];
  uint32_t dim_offsets[MAX_RANK];
  uint32_t batch_dims[MAX_RANK];
  uint32_t row_dims[MAX_RANK];
  uint32_t col_dims[MAX_RANK];
};

struct GatherIndexReader {
  InterleavedAddrGenFast<true> indices;
  uint32_t cb;
  uint32_t loaded_tile;
};

void load_array(uint32_t base, uint32_t *target) {
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    target[i] = A(base + i);
  }
}

View load_view(uint32_t arg_view_kind) {
  const uint32_t arg_view_shape = arg_view_kind + 17;
  const uint32_t arg_view_physical_shape = arg_view_shape + MAX_RANK;
  const uint32_t arg_view_reshape_shape = arg_view_physical_shape + MAX_RANK;
  const uint32_t arg_view_dim_strides = arg_view_reshape_shape + MAX_RANK;
  const uint32_t arg_view_dim_offsets = arg_view_dim_strides + MAX_RANK;
  const uint32_t arg_view_batch_dims = arg_view_dim_offsets + MAX_RANK;
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
  view.gather_dim = A(arg_view_kind + 11);
  view.reshape_source = A(arg_view_kind + 12);
  view.source_rows = A(arg_view_kind + 13);
  view.source_cols = A(arg_view_kind + 14);
  view.source_tile_rows = A(arg_view_kind + 15);
  view.source_tiles_per_row = A(arg_view_kind + 16);
  load_array(arg_view_shape, view.shape);
  load_array(arg_view_physical_shape, view.physical_shape);
  load_array(arg_view_reshape_shape, view.reshape_shape);
  load_array(arg_view_dim_strides, view.dim_strides);
  load_array(arg_view_dim_offsets, view.dim_offsets);
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

GatherIndexReader make_gather_index_reader(uint32_t addr, uint32_t cb) {
  return GatherIndexReader{
      .indices = {
          .bank_base_address = addr,
          .page_size = sizeof(uint32_t) * TILE_R * TILE_C,
          .data_format = DataFormat::Int32,
      },
      .cb = cb,
      .loaded_tile = INVALID_TILE,
  };
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
  auto physical_index = [&](uint32_t dim) -> uint32_t {
    uint32_t index = indices[dim];
    if (dim == view.grouped_dim) {
      index /= view.group_size;
    }
    return index * view.dim_strides[dim] + view.dim_offsets[dim];
  };
  if (view.reshape_source != 0) {
    uint32_t flat = 0;
    for (uint32_t dim = 0; dim < view.rank; ++dim) {
      flat = flat * view.reshape_shape[dim] + physical_index(dim);
    }
    uint32_t col = flat % view.source_cols;
    uint32_t row_major = flat / view.source_cols;
    uint32_t row = row_major % view.source_rows;
    uint32_t batch = row_major / view.source_rows;
    *row_in_tile = row % TILE_R;
    *col_in_tile = col % TILE_C;
    return (batch * view.source_tile_rows + row / TILE_R) *
               view.source_tiles_per_row +
           col / TILE_C;
  }
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

void ensure_gather_index_tile(GatherIndexReader &reader, uint32_t tile) {
  if (reader.loaded_tile == tile) {
    return;
  }
  if (reader.loaded_tile != INVALID_TILE) {
    cb_pop_front(reader.cb, 1);
  }
  cb_reserve_back(reader.cb, 1);
  noc_async_read_tile(tile, reader.indices, get_write_ptr(reader.cb));
  noc_async_read_barrier();
  cb_push_back(reader.cb, 1);
  cb_wait_front(reader.cb, 1);
  reader.loaded_tile = tile;
}

int32_t read_gather_index(GatherIndexReader &reader, uint32_t logical_index) {
  uint32_t tile = logical_index / TILE_R;
  uint32_t row = logical_index % TILE_R;
  ensure_gather_index_tile(reader, tile);
  volatile tt_l1_ptr int32_t *values =
      reinterpret_cast<volatile tt_l1_ptr int32_t *>(get_read_ptr(reader.cb));
  return values[tile_element_index(row, 0)];
}

uint32_t tile_id_for_indices(const View &view, GatherIndexReader &gather_indices,
                             const uint32_t *indices, uint32_t *row_in_tile,
                             uint32_t *col_in_tile, bool *valid) {
  auto physical_index = [&](uint32_t dim) -> uint32_t {
    uint32_t index = indices[dim];
    if (dim == view.grouped_dim) {
      index /= view.group_size;
    }
    if (dim == view.gather_dim) {
      int32_t gathered = read_gather_index(gather_indices, index);
      if (gathered < 0 || static_cast<uint32_t>(gathered) >= view.shape[dim]) {
        *valid = false;
        return 0u;
      }
      index = static_cast<uint32_t>(gathered);
    }
    uint32_t physical = index * view.dim_strides[dim] + view.dim_offsets[dim];
    uint32_t extent = view.reshape_source != 0 ? view.reshape_shape[dim] : view.physical_shape[dim];
    if (physical >= extent) {
      *valid = false;
      return 0u;
    }
    return physical;
  };
  if (view.reshape_source != 0) {
    uint32_t flat = 0;
    for (uint32_t dim = 0; dim < view.rank; ++dim) {
      flat = flat * view.reshape_shape[dim] + physical_index(dim);
      if (!*valid) {
        return 0;
      }
    }
    uint32_t col = flat % view.source_cols;
    uint32_t row_major = flat / view.source_cols;
    uint32_t row = row_major % view.source_rows;
    uint32_t batch = row_major / view.source_rows;
    *row_in_tile = row % TILE_R;
    *col_in_tile = col % TILE_C;
    return (batch * view.source_tile_rows + row / TILE_R) *
               view.source_tiles_per_row +
           col / TILE_C;
  }
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

// GQA views often need one row from many source tiles. Keep the scratch tile in
// normal tile layout so the existing row-copy helpers handle placement.
template <uint32_t DatumBytes>
void read_source_row(const InterleavedAddrGenFast<true> &input, uint32_t source_tile,
                     uint32_t source_row, uint32_t cb_source) {
  cb_reserve_back(cb_source, 1);
  uint32_t l1_addr = get_write_ptr(cb_source);
  const uint32_t face0 = tile_element_index(source_row, 0) * DatumBytes;
  const uint32_t face1 = tile_element_index(source_row, FACE_C) * DatumBytes;
  noc_async_read(get_noc_addr(source_tile, input, face0), l1_addr + face0,
                 FACE_C * DatumBytes);
  noc_async_read(get_noc_addr(source_tile, input, face1), l1_addr + face1,
                 FACE_C * DatumBytes);
  noc_async_read_barrier();
  cb_push_back(cb_source, 1);
  cb_wait_front(cb_source, 1);
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

uint32_t physical_index_for_dim(const View &view, uint32_t dim, uint32_t index) {
  if (dim == view.grouped_dim) {
    index /= view.group_size;
  }
  return index * view.dim_strides[dim] + view.dim_offsets[dim];
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
                                         uint32_t cb_source,
                                         GatherIndexReader &gather_indices) {
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
  if (valid_rows != TILE_R || valid_cols != TILE_C ||
      view.gather_dim != GROUPED_DIM_NONE) {
    zero_tile_at(dst_addr, tile_bytes);
  }

  const uint32_t row_dim = view.row_dims[0];
  const bool can_stride_source_tiles =
      view.reshape_source == 0 && view.gather_dim == GROUPED_DIM_NONE &&
      row_dim != view.grouped_dim;
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

  if (can_stride_source_tiles && strided_source_col == 0 && valid_cols == TILE_C) {
    for (uint32_t row = 0; row < valid_rows; ++row) {
      read_source_row<DatumBytes>(
          input, source_tile_base + row * source_tile_stride, strided_source_row, cb_source);
      copy_source_row_to_dst_row<DatumBytes>(
          cb_source, dst_addr, strided_source_row, 0, row, 0, valid_cols);
      cb_pop_front(cb_source, 1);
    }
    return;
  }

  for (uint32_t row = 0; row < valid_rows; ++row) {
    uint32_t source_row = strided_source_row;
    uint32_t source_col = strided_source_col;
    uint32_t source_tile = source_tile_base + row * source_tile_stride;
    if (!can_stride_source_tiles) {
      indices[row_dim] = row_base + row;
      bool valid = true;
      if (view.gather_dim == GROUPED_DIM_NONE) {
        source_tile = tile_id_for_indices(view, indices, &source_row, &source_col);
      } else {
        source_tile = tile_id_for_indices(
            view, gather_indices, indices, &source_row, &source_col, &valid);
      }
      if (!valid) {
        continue;
      }
    }
    if (source_col == 0 && valid_cols == TILE_C) {
      read_source_row<DatumBytes>(input, source_tile, source_row, cb_source);
    } else {
      read_source_tile(input, source_tile, cb_source);
    }
    copy_source_row_to_dst_row<DatumBytes>(
        cb_source, dst_addr, source_row, source_col, row, 0, valid_cols);
    cb_pop_front(cb_source, 1);
  }
}

template <uint32_t DatumBytes>
void fill_generic_tile_impl(const InterleavedAddrGenFast<true> &input, const View &view,
                            uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                            uint32_t dst_addr, uint32_t tile_bytes, uint32_t cb_source,
                            GatherIndexReader &gather_indices) {
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    zero_tile_at(dst_addr, tile_bytes);
    return;
  }
  if (row_base + TILE_R > view.logical_rows || col_base + TILE_C > view.logical_cols ||
      view.gather_dim != GROUPED_DIM_NONE) {
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
      bool valid = true;
      uint32_t source_tile = view.gather_dim == GROUPED_DIM_NONE
                                 ? tile_id_for_indices(view, indices, &source_row, &source_col)
                                 : tile_id_for_indices(view, gather_indices, indices, &source_row,
                                                       &source_col, &valid);
      if (!valid) {
        continue;
      }
      if (view.gather_dim == GROUPED_DIM_NONE) {
        ensure_source_tile(input, source_tile, cb_source, &loaded_tile);
      } else {
        read_source_tile(input, source_tile, cb_source);
      }
      copy_element_from_source<DatumBytes>(cb_source, dst_addr, source_row, source_col, row,
                                           col);
      if (view.gather_dim != GROUPED_DIM_NONE) {
        cb_pop_front(cb_source, 1);
      }
    }
  }
  if (loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
}

void fill_generic_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                       uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                       uint32_t dst_addr, uint32_t tile_bytes, uint32_t cb_source,
                       GatherIndexReader &gather_indices);

void fill_generic_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                       uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                       uint32_t dst_addr, uint32_t tile_bytes, uint32_t cb_source) {
  GatherIndexReader no_gather = make_gather_index_reader(0, cb_source);
  fill_generic_tile(
      input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source,
      no_gather);
}

void fill_generic_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                       uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                       uint32_t dst_addr, uint32_t tile_bytes, uint32_t cb_source,
                       GatherIndexReader &gather_indices) {
  const bool prefix_row_inner_col = is_prefix_row_inner_col_view(view);
  if (tile_bytes == sizeof(uint32_t) * TILE_R * TILE_C) {
    if (prefix_row_inner_col) {
      fill_prefix_row_inner_col_tile_impl<sizeof(uint32_t)>(
          input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source,
          gather_indices);
    } else {
      fill_generic_tile_impl<sizeof(uint32_t)>(
          input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source,
          gather_indices);
    }
  } else {
    if (prefix_row_inner_col) {
      fill_prefix_row_inner_col_tile_impl<sizeof(uint16_t)>(
          input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source,
          gather_indices);
    } else {
      fill_generic_tile_impl<sizeof(uint16_t)>(
          input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source,
          gather_indices);
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
                                    uint32_t tile_bytes, uint32_t cb_source,
                                    GatherIndexReader &gather_indices) {
  uint32_t row_base = row_tile * TILE_R;
  uint32_t col_base = col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    zero_tile_at(dst_addr, tile_bytes);
    return;
  }
  const uint32_t valid_rows = min_u32(TILE_R, view.logical_rows - row_base);
  const uint32_t valid_cols = min_u32(TILE_C, view.logical_cols - col_base);
  if (valid_rows != TILE_R || valid_cols != TILE_C ||
      view.gather_dim != GROUPED_DIM_NONE) {
    zero_tile_at(dst_addr, tile_bytes);
  }

  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);

  TiledIndexMap map = {view.row_dims[0], view.col_dims[0]};
  indices[map.source_row_dim] = row_base;
  const uint32_t source_row_base =
      physical_index_for_dim(view, view.rank - 2, indices[view.rank - 2]);
  const uint32_t source_col_base =
      physical_index_for_dim(view, view.rank - 1, row_base);
  const uint32_t source_row_in_tile = source_row_base % TILE_R;
  const uint32_t source_col_in_tile = source_col_base % TILE_C;
  const uint32_t source_tile_row = source_row_base / TILE_R;
  const uint32_t source_tile_col = source_col_base / TILE_C;

  const bool can_stride_source_tiles =
      view.reshape_source == 0 && view.gather_dim == GROUPED_DIM_NONE &&
      map.source_col_dim + 2 < view.rank &&
      map.source_col_dim != view.grouped_dim;
  uint32_t source_tile_base = 0;
  uint32_t source_tile_stride = 0;
  if (can_stride_source_tiles) {
    uint32_t prefix = 0;
    for (uint32_t dim = 0; dim + 2 < view.rank; ++dim) {
      uint32_t index = dim == map.source_col_dim ? col_base : indices[dim];
      prefix = prefix * view.physical_shape[dim] +
               physical_index_for_dim(view, dim, index);
    }
    source_tile_base =
        (prefix * view.tile_rows + source_tile_row) * view.tiles_per_row + source_tile_col;
    source_tile_stride = view.tile_rows * view.tiles_per_row;
    for (uint32_t dim = map.source_col_dim + 1; dim + 2 < view.rank; ++dim) {
      source_tile_stride *= view.physical_shape[dim];
    }
  }

  for (uint32_t col = 0; col < valid_cols; ++col) {
    uint32_t source_tile = source_tile_base + col * source_tile_stride;
    uint32_t source_row = source_row_in_tile;
    uint32_t source_col = source_col_in_tile;
    if (!can_stride_source_tiles) {
      uint32_t logical_col = col_base + col;
      indices[map.source_col_dim] = logical_col;
      if (view.gather_dim == GROUPED_DIM_NONE) {
        uint32_t prefix = 0;
        for (uint32_t dim = 0; dim + 2 < view.rank; ++dim) {
          uint32_t index = dim == map.source_col_dim ? logical_col : indices[dim];
          prefix = prefix * view.physical_shape[dim] +
                   physical_index_for_dim(view, dim, index);
        }
        source_tile =
            (prefix * view.tile_rows + source_tile_row) * view.tiles_per_row + source_tile_col;
      } else {
        bool valid = true;
        source_tile = tile_id_for_indices(
            view, gather_indices, indices, &source_row, &source_col, &valid);
        if (!valid) {
          continue;
        }
      }
    }
    if (valid_rows == TILE_R && source_col == 0) {
      read_source_row<DatumBytes>(input, source_tile, source_row, cb_source);
    } else {
      read_source_tile(input, source_tile, cb_source);
    }
    copy_source_row_to_dst_col<DatumBytes>(
        cb_source, dst_addr, source_row, source_col, 0, col, valid_rows);
    cb_pop_front(cb_source, 1);
  }
}

void fill_tiled_index_map_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                               uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                               uint32_t dst_addr, uint32_t tile_bytes,
                               uint32_t cb_source,
                               GatherIndexReader &gather_indices);

void fill_tiled_index_map_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                               uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                               uint32_t dst_addr, uint32_t tile_bytes,
                               uint32_t cb_source) {
  GatherIndexReader no_gather = make_gather_index_reader(0, cb_source);
  fill_tiled_index_map_tile(
      input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source,
      no_gather);
}

void fill_tiled_index_map_tile(const InterleavedAddrGenFast<true> &input, const View &view,
                               uint32_t batch, uint32_t row_tile, uint32_t col_tile,
                               uint32_t dst_addr, uint32_t tile_bytes,
                               uint32_t cb_source,
                               GatherIndexReader &gather_indices) {
  if (tile_bytes == sizeof(uint32_t) * TILE_R * TILE_C) {
    fill_tiled_index_map_tile_impl<sizeof(uint32_t)>(
        input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source,
        gather_indices);
  } else {
    fill_tiled_index_map_tile_impl<sizeof(uint16_t)>(
        input, view, batch, row_tile, col_tile, dst_addr, tile_bytes, cb_source,
        gather_indices);
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
