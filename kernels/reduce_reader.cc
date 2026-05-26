#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t FACE_R = 16;
constexpr uint32_t FACE_C = 16;
constexpr uint32_t INVALID_TILE = 0xffffffffu;
constexpr uint32_t RANK = REDUCE_RANK;
constexpr uint32_t REDUCE_DIM = REDUCE_DIMENSION;
constexpr uint32_t OUT_RANK = RANK - 1;
constexpr uint32_t COORD_COUNT = RANK == 0 ? 1 : RANK;
constexpr uint32_t OUT_COORD_COUNT = OUT_RANK == 0 ? 1 : OUT_RANK;
constexpr uint32_t INPUT_SHAPE[COORD_COUNT] = REDUCE_INPUT_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[OUT_COORD_COUNT] = REDUCE_OUTPUT_SHAPE;
constexpr uint32_t INPUT_TILE_ROWS = REDUCE_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = REDUCE_INPUT_TILES_PER_ROW;
constexpr uint32_t INNER_OUTPUT_TILES = REDUCE_INNER_OUTPUT_TILES;
using Element = REDUCE_ELEMENT_TYPE;
constexpr Element IDENTITY = static_cast<Element>(REDUCE_IDENTITY);

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
  return ((face_row * 2 + face_col) * FACE_R * FACE_C) + row_in_face * FACE_C +
         col_in_face;
}

void fill_tile(uint32_t cb, Element value) {
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb));
  uint32_t elements = get_tile_size(cb) / sizeof(Element);
  for (uint32_t i = 0; i < elements; ++i) {
    ptr[i] = value;
  }
}

void fill_padded_columns(uint32_t tile_l1_addr, uint32_t valid_cols,
                         uint32_t identity_bits) {
  volatile tt_l1_ptr uint32_t *tile =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(tile_l1_addr);
  for (uint32_t row = 0; row < TILE_R; ++row) {
    for (uint32_t col = valid_cols; col < TILE_C; ++col) {
      tile[tile_element_index(row, col)] = identity_bits;
    }
  }
}

uint32_t read_tile_to_reserved_cb(const InterleavedAddrGenFast<true> &input,
                                  uint32_t tile_id, uint32_t cb) {
  cb_reserve_back(cb, 1);
  uint32_t l1_addr = get_write_ptr(cb);
  noc_async_read_tile(tile_id, input, l1_addr);
  noc_async_read_barrier();
  return l1_addr;
}

void read_tile_to_front(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
                        uint32_t cb) {
  read_tile_to_reserved_cb(input, tile_id, cb);
  cb_push_back(cb, 1);
  cb_wait_front(cb, 1);
}

void decode_output_prefix(uint32_t prefix, uint32_t coords[OUT_COORD_COUNT]) {
  for (uint32_t dim = 0; dim < OUT_RANK; ++dim) {
    coords[dim] = 0;
  }
  if constexpr (OUT_RANK >= 3) {
    for (uint32_t index = 0; index < OUT_RANK - 2; ++index) {
      uint32_t dim = OUT_RANK - 3 - index;
      coords[dim] = prefix % OUTPUT_SHAPE[dim];
      prefix /= OUTPUT_SHAPE[dim];
    }
  }
}

bool output_coords_for_lane(uint32_t group, uint32_t lane,
                            uint32_t coords[OUT_COORD_COUNT]) {
  uint32_t output_col = (group % INNER_OUTPUT_TILES) * TILE_C + lane;
  if constexpr (OUT_RANK == 1) {
    if (output_col >= OUTPUT_SHAPE[0]) {
      return false;
    }
    coords[0] = output_col;
    return true;
  } else {
    uint32_t row_group = group / INNER_OUTPUT_TILES;
    uint32_t prefix = row_group / OUTPUT_SHAPE[OUT_RANK - 2];
    uint32_t output_row = row_group % OUTPUT_SHAPE[OUT_RANK - 2];
    decode_output_prefix(prefix, coords);
    if (output_col >= OUTPUT_SHAPE[OUT_RANK - 1]) {
      return false;
    }
    coords[OUT_RANK - 2] = output_row;
    coords[OUT_RANK - 1] = output_col;
    return true;
  }
}

void input_coords_for_output(const uint32_t output_coords[OUT_COORD_COUNT],
                             uint32_t reduce_index,
                             uint32_t input_coords[COORD_COUNT]) {
  uint32_t output_dim = 0;
  for (uint32_t input_dim = 0; input_dim < RANK; ++input_dim) {
    if (input_dim == REDUCE_DIM) {
      input_coords[input_dim] = reduce_index;
    } else {
      input_coords[input_dim] = output_coords[output_dim];
      ++output_dim;
    }
  }
}

Location input_location(const uint32_t coords[COORD_COUNT]) {
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

void ensure_source_tile(const InterleavedAddrGenFast<true> &input,
                        uint32_t requested_tile, uint32_t *loaded_tile) {
  constexpr uint32_t cb_source = tt::CBIndex::c_1;
  if (requested_tile == *loaded_tile) {
    return;
  }
  if (*loaded_tile != INVALID_TILE) {
    cb_pop_front(cb_source, 1);
  }
  read_tile_to_front(input, requested_tile, cb_source);
  *loaded_tile = requested_tile;
}

Element read_input_element(const InterleavedAddrGenFast<true> &input,
                           const uint32_t coords[COORD_COUNT],
                           uint32_t *loaded_source_tile) {
  constexpr uint32_t cb_source = tt::CBIndex::c_1;
  Location source = input_location(coords);
  ensure_source_tile(input, source.tile, loaded_source_tile);
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_read_ptr(cb_source));
  return ptr[tile_element_index(source.row, source.col)];
}

void write_reduce_element(uint32_t row, uint32_t col, Element value) {
  constexpr uint32_t cb_reduce = tt::CBIndex::c_0;
  volatile tt_l1_ptr Element *ptr =
      reinterpret_cast<volatile tt_l1_ptr Element *>(get_write_ptr(cb_reduce));
  ptr[tile_element_index(row, col)] = value;
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t group_offset = get_arg_val<uint32_t>(1);
  uint32_t reduce_groups = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_reduce = tt::CBIndex::c_0;

  if constexpr (REDUCE_LAST_DIM_TILED) {
    uint32_t width_tiles = get_arg_val<uint32_t>(3);
    uint32_t valid_last_width = get_arg_val<uint32_t>(4);
    uint32_t padding_identity_bits = get_arg_val<uint32_t>(5);
    const InterleavedAddrGenFast<true> input = {
        .bank_base_address = input_addr,
        .page_size = get_tile_size(cb_reduce),
        .data_format = get_dataformat(cb_reduce),
    };

    for (uint32_t group = 0; group < reduce_groups; ++group) {
      uint32_t tile_base = (group_offset + group) * width_tiles;
      for (uint32_t wt = 0; wt < width_tiles; ++wt) {
        cb_reserve_back(cb_reduce, 1);
        uint32_t tile_l1_addr = get_write_ptr(cb_reduce);
        noc_async_read_tile(tile_base + wt, input, tile_l1_addr);
        noc_async_read_barrier();
        if (wt == width_tiles - 1 && valid_last_width < TILE_C) {
          fill_padded_columns(tile_l1_addr, valid_last_width, padding_identity_bits);
        }
        cb_push_back(cb_reduce, 1);
      }
    }
    return;
  }

  constexpr uint32_t cb_source = tt::CBIndex::c_1;
  uint32_t reduce_count = get_arg_val<uint32_t>(3);
  const InterleavedAddrGenFast<true> source_input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_source),
      .data_format = get_dataformat(cb_source),
  };

  for (uint32_t group = 0; group < reduce_groups; ++group) {
    uint32_t global_group = group_offset + group;
    for (uint32_t reduce_index = 0; reduce_index < reduce_count; ++reduce_index) {
      cb_reserve_back(cb_reduce, 1);
      fill_tile(cb_reduce, IDENTITY);

      uint32_t loaded_source_tile = INVALID_TILE;
      for (uint32_t lane = 0; lane < TILE_R; ++lane) {
        uint32_t output_coords[OUT_COORD_COUNT];
        if (!output_coords_for_lane(global_group, lane, output_coords)) {
          continue;
        }

        uint32_t input_coords[COORD_COUNT];
        input_coords_for_output(output_coords, reduce_index, input_coords);
        Element value = read_input_element(source_input, input_coords, &loaded_source_tile);
        write_reduce_element(0, lane, value);
      }

      if (loaded_source_tile != INVALID_TILE) {
        cb_pop_front(cb_source, 1);
      }
      cb_push_back(cb_reduce, 1);
    }
  }
}
