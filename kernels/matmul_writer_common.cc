#include <cstdint>
namespace {
constexpr uint32_t ARG_RHS_VIEW_KIND = 37;
constexpr uint32_t ARG_OUTPUT_VIEW_KIND = ARG_RHS_VIEW_KIND + VIEW_ARG_COUNT;
uint32_t output_tile_for_element(const View &view, uint32_t batch, uint32_t logical_row,
                                 uint32_t logical_col, uint32_t *row_in_tile,
                                 uint32_t *col_in_tile) {
  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  decompose_into_dims(logical_row, view.row_dims, view.row_rank, view.shape, indices);
  decompose_into_dims(logical_col, view.col_dims, view.col_rank, view.shape, indices);
  return tile_id_for_indices(view, indices, row_in_tile, col_in_tile);
}
void copy_l1_bytes(uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t bytes) {
  volatile tt_l1_ptr uint16_t *dst =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(dst_l1_addr);
  volatile tt_l1_ptr uint16_t *src =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(src_l1_addr);
  for (uint32_t i = 0; i < bytes / sizeof(uint16_t); ++i) {
    dst[i] = src[i];
  }
}
void write_output_run(const InterleavedAddrGenFast<true> &out_gen, uint32_t dst_tile,
                      uint32_t dst_offset, uint32_t src_l1_addr, uint32_t bytes,
                      uint32_t scratch_l1_addr) {
  while (bytes > 0) {
    // Blackhole DRAM NOC writes need 16-byte alignment. Full aligned blocks can
    // go directly; fragments use a scratch block so the actual write is aligned.
    const bool aligned = ((dst_offset | src_l1_addr) & 0xfu) == 0;
    if (aligned && bytes >= 16) {
      uint32_t direct_bytes = bytes & ~0xfu;
      noc_async_write(src_l1_addr, get_noc_addr(dst_tile, out_gen, dst_offset),
                      direct_bytes);
      src_l1_addr += direct_bytes;
      dst_offset += direct_bytes;
      bytes -= direct_bytes;
      continue;
    }

    const uint32_t dst_block = dst_offset & ~0xfu;
    const uint32_t block_offset = dst_offset - dst_block;
    uint32_t block_bytes = 16 - block_offset;
    if (block_bytes > bytes) {
      block_bytes = bytes;
    }
    if (block_offset != 0 || block_bytes != 16) {
      noc_async_read(get_noc_addr(dst_tile, out_gen, dst_block), scratch_l1_addr, 16);
      noc_async_read_barrier();
    }
    copy_l1_bytes(scratch_l1_addr + block_offset, src_l1_addr, block_bytes);
    noc_async_write(scratch_l1_addr, get_noc_addr(dst_tile, out_gen, dst_block), 16);
    noc_async_write_barrier();
    src_l1_addr += block_bytes;
    dst_offset += block_bytes;
    bytes -= block_bytes;
  }
}

bool is_grouped_row_output_view(const View &view) {
  return view.rank == 5 && view.batch_rank == 2 && view.row_rank == 2 &&
         view.col_rank == 1 && view.batch_dims[0] == 0 &&
         view.batch_dims[1] == 1 && view.row_dims[0] == 2 &&
         view.row_dims[1] == 3 && view.col_dims[0] == 4 &&
         view.shape[3] > 0 && view.shape[3] <= TILE_R &&
         TILE_R % view.shape[3] == 0;
}

void write_grouped_row_output_tile(const InterleavedAddrGenFast<true> &out_gen,
                                   const View &view, uint32_t batch,
                                   uint32_t canonical_row_tile,
                                   uint32_t canonical_col_tile,
                                   uint32_t src_l1_addr,
                                   uint32_t element_bytes,
                                   uint32_t cb_scratch) {
  const uint32_t row_base = canonical_row_tile * TILE_R;
  const uint32_t col_base = canonical_col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    return;
  }
  const uint32_t group_rows = view.shape[3];
  const uint32_t valid_cols =
      (view.logical_cols - col_base) < TILE_C ? (view.logical_cols - col_base) : TILE_C;
  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  indices[4] = col_base;

  cb_reserve_back(cb_scratch, 1);
  uint32_t scratch_l1_addr = get_write_ptr(cb_scratch);
  for (uint32_t row_group = 0; row_group < TILE_R; row_group += group_rows) {
    const uint32_t logical_row = row_base + row_group;
    if (logical_row >= view.logical_rows) {
      break;
    }
    indices[2] = logical_row / group_rows;
    indices[3] = 0;
    uint32_t dst_row = 0;
    uint32_t dst_col = 0;
    uint32_t dst_tile = tile_id_for_indices(view, indices, &dst_row, &dst_col);
    for (uint32_t inner = 0; inner < group_rows; ++inner) {
      if (logical_row + inner >= view.logical_rows) {
        break;
      }
      const uint32_t src_offset =
          tile_element_index(row_group + inner, 0) * element_bytes;
      const uint32_t dst_offset =
          tile_element_index(dst_row + inner, dst_col) * element_bytes;
      write_output_run(out_gen, dst_tile, dst_offset, src_l1_addr + src_offset,
                       valid_cols * element_bytes, scratch_l1_addr);
    }
  }
  cb_push_back(cb_scratch, 1);
  cb_pop_front(cb_scratch, 1);
}

bool is_row_by_grouped_col_output_view(const View &view) {
  return view.rank == 5 && view.batch_rank == 2 && view.row_rank == 1 &&
         view.col_rank == 2 && view.batch_dims[0] == 0 &&
         view.batch_dims[1] == 1 && view.row_dims[0] == 2 &&
         view.col_dims[0] == 3 && view.col_dims[1] == 4 &&
         view.shape[3] > 0 && view.shape[3] <= TILE_R &&
         view.shape[4] > 0 && view.shape[4] <= TILE_C;
}

void write_row_by_grouped_col_output_tile(
    const InterleavedAddrGenFast<true> &out_gen, const View &view,
    uint32_t batch, uint32_t canonical_row_tile, uint32_t canonical_col_tile,
    uint32_t src_l1_addr, uint32_t element_bytes, uint32_t cb_scratch) {
  const uint32_t row_base = canonical_row_tile * TILE_R;
  const uint32_t col_base = canonical_col_tile * TILE_C;
  if (row_base >= view.logical_rows || col_base >= view.logical_cols) {
    return;
  }
  const uint32_t token_count = view.shape[4];
  const uint32_t valid_rows =
      (view.logical_rows - row_base) < TILE_R ? (view.logical_rows - row_base) : TILE_R;
  uint32_t indices[MAX_RANK] = {};
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);

  cb_reserve_back(cb_scratch, 1);
  uint32_t scratch_l1_addr = get_write_ptr(cb_scratch);
  for (uint32_t row = 0; row < valid_rows; ++row) {
    indices[2] = row_base + row;
    uint32_t col = 0;
    while (col < TILE_C) {
      uint32_t logical_col = col_base + col;
      if (logical_col >= view.logical_cols) {
        break;
      }
      uint32_t repeat = logical_col / token_count;
      uint32_t token = logical_col - repeat * token_count;
      indices[3] = repeat;
      indices[4] = token;
      uint32_t dst_row = 0;
      uint32_t dst_col = 0;
      uint32_t dst_tile = tile_id_for_indices(view, indices, &dst_row, &dst_col);
      uint32_t run = 1;
      while (col + run < TILE_C && col_base + col + run < view.logical_cols) {
        uint32_t next_logical_col = col_base + col + run;
        if (next_logical_col / token_count != repeat) {
          break;
        }
        ++run;
      }
      const uint32_t src_offset = tile_element_index(row, col) * element_bytes;
      const uint32_t dst_offset = tile_element_index(dst_row, dst_col) * element_bytes;
      write_output_run(out_gen, dst_tile, dst_offset, src_l1_addr + src_offset,
                       run * element_bytes, scratch_l1_addr);
      col += run;
    }
  }
  cb_push_back(cb_scratch, 1);
  cb_pop_front(cb_scratch, 1);
}

void write_output_tile(const InterleavedAddrGenFast<true> &out_gen, const View &output_view,
                       uint32_t batch, uint32_t canonical_row_tile,
                       uint32_t canonical_col_tile, uint32_t output_batch_stride,
                       uint32_t logical_nt, uint32_t src_l1_addr, uint32_t element_bytes,
                       uint32_t cb_scratch) {
  if (output_view.kind == VIEW_CONTIGUOUS) {
    noc_async_write_tile(batch * output_batch_stride + canonical_row_tile * logical_nt +
                             canonical_col_tile,
                         out_gen, src_l1_addr);
    return;
  }
  if (is_grouped_row_output_view(output_view)) {
    write_grouped_row_output_tile(out_gen, output_view, batch,
                                  canonical_row_tile, canonical_col_tile,
                                  src_l1_addr, element_bytes, cb_scratch);
    return;
  }
  if (is_row_by_grouped_col_output_view(output_view)) {
    write_row_by_grouped_col_output_tile(
        out_gen, output_view, batch, canonical_row_tile, canonical_col_tile,
        src_l1_addr, element_bytes, cb_scratch);
    return;
  }
  const uint32_t row_base = canonical_row_tile * TILE_R;
  const uint32_t col_base = canonical_col_tile * TILE_C;
  uint32_t batch_indices[MAX_RANK] = {};
  decompose_into_dims(batch, output_view.batch_dims, output_view.batch_rank,
                      output_view.shape, batch_indices);
  uint32_t col_indices[TILE_C][MAX_RANK] = {};
  bool valid_cols[TILE_C] = {};
  for (uint32_t col = 0; col < TILE_C; ++col) {
    const uint32_t logical_col = col_base + col;
    if (logical_col >= output_view.logical_cols) {
      continue;
    }
    valid_cols[col] = true;
    decompose_into_dims(logical_col, output_view.col_dims,
                        output_view.col_rank, output_view.shape,
                        col_indices[col]);
  }
  cb_reserve_back(cb_scratch, 1);
  uint32_t scratch_l1_addr = get_write_ptr(cb_scratch);
  for (uint32_t row = 0; row < TILE_R; ++row) {
    const uint32_t logical_row = row_base + row;
    if (logical_row >= output_view.logical_rows) {
      continue;
    }
    uint32_t indices[MAX_RANK] = {};
    for (uint32_t i = 0; i < MAX_RANK; ++i) {
      indices[i] = batch_indices[i];
    }
    decompose_into_dims(logical_row, output_view.row_dims,
                        output_view.row_rank, output_view.shape, indices);
    uint32_t col = 0;
    while (col < TILE_C) {
      if (!valid_cols[col]) {
        break;
      }
      for (uint32_t i = 0; i < output_view.col_rank; ++i) {
        uint32_t dim = output_view.col_dims[i];
        indices[dim] = col_indices[col][dim];
      }
      uint32_t dst_row = 0;
      uint32_t dst_col = 0;
      const uint32_t dst_tile =
          tile_id_for_indices(output_view, indices, &dst_row, &dst_col);
      const uint32_t src_offset = tile_element_index(row, col) * element_bytes;
      const uint32_t dst_offset = tile_element_index(dst_row, dst_col) * element_bytes;
      uint32_t run = 1;
      while (col + run < TILE_C && valid_cols[col + run]) {
        for (uint32_t i = 0; i < output_view.col_rank; ++i) {
          uint32_t dim = output_view.col_dims[i];
          indices[dim] = col_indices[col + run][dim];
        }
        uint32_t next_dst_row = 0;
        uint32_t next_dst_col = 0;
        const uint32_t next_dst_tile = tile_id_for_indices(
            output_view, indices, &next_dst_row, &next_dst_col);
        const uint32_t next_src_offset = tile_element_index(row, col + run) * element_bytes;
        const uint32_t next_dst_offset = tile_element_index(next_dst_row, next_dst_col) * element_bytes;
        if (next_dst_tile != dst_tile ||
            next_src_offset != src_offset + run * element_bytes ||
            next_dst_offset != dst_offset + run * element_bytes) {
          break;
        }
        ++run;
      }
      write_output_run(out_gen, dst_tile, dst_offset, src_l1_addr + src_offset,
                       run * element_bytes, scratch_l1_addr);
      col += run;
    }
  }
  cb_push_back(cb_scratch, 1);
  cb_pop_front(cb_scratch, 1);
}
struct OutputDrain {
  View view;
  InterleavedAddrGenFast<true> gen;
  uint32_t tile_bytes, start, stride_w, stride_h, next_sb_w, next_sb_h;
  uint32_t sb_w, sb_h, sb_tiles, num_sb_w, num_sb_h;
  uint32_t logical_mt, logical_nt, col_offset, batch_stride;
};
OutputDrain load_output_drain() {
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_out);
  return {
      .view = load_view(ARG_OUTPUT_VIEW_KIND),
      .gen = {
          .bank_base_address = A(18),
          .page_size = tile_bytes,
          .data_format = get_dataformat(cb_out),
      },
      .tile_bytes = tile_bytes,
      .start = A(19),
      .stride_w = A(20),
      .stride_h = A(21),
      .next_sb_w = A(22),
      .next_sb_h = A(23),
      .sb_w = A(24),
      .sb_h = A(25),
      .sb_tiles = A(26),
      .num_sb_w = A(27),
      .num_sb_h = A(28),
      .logical_mt = A(29),
      .logical_nt = A(30),
      .col_offset = A(31),
      .batch_stride = A(36),
  };
}
void drain_output_blocks(const OutputDrain &output, uint32_t batch, bool valid_batch) {
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  constexpr uint32_t cb_scratch = tt::CBIndex::c_4;
  const uint32_t element_bytes = output.tile_bytes / (TILE_R * TILE_C);
  const uint32_t padded_nt = output.next_sb_h / output.sb_h;
  uint32_t sbh_start = output.start;
  for (uint32_t sbh = 0; sbh < output.num_sb_h; sbh++) {
    uint32_t sbw_start = sbh_start;
    for (uint32_t sbw = 0; sbw < output.num_sb_w; sbw++) {
      cb_wait_front(cb_out, output.sb_tiles);
      uint32_t l1_addr = get_read_ptr(cb_out);
      uint32_t row_start = sbw_start;
      for (uint32_t h = 0; h < output.sb_h; h++) {
        uint32_t tile_id = row_start;
        for (uint32_t w = 0; w < output.sb_w; w++) {
          const uint32_t out_row = tile_id / padded_nt;
          const uint32_t out_col = output.col_offset + tile_id - out_row * padded_nt;
          if (valid_batch && out_row < output.logical_mt &&
              out_col < output.logical_nt) {
            write_output_tile(output.gen, output.view, batch, out_row, out_col,
                              output.batch_stride, output.logical_nt, l1_addr,
                              element_bytes, cb_scratch);
          }
          l1_addr += output.tile_bytes;
          tile_id += output.stride_w;
        }
        row_start += output.stride_h;
      }
      noc_async_write_barrier();
      cb_pop_front(cb_out, output.sb_tiles);
      sbw_start += output.next_sb_w;
    }
    sbh_start += output.next_sb_h;
  }
}
}  // namespace
