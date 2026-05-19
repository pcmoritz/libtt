#include <cstdint>

namespace {
constexpr uint32_t ARG_RHS_VIEW_KIND = 37;
constexpr uint32_t ARG_OUTPUT_VIEW_KIND = ARG_RHS_VIEW_KIND + VIEW_ARG_COUNT;

uint32_t output_tile_for_element(
    const View &view,
    uint32_t batch,
    uint32_t logical_row,
    uint32_t logical_col,
    uint32_t *row_in_tile,
    uint32_t *col_in_tile) {
  uint32_t indices[MAX_RANK];
  for (uint32_t i = 0; i < MAX_RANK; ++i) {
    indices[i] = 0;
  }
  decompose_into_dims(batch, view.batch_dims, view.batch_rank, view.shape, indices);
  decompose_into_dims(logical_row, view.row_dims, view.row_rank, view.shape, indices);
  decompose_into_dims(logical_col, view.col_dims, view.col_rank, view.shape, indices);
  return tile_id_for_indices(view, indices, row_in_tile, col_in_tile);
}

void write_output_fragment(
    const InterleavedAddrGenFast<true> &out_gen,
    uint32_t dst_tile,
    uint32_t src_l1_addr,
    uint32_t src_offset,
    uint32_t dst_offset,
    uint32_t bytes) {
  noc_async_write(
      src_l1_addr + src_offset,
      get_noc_addr(dst_tile, out_gen, dst_offset),
      bytes);
}

bool output_rows_are_physical_tiles(const View &view) {
  if (view.kind == VIEW_CONTIGUOUS || view.row_rank != 1 || view.rank < 2) {
    return false;
  }
  const uint32_t physical_row_dim = view.rank - 2;
  for (uint32_t i = 0; i < view.col_rank; ++i) {
    if (view.col_dims[i] == physical_row_dim) {
      return true;
    }
  }
  return false;
}

void write_output_row_physical_tiles(
    const InterleavedAddrGenFast<true> &out_gen,
    const View &output_view,
    uint32_t cb_scratch,
    uint32_t batch,
    uint32_t canonical_row_tile,
    uint32_t first_col_tile,
    uint32_t col_tile_count,
    uint32_t src_l1_addr,
    uint32_t tile_bytes,
    uint32_t element_bytes) {
  (void)tile_bytes;
  const uint32_t row_base = canonical_row_tile * TILE_R;
  cb_reserve_back(cb_scratch, 1);
  uint32_t scratch_l1_addr = get_write_ptr(cb_scratch);
  for (uint32_t row = 0; row < TILE_R; ++row) {
    const uint32_t logical_row = row_base + row;
    if (logical_row >= output_view.logical_rows) {
      continue;
    }
    uint32_t current_tile = INVALID_TILE;
    uint32_t current_block = INVALID_TILE;
    bool have_block = false;
    for (uint32_t tile_col = 0; tile_col < col_tile_count; ++tile_col) {
      const uint32_t canonical_col_tile = first_col_tile + tile_col;
      const uint32_t col_base = canonical_col_tile * TILE_C;
      uint32_t source_tile_l1_addr = src_l1_addr + tile_col * tile_bytes;
      for (uint32_t col = 0; col < TILE_C; ++col) {
        const uint32_t logical_col = col_base + col;
        if (logical_col >= output_view.logical_cols) {
          break;
        }
        uint32_t dst_row = 0;
        uint32_t dst_col = 0;
        const uint32_t dst_tile = output_tile_for_element(
            output_view, batch, logical_row, logical_col, &dst_row, &dst_col);
        const uint32_t dst_offset =
            tile_element_index(dst_row, dst_col) * element_bytes;
        const uint32_t dst_block = dst_offset & ~0xfu;
        if (!have_block) {
          current_tile = dst_tile;
          current_block = dst_block;
          have_block = true;
          volatile tt_l1_ptr uint32_t *scratch =
              reinterpret_cast<volatile tt_l1_ptr uint32_t *>(scratch_l1_addr);
          for (uint32_t i = 0; i < 4; ++i) {
            scratch[i] = 0;
          }
        } else if (dst_tile != current_tile || dst_block != current_block) {
          noc_async_write(
              scratch_l1_addr,
              get_noc_addr(current_tile, out_gen, current_block),
              16);
          noc_async_write_barrier();
          current_tile = dst_tile;
          current_block = dst_block;
          volatile tt_l1_ptr uint32_t *scratch =
              reinterpret_cast<volatile tt_l1_ptr uint32_t *>(scratch_l1_addr);
          for (uint32_t i = 0; i < 4; ++i) {
            scratch[i] = 0;
          }
        }
        volatile tt_l1_ptr uint16_t *src =
            reinterpret_cast<volatile tt_l1_ptr uint16_t *>(
                source_tile_l1_addr + tile_element_index(row, col) * element_bytes);
        volatile tt_l1_ptr uint16_t *dst =
            reinterpret_cast<volatile tt_l1_ptr uint16_t *>(
                scratch_l1_addr + (dst_offset - current_block));
        for (uint32_t i = 0; i < element_bytes / sizeof(uint16_t); ++i) {
          dst[i] = src[i];
        }
      }
    }
    if (have_block) {
      noc_async_write(
          scratch_l1_addr,
          get_noc_addr(current_tile, out_gen, current_block),
          16);
      noc_async_write_barrier();
    }
  }
  cb_push_back(cb_scratch, 1);
  cb_pop_front(cb_scratch, 1);
}

void write_output_tile(
    const InterleavedAddrGenFast<true> &out_gen,
    const View &output_view,
    uint32_t batch,
    uint32_t canonical_row_tile,
    uint32_t canonical_col_tile,
    uint32_t output_batch_stride,
    uint32_t logical_nt,
    uint32_t src_l1_addr,
    uint32_t element_bytes) {
  if (output_view.kind == VIEW_CONTIGUOUS) {
    noc_async_write_tile(
        batch * output_batch_stride + canonical_row_tile * logical_nt + canonical_col_tile,
        out_gen,
        src_l1_addr);
    return;
  }

  const uint32_t row_base = canonical_row_tile * TILE_R;
  const uint32_t col_base = canonical_col_tile * TILE_C;
  for (uint32_t row = 0; row < TILE_R; ++row) {
    const uint32_t logical_row = row_base + row;
    if (logical_row >= output_view.logical_rows) {
      continue;
    }
    uint32_t col = 0;
    while (col < TILE_C) {
      const uint32_t logical_col = col_base + col;
      if (logical_col >= output_view.logical_cols) {
        break;
      }
      uint32_t dst_row = 0;
      uint32_t dst_col = 0;
      const uint32_t dst_tile = output_tile_for_element(
          output_view, batch, logical_row, logical_col, &dst_row, &dst_col);
      const uint32_t src_offset = tile_element_index(row, col) * element_bytes;
      const uint32_t dst_offset = tile_element_index(dst_row, dst_col) * element_bytes;
      uint32_t run = 1;
      while (col + run < TILE_C && col_base + col + run < output_view.logical_cols) {
        uint32_t next_dst_row = 0;
        uint32_t next_dst_col = 0;
        const uint32_t next_dst_tile = output_tile_for_element(
            output_view,
            batch,
            logical_row,
            col_base + col + run,
            &next_dst_row,
            &next_dst_col);
        const uint32_t next_src_offset =
            tile_element_index(row, col + run) * element_bytes;
        const uint32_t next_dst_offset =
            tile_element_index(next_dst_row, next_dst_col) * element_bytes;
        if (next_dst_tile != dst_tile ||
            next_src_offset != src_offset + run * element_bytes ||
            next_dst_offset != dst_offset + run * element_bytes) {
          break;
        }
        ++run;
      }
      write_output_fragment(
          out_gen,
          dst_tile,
          src_l1_addr,
          src_offset,
          dst_offset,
          run * element_bytes);
      col += run;
    }
  }
}

struct OutputDrain {
  View view;
  InterleavedAddrGenFast<true> gen;
  uint32_t tile_bytes;
  uint32_t start;
  uint32_t stride_w;
  uint32_t stride_h;
  uint32_t next_sb_w;
  uint32_t next_sb_h;
  uint32_t sb_w;
  uint32_t sb_h;
  uint32_t sb_tiles;
  uint32_t num_sb_w;
  uint32_t num_sb_h;
  uint32_t logical_mt;
  uint32_t logical_nt;
  uint32_t col_offset;
  uint32_t batch_stride;
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
      if (valid_batch && output_rows_are_physical_tiles(output.view) &&
          output.col_offset == 0 && output.sb_w == output.logical_nt) {
        for (uint32_t h = 0; h < output.sb_h; h++) {
          const uint32_t out_row = row_start / padded_nt;
          if (out_row < output.logical_mt) {
            write_output_row_physical_tiles(
                output.gen,
                output.view,
                cb_scratch,
                batch,
                out_row,
                0,
                output.sb_w,
                l1_addr,
                output.tile_bytes,
                element_bytes);
          }
          l1_addr += output.sb_w * output.tile_bytes;
          row_start += output.stride_h;
        }
      } else {
        for (uint32_t h = 0; h < output.sb_h; h++) {
          uint32_t tile_id = row_start;
          for (uint32_t w = 0; w < output.sb_w; w++) {
            const uint32_t out_row = tile_id / padded_nt;
            const uint32_t out_col =
                output.col_offset + tile_id - out_row * padded_nt;
            if (valid_batch && out_row < output.logical_mt &&
                out_col < output.logical_nt) {
              write_output_tile(
                  output.gen,
                  output.view,
                  batch,
                  out_row,
                  out_col,
                  output.batch_stride,
                  output.logical_nt,
                  l1_addr,
                  element_bytes);
            }
            l1_addr += output.tile_bytes;
            tile_id += output.stride_w;
          }
          row_start += output.stride_h;
        }
      }
      noc_async_write_barrier();
      cb_pop_front(cb_out, output.sb_tiles);
      sbw_start += output.next_sb_w;
    }
    sbh_start += output.next_sb_h;
  }
}
}  // namespace
