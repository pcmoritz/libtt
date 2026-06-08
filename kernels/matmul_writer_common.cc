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
void write_contiguous_single_row_tile(
    const InterleavedAddrGenFast<true> &out_gen,
    uint32_t dst_tile,
    uint32_t src_l1_addr,
    uint32_t element_bytes) {
  constexpr uint32_t row = 0;
  for (uint32_t face_col = 0; face_col < 2; ++face_col) {
    const uint32_t col = face_col * FACE_C;
    const uint32_t offset = tile_element_index(row, col) * element_bytes;
    noc_async_write(
        src_l1_addr + offset,
        get_noc_addr(dst_tile, out_gen, offset),
        FACE_C * element_bytes);
  }
}
void write_output_tile(const InterleavedAddrGenFast<true> &out_gen, const View &output_view,
                       uint32_t batch, uint32_t canonical_row_tile,
                       uint32_t canonical_col_tile, uint32_t output_batch_stride,
                       uint32_t logical_nt, uint32_t src_l1_addr, uint32_t element_bytes,
                       uint32_t cb_scratch) {
  if (output_view.kind == VIEW_CONTIGUOUS) {
    const uint32_t dst_tile =
        batch * output_batch_stride + canonical_row_tile * logical_nt + canonical_col_tile;
    if (output_view.logical_rows == 1 &&
        (canonical_col_tile + 1) * TILE_C <= output_view.logical_cols) {
      write_contiguous_single_row_tile(out_gen, dst_tile, src_l1_addr, element_bytes);
    } else {
      noc_async_write_tile(dst_tile, out_gen, src_l1_addr);
    }
    return;
  }
  const uint32_t row_base = canonical_row_tile * TILE_R;
  const uint32_t col_base = canonical_col_tile * TILE_C;
  cb_reserve_back(cb_scratch, 1);
  uint32_t scratch_l1_addr = get_write_ptr(cb_scratch);
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
