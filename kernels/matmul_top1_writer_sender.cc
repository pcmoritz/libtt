#include <cstdint>

namespace {
constexpr uint32_t ARG_RHS_VIEW_KIND = 38;
constexpr uint32_t ARG_OUTPUT_VIEW_KIND = ARG_RHS_VIEW_KIND + VIEW_ARG_COUNT;
constexpr uint32_t BF16_NEG_INF = 0xff80;

uint32_t ordered_float_key(uint32_t bits) {
  return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

uint32_t value_key(uint32_t value_bits) {
  return ordered_float_key(value_bits << 16);
}

bool candidate_before(uint32_t lhs_key, uint32_t lhs_index, uint32_t rhs_key,
                      uint32_t rhs_index) {
  return lhs_key > rhs_key || (lhs_key == rhs_key && lhs_index < rhs_index);
}

struct Top1 {
  bool have_best;
  uint32_t key;
  uint32_t value;
  uint32_t index;
};

struct OutputTop1 {
  View view;
  uint32_t start, stride_w, stride_h, next_sb_w, next_sb_h;
  uint32_t sb_w, sb_h, sb_tiles, num_sb_w, num_sb_h;
  uint32_t logical_mt, logical_nt, col_offset, partial_tile_id;
};

OutputTop1 load_output_top1() {
  return {
      .view = load_view(ARG_OUTPUT_VIEW_KIND),
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
      .partial_tile_id = A(32),
  };
}

void consider_output_tile(Top1 *best, uint32_t l1_addr, uint32_t col_tile,
                          uint32_t logical_cols) {
  volatile tt_l1_ptr uint16_t *tile =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(l1_addr);
  for (uint32_t col = 0; col < TILE_C; ++col) {
    uint32_t index = col_tile * TILE_C + col;
    if (index >= logical_cols) {
      break;
    }
    uint32_t value = tile[tile_element_index(0, col)];
    uint32_t key = value_key(value);
    if (!best->have_best || candidate_before(key, index, best->key, best->index)) {
      best->have_best = true;
      best->key = key;
      best->value = value;
      best->index = index;
    }
  }
}

void drain_output_top1(const OutputTop1 &output, uint32_t batch, bool valid_batch,
                       uint32_t wave_stride_tiles, Top1 *best) {
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_out);
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
          const uint32_t global_col = batch * wave_stride_tiles + out_col;
          if (valid_batch && out_row == 0 && global_col < output.logical_nt) {
            consider_output_tile(best, l1_addr, global_col, output.view.logical_cols);
          }
          l1_addr += tile_bytes;
          tile_id += output.stride_w;
        }
        row_start += output.stride_h;
      }
      cb_pop_front(cb_out, output.sb_tiles);
      sbw_start += output.next_sb_w;
    }
    sbh_start += output.next_sb_h;
  }
}

void zero_tile(uint32_t cb) {
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb));
  uint32_t words = get_tile_size(cb) / sizeof(uint32_t);
  for (uint32_t i = 0; i < words; ++i) {
    ptr[i] = 0;
  }
}

void write_partial(const OutputTop1 &output, const Top1 &best) {
  constexpr uint32_t cb_pairs = tt::CBIndex::c_4;

  const InterleavedAddrGenFast<true> partial_pairs = {
      .bank_base_address = A(18),
      .page_size = get_tile_size(cb_pairs),
      .data_format = get_dataformat(cb_pairs),
  };

  cb_reserve_back(cb_pairs, 1);
  zero_tile(cb_pairs);
  volatile tt_l1_ptr uint32_t *pair_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb_pairs));
  pair_ptr[tile_element_index(0, 0)] = best.have_best ? best.value : BF16_NEG_INF;
  pair_ptr[tile_element_index(0, 1)] = best.have_best ? best.index : 0xffffffffu;
  noc_async_write_tile(output.partial_tile_id, partial_pairs, get_write_ptr(cb_pairs));
  noc_async_write_barrier();
  cb_push_back(cb_pairs, 1);
  cb_pop_front(cb_pairs, 1);
}

}  // namespace

void kernel_main() {
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t cb_source = tt::CBIndex::c_3;
  const uint32_t in1_tile_bytes = get_tile_size(cb_in1);
  const uint32_t block_w = A(5);
  const uint32_t block_h = A(6);
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t i1_nd = A(13);
  const uint32_t logical_nt = A(30);
  const uint32_t local_batch_count = A(33);
  const uint32_t batch_start = A(34);
  const uint32_t total_batch_count = A(35);
  const uint32_t rhs_batch_stride = A(36);
  const View view = load_view(ARG_RHS_VIEW_KIND);
  const OutputTop1 output_top1 = load_output_top1();
  Top1 best = {.have_best = false, .key = 0, .value = BF16_NEG_INF, .index = 0xffffffffu};

  volatile tt_l1_ptr uint32_t *sender_sem = SEM(16);
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(17);
  *recv_sem = VALID;

  const InterleavedAddrGenFast<true> in1_gen = {
      .bank_base_address = A(0),
      .page_size = in1_tile_bytes,
      .data_format = DataFormat::Float16_b,
  };

  for (uint32_t local_batch = 0; local_batch < local_batch_count; local_batch++) {
    const uint32_t batch = batch_start + local_batch;
    const bool valid_batch = batch < total_batch_count;
    uint32_t cur_block = A(1) + batch * rhs_batch_stride;
    for (uint32_t block = 0; block < nblocks; block++) {
      cb_reserve_back(cb_in1, block_tiles);
      uint32_t l1_addr = get_write_ptr(cb_in1);
      uint32_t start_addr = l1_addr;
      uint32_t row = cur_block;
      uint32_t block_bytes = 0;
      if (!valid_batch) {
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
          zero_tile_at(l1_addr, in1_tile_bytes);
          l1_addr += in1_tile_bytes;
          block_bytes += in1_tile_bytes;
        }
      } else if (view.kind == VIEW_CONTIGUOUS) {
        for (uint32_t h = 0; h < block_h; h++) {
          uint32_t tile_id = row;
          for (uint32_t w = 0; w < block_w; w++) {
            if (A(1) + w < logical_nt) {
              noc_async_read_tile(tile_id, in1_gen, l1_addr);
            }
            l1_addr += in1_tile_bytes;
            tile_id += A(2);
            block_bytes += in1_tile_bytes;
          }
          row += A(3);
        }
        noc_async_read_barrier();
      } else {
        uint32_t canonical_base = cur_block - batch * rhs_batch_stride;
        for (uint32_t h = 0; h < block_h; h++) {
          for (uint32_t w = 0; w < block_w; w++) {
            uint32_t canonical_tile = canonical_base + h * A(3) + w;
            uint32_t canonical_row_tile = canonical_tile / A(3);
            uint32_t canonical_col_tile = canonical_tile - canonical_row_tile * A(3);
            if (view.kind == VIEW_TILED_INDEX_MAP) {
              fill_tiled_index_map_tile(
                  in1_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  in1_tile_bytes,
                  cb_source);
            } else {
              fill_generic_tile(
                  in1_gen,
                  view,
                  batch,
                  canonical_row_tile,
                  canonical_col_tile,
                  l1_addr,
                  in1_tile_bytes,
                  cb_source);
            }
            l1_addr += in1_tile_bytes;
            block_bytes += in1_tile_bytes;
          }
        }
      }
      cur_block += A(4);

      if (i1_nd > 0) {
        noc_semaphore_wait(sender_sem, i1_nd);
        noc_semaphore_set(sender_sem, 0);
        uint64_t ma = get_noc_multicast_addr(A(9), A(10), A(11), A(12), start_addr);
        noc_async_write_multicast(start_addr, ma, block_bytes, i1_nd);
        noc_async_writes_flushed();
        noc_semaphore_set_multicast(
            get_semaphore(A(17)),
            get_noc_multicast_addr(A(9), A(10), A(11), A(12), get_semaphore(A(17))),
            i1_nd);
      }
      cb_push_back(cb_in1, block_tiles);
    }

    drain_output_top1(output_top1, batch, valid_batch, rhs_batch_stride, &best);
  }

  write_partial(output_top1, best);
}
