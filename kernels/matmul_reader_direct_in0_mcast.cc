#include <cstdint>

namespace {
constexpr uint32_t ARG_VIEW_KIND = 28;

void mcast_block(uint32_t start_addr, uint32_t block_bytes, uint32_t rect_arg,
                 uint32_t recv_sem_arg, bool loopback_src) {
  const uint32_t count = A(rect_arg + 4);
  if (count == 0) {
    return;
  }
  const uint64_t data_addr = get_noc_multicast_addr(
      A(rect_arg), A(rect_arg + 1), A(rect_arg + 2), A(rect_arg + 3), start_addr);
  const uint64_t sem_addr = get_noc_multicast_addr(
      A(rect_arg),
      A(rect_arg + 1),
      A(rect_arg + 2),
      A(rect_arg + 3),
      get_semaphore(A(recv_sem_arg)));
  if (loopback_src) {
    noc_async_write_multicast_loopback_src(start_addr, data_addr, block_bytes, count, true);
    noc_semaphore_set_multicast_loopback_src(get_semaphore(A(recv_sem_arg)), sem_addr, count);
  } else {
    noc_async_write_multicast(start_addr, data_addr, block_bytes, count, true);
    noc_semaphore_set_multicast(get_semaphore(A(recv_sem_arg)), sem_addr, count);
  }
}
}  // namespace

void kernel_main() {
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t cb_source = tt::CBIndex::c_2;
  const uint32_t tile_bytes = get_tile_size(cb_in0);
  const uint32_t block_w = A(5);
  const uint32_t block_h = A(6);
  const uint32_t block_tiles = A(7);
  const uint32_t nblocks = A(8);
  const uint32_t logical_mt = A(23);
  const uint32_t local_batch_count = A(24);
  const uint32_t batch_start = A(25);
  const uint32_t total_batch_count = A(26);
  const uint32_t batch_stride = A(27);
  const View view = load_view(ARG_VIEW_KIND);
  volatile tt_l1_ptr uint32_t *sender_sem = SEM(21);
  volatile tt_l1_ptr uint32_t *recv_sem = SEM(22);
  const bool is_sender = my_x[0] == A(19) && my_y[0] == A(20);
  const uint32_t receiver_count = A(13) + A(18) - 1;

  const InterleavedAddrGenFast<true> in0_gen = {
      .bank_base_address = A(0),
      .page_size = tile_bytes,
      .data_format = get_dataformat(cb_in0),
  };
  for (uint32_t local_batch = 0; local_batch < local_batch_count; local_batch++) {
    const uint32_t batch = batch_start + local_batch;
    const bool valid_batch = batch < total_batch_count;
    uint32_t cur_block = A(1) + batch * batch_stride;
    for (uint32_t block = 0; block < nblocks; block++) {
      cb_reserve_back(cb_in0, block_tiles);
      uint32_t l1_addr = get_write_ptr(cb_in0);
      uint32_t start_addr = l1_addr;
      uint32_t block_bytes = block_tiles * tile_bytes;

      if (is_sender) {
        uint32_t row = cur_block;
        uint32_t row_tile = (row - batch * batch_stride) / A(3);
        if (!valid_batch) {
          for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            zero_tile_at(l1_addr, tile_bytes);
            l1_addr += tile_bytes;
          }
        } else if (view.kind == VIEW_CONTIGUOUS) {
          for (uint32_t h = 0; h < block_h; h++) {
            uint32_t tile_id = row;
            for (uint32_t w = 0; w < block_w; w++) {
              if (row_tile < logical_mt) {
                noc_async_read_tile(tile_id, in0_gen, l1_addr);
              }
              l1_addr += tile_bytes;
              tile_id += A(2);
            }
            row += A(3);
            row_tile++;
          }
          noc_async_read_barrier();
        } else if (view.kind == VIEW_TILE_TRANSPOSE) {
          uint32_t canonical_base = cur_block - batch * batch_stride;
          fill_tile_transpose_block(
              in0_gen,
              view,
              batch,
              canonical_base,
              A(3),
              block_h,
              block_w,
              l1_addr,
              tile_bytes,
              cb_source);
        } else {
          uint32_t canonical_base = cur_block - batch * batch_stride;
          for (uint32_t h = 0; h < block_h; h++) {
            for (uint32_t w = 0; w < block_w; w++) {
              uint32_t canonical_tile = canonical_base + h * A(3) + w;
              uint32_t canonical_row_tile = canonical_tile / A(3);
              uint32_t canonical_col_tile =
                  canonical_tile - canonical_row_tile * A(3);
              if (view.kind == VIEW_TILED_INDEX_MAP) {
                fill_tiled_index_map_tile(
                    in0_gen,
                    view,
                    batch,
                    canonical_row_tile,
                    canonical_col_tile,
                    l1_addr,
                    tile_bytes,
                    cb_source);
              } else {
              fill_generic_tile(
                  in0_gen,
                  view,
                  batch,
                    canonical_row_tile,
                    canonical_col_tile,
                    l1_addr,
                    tile_bytes,
                    cb_source);
              }
              l1_addr += tile_bytes;
            }
          }
        }

        noc_semaphore_wait(sender_sem, receiver_count);
        noc_semaphore_set(sender_sem, 0);
        *recv_sem = VALID;
        mcast_block(start_addr, block_bytes, 9, 22, true);
        mcast_block(start_addr, block_bytes, 14, 22, false);
      } else {
        noc_semaphore_set(recv_sem, INVALID);
        noc_semaphore_inc(get_noc_addr(A(19), A(20), get_semaphore(A(21))), 1);
        noc_semaphore_wait(recv_sem, VALID);
      }

      cur_block += A(4);
      cb_push_back(cb_in0, block_tiles);
    }
  }
}
