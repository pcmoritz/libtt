#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)

void kernel_main() {
  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  const uint32_t tile_bytes = get_tile_size(cb_in0);

  const uint32_t in0_addr = A(0);
  const uint32_t in1_addr = A(1);
  const uint32_t bank_x = A(2);
  const uint32_t bank_y = A(3);
  const uint32_t kt = A(4);
  const uint32_t block_w = A(5);
  const uint32_t per_core_n = A(6);
  const uint32_t shard_nt = A(7);
  const uint32_t col_offset = A(8);

  const uint32_t num_blocks = kt / block_w;
  const uint32_t in1_block_tiles = block_w * per_core_n;
  const uint32_t row_bytes = per_core_n * tile_bytes;
  const uint32_t row_stride_bytes = shard_nt * tile_bytes;

  const InterleavedAddrGenFast<true> in0_gen = {
      .bank_base_address = in0_addr,
      .page_size = tile_bytes,
      .data_format = get_dataformat(cb_in0),
  };

  uint64_t in1_row_addr =
      get_noc_addr(bank_x, bank_y, in1_addr + col_offset * tile_bytes);
  uint32_t in0_tile = 0;
  for (uint32_t block = 0; block < num_blocks; block++) {
    cb_reserve_back(cb_in0, block_w);
    uint32_t l1_addr = get_write_ptr(cb_in0);
    for (uint32_t w = 0; w < block_w; w++) {
      noc_async_read_tile(in0_tile, in0_gen, l1_addr);
      in0_tile++;
      l1_addr += tile_bytes;
    }

    cb_reserve_back(cb_in1, in1_block_tiles);
    l1_addr = get_write_ptr(cb_in1);
    for (uint32_t w = 0; w < block_w; w++) {
      noc_async_read(in1_row_addr, l1_addr, row_bytes);
      in1_row_addr += row_stride_bytes;
      l1_addr += row_bytes;
    }

    noc_async_read_barrier();
    cb_push_back(cb_in0, block_w);
    cb_push_back(cb_in1, in1_block_tiles);
  }
}
