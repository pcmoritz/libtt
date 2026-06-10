#include <cstdint>

#define A(n) get_arg_val<uint32_t>(n)

void kernel_main() {
  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(cb_out);

  const uint32_t out_addr = A(0);
  const uint32_t col_offset = A(1);
  const uint32_t per_core_n = A(2);
  const uint32_t out_subblock_w = A(3);
  const uint32_t logical_nt = A(4);

  const InterleavedAddrGenFast<true> out_gen = {
      .bank_base_address = out_addr,
      .page_size = tile_bytes,
      .data_format = get_dataformat(cb_out),
  };

  const uint32_t num_subblocks = per_core_n / out_subblock_w;
  uint32_t out_tile = col_offset;
  for (uint32_t sb = 0; sb < num_subblocks; sb++) {
    cb_wait_front(cb_out, out_subblock_w);
    uint32_t l1_addr = get_read_ptr(cb_out);
    for (uint32_t w = 0; w < out_subblock_w; w++) {
      if (out_tile < logical_nt) {
        noc_async_write_tile(out_tile, out_gen, l1_addr);
      }
      out_tile++;
      l1_addr += tile_bytes;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, out_subblock_w);
  }
}
