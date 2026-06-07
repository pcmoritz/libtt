#include <cstdint>

namespace {

constexpr uint32_t TILE_C = 32;
constexpr uint32_t BF16_ONE_BITS = 0x3f80u;

void fill_bf16_tile(uint32_t cb, uint32_t value_bits) {
  volatile tt_l1_ptr uint16_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint16_t *>(get_write_ptr(cb));
  uint16_t value = static_cast<uint16_t>(value_bits);
  uint32_t elements = get_tile_size(cb) / sizeof(uint16_t);
  for (uint32_t i = 0; i < elements; ++i) {
    ptr[i] = value;
  }
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t weight_addr = get_arg_val<uint32_t>(1);
  uint32_t group_offset = get_arg_val<uint32_t>(2);
  uint32_t group_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb_input = tt::CBIndex::c_0;
  constexpr uint32_t cb_weight = tt::CBIndex::c_1;
  constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
  constexpr uint32_t width_tiles = RMS_NORM_WIDTH_TILES;

  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> weight = {
      .bank_base_address = weight_addr,
      .page_size = get_tile_size(cb_weight),
      .data_format = get_dataformat(cb_weight),
  };

  cb_reserve_back(cb_scaler, 1);
  fill_bf16_tile(cb_scaler, BF16_ONE_BITS);
  cb_push_back(cb_scaler, 1);

  for (uint32_t group = 0; group < group_count; ++group) {
    uint32_t base_tile = (group_offset + group) * width_tiles;
    uint32_t input_tile_bytes = get_tile_size(cb_input);
    uint32_t weight_tile_bytes = get_tile_size(cb_weight);

    cb_reserve_back(cb_input, width_tiles);
    uint32_t input_write_ptr = get_write_ptr(cb_input);
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      noc_async_read_tile(base_tile + wt, input,
                          input_write_ptr + wt * input_tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_input, width_tiles);

    cb_reserve_back(cb_weight, width_tiles);
    uint32_t weight_write_ptr = get_write_ptr(cb_weight);
    for (uint32_t wt = 0; wt < width_tiles; ++wt) {
      noc_async_read_tile(wt, weight, weight_write_ptr + wt * weight_tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_weight, width_tiles);
  }
}
