#include <cstdint>

namespace {

constexpr uint32_t cb_input = tt::CBIndex::c_0;
constexpr uint32_t cb_pair = tt::CBIndex::c_1;
constexpr uint32_t cb_cos = tt::CBIndex::c_2;
constexpr uint32_t cb_sin = tt::CBIndex::c_3;
constexpr uint32_t TILE_R = 32;
constexpr uint32_t tiles_per_row = ROPE_TILES_PER_ROW;
constexpr uint32_t output_tile_rows = ROPE_OUTPUT_TILE_ROWS;
constexpr uint32_t half_tiles = ROPE_HALF_TILES;

void read_tile(const InterleavedAddrGenFast<true> &input, uint32_t tile_id,
               uint32_t cb) {
  cb_reserve_back(cb, 1);
  noc_async_read_tile(tile_id, input, get_write_ptr(cb));
  noc_async_read_barrier();
  cb_push_back(cb, 1);
}

}  // namespace

void kernel_main() {
  uint32_t input_addr = get_arg_val<uint32_t>(0);
  uint32_t cos_addr = get_arg_val<uint32_t>(1);
  uint32_t sin_addr = get_arg_val<uint32_t>(2);
  uint32_t tile_offset = get_arg_val<uint32_t>(3);
  uint32_t n_tiles = get_arg_val<uint32_t>(4);

  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_input),
      .data_format = get_dataformat(cb_input),
  };
  const InterleavedAddrGenFast<true> cos = {
      .bank_base_address = cos_addr,
      .page_size = get_tile_size(cb_cos),
      .data_format = get_dataformat(cb_cos),
  };
  const InterleavedAddrGenFast<true> sin = {
      .bank_base_address = sin_addr,
      .page_size = get_tile_size(cb_sin),
      .data_format = get_dataformat(cb_sin),
  };

  for (uint32_t i = 0; i < n_tiles; ++i) {
    uint32_t output_tile = tile_offset + i;
    uint32_t col_tile = output_tile % tiles_per_row;
    uint32_t row_major = output_tile / tiles_per_row;
    uint32_t batch = row_major / output_tile_rows;
    bool lower_half = col_tile < half_tiles;
    uint32_t half_col = lower_half ? col_tile : col_tile - half_tiles;
    uint32_t pair_tile = lower_half ? output_tile + half_tiles
                                    : output_tile - half_tiles;
    uint32_t trig_tile = (batch / TILE_R) * half_tiles + half_col;

    read_tile(input, output_tile, cb_input);
    read_tile(input, pair_tile, cb_pair);
    read_tile(cos, trig_tile, cb_cos);
    read_tile(sin, trig_tile, cb_sin);
  }
}
