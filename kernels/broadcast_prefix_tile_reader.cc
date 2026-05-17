#include <cstdint>

namespace {

constexpr uint32_t TILE_R = 32;
constexpr uint32_t TILE_C = 32;
constexpr uint32_t INPUT_RANK = BROADCAST_INPUT_RANK;
constexpr uint32_t OUTPUT_RANK = BROADCAST_OUTPUT_RANK;
constexpr uint32_t INPUT_COORD_COUNT = INPUT_RANK == 0 ? 1 : INPUT_RANK;
constexpr uint32_t OUTPUT_COORD_COUNT = OUTPUT_RANK == 0 ? 1 : OUTPUT_RANK;
constexpr uint32_t INPUT_SHAPE[INPUT_COORD_COUNT] = BROADCAST_INPUT_SHAPE;
constexpr uint32_t OUTPUT_SHAPE[OUTPUT_COORD_COUNT] = BROADCAST_OUTPUT_SHAPE;
constexpr uint32_t BROADCAST_DIMS[INPUT_COORD_COUNT] = BROADCAST_DIMENSIONS;
constexpr uint32_t INPUT_TILE_ROWS = BROADCAST_INPUT_TILE_ROWS;
constexpr uint32_t INPUT_TILES_PER_ROW = BROADCAST_INPUT_TILES_PER_ROW;
constexpr uint32_t OUTPUT_TILE_ROWS = BROADCAST_OUTPUT_TILE_ROWS;
constexpr uint32_t OUTPUT_TILES_PER_ROW = BROADCAST_OUTPUT_TILES_PER_ROW;

void decode_output_batch(uint32_t output_batch, uint32_t output_coords[OUTPUT_COORD_COUNT]) {
  for (uint32_t dim = 0; dim < OUTPUT_RANK; ++dim) {
    output_coords[dim] = 0;
  }
  if constexpr (OUTPUT_RANK >= 3) {
    for (uint32_t index = 0; index < OUTPUT_RANK - 2; ++index) {
      uint32_t dim = OUTPUT_RANK - 3 - index;
      output_coords[dim] = output_batch % OUTPUT_SHAPE[dim];
      output_batch /= OUTPUT_SHAPE[dim];
    }
  }
}

uint32_t input_prefix_from_output(const uint32_t output_coords[OUTPUT_COORD_COUNT]) {
  uint32_t input_prefix = 0;
  if constexpr (INPUT_RANK >= 3) {
    for (uint32_t dim = 0; dim < INPUT_RANK - 2; ++dim) {
      uint32_t coord = INPUT_SHAPE[dim] == 1 ? 0 : output_coords[BROADCAST_DIMS[dim]];
      input_prefix = input_prefix * INPUT_SHAPE[dim] + coord;
    }
  }
  return input_prefix;
}

}  // namespace

void kernel_main() {
  const uint32_t input_addr = get_arg_val<uint32_t>(0);
  const uint32_t output_tile_offset = get_arg_val<uint32_t>(1);
  const uint32_t output_tile_count = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_output = tt::CBIndex::c_16;
  const InterleavedAddrGenFast<true> input = {
      .bank_base_address = input_addr,
      .page_size = get_tile_size(cb_output),
      .data_format = get_dataformat(cb_output),
  };

  constexpr uint32_t output_matrix_tiles = OUTPUT_TILE_ROWS * OUTPUT_TILES_PER_ROW;
  uint32_t output_coords[OUTPUT_COORD_COUNT];
  for (uint32_t tile = 0; tile < output_tile_count; ++tile) {
    const uint32_t output_tile_id = output_tile_offset + tile;
    const uint32_t output_batch = output_tile_id / output_matrix_tiles;
    const uint32_t output_matrix_tile = output_tile_id % output_matrix_tiles;
    const uint32_t output_tile_row = output_matrix_tile / OUTPUT_TILES_PER_ROW;
    const uint32_t output_tile_col = output_matrix_tile % OUTPUT_TILES_PER_ROW;
    decode_output_batch(output_batch, output_coords);
    const uint32_t input_prefix = input_prefix_from_output(output_coords);
    const uint32_t input_tile =
        (input_prefix * INPUT_TILE_ROWS + output_tile_row) * INPUT_TILES_PER_ROW + output_tile_col;

    cb_reserve_back(cb_output, 1);
    noc_async_read_tile(input_tile, input, get_write_ptr(cb_output));
    noc_async_read_barrier();
    cb_push_back(cb_output, 1);
  }
}
