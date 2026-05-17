#include <cstdint>
#define PACKER_L1_ACC 1
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
  constexpr uint32_t in0_block_w = @IN0_BLOCK_W@;
  constexpr uint32_t in0_num_subblocks = @IN0_NUM_SUBBLOCKS@;
  constexpr uint32_t in0_block_num_tiles = @IN0_BLOCK_NUM_TILES@;
  constexpr uint32_t in0_subblock_num_tiles = @IN0_SUBBLOCK_NUM_TILES@;
  constexpr uint32_t in1_num_subblocks = @IN1_NUM_SUBBLOCKS@;
  constexpr uint32_t in1_block_num_tiles = @IN1_BLOCK_NUM_TILES@;
  constexpr uint32_t in1_per_core_w = @IN1_PER_CORE_W@;
  constexpr uint32_t num_blocks = @NUM_BLOCKS@;
  constexpr uint32_t out_subblock_h = @OUT_SUBBLOCK_H@;
  constexpr uint32_t out_subblock_w = @OUT_SUBBLOCK_W@;
  constexpr uint32_t out_subblock_num_tiles = @OUT_SUBBLOCK_NUM_TILES@;
  constexpr uint32_t out_block_num_tiles = @OUT_BLOCK_NUM_TILES@;
  constexpr uint32_t batch_count = @BATCH_COUNT@;
  constexpr uint32_t transpose = @TRANSPOSE_B@;

  mm_block_init(
      tt::CBIndex::c_0,
      tt::CBIndex::c_1,
      tt::CBIndex::c_16,
      transpose,
      out_subblock_w,
      out_subblock_h,
      in0_block_w);

  for (uint32_t batch = 0; batch < batch_count; batch++) {
    bool enable_reload = false;
    uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
    for (uint32_t block = 0; block < num_blocks; block++) {
      const bool last_out = (block == (num_blocks - 1));
      cb_wait_front(tt::CBIndex::c_0, in0_block_num_tiles);
      cb_wait_front(tt::CBIndex::c_1, in1_block_num_tiles);
      int in0_index_subblock_offset = 0;
      for (uint32_t in0_sb = 0; in0_sb < in0_num_subblocks; in0_sb++) {
        int in1_index_subblock_offset = 0;
        for (uint32_t in1_sb = 0; in1_sb < in1_num_subblocks; in1_sb++) {
          tile_regs_acquire();
          if (enable_reload) {
            copy_tile_to_dst_init_short(tt::CBIndex::c_24);
            cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
#pragma GCC unroll 0
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
              copy_tile(tt::CBIndex::c_24, i, i);
            }
            cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
            mm_block_init_short(
                tt::CBIndex::c_0,
                tt::CBIndex::c_1,
                transpose,
                out_subblock_w,
                out_subblock_h,
                in0_block_w);
          }
#pragma GCC unroll 0
          for (uint32_t inner = 0; inner < in0_block_w; inner++) {
            const uint32_t in0_tile_index = (uint32_t)(in0_index_subblock_offset + (int)inner);
            const uint32_t in1_tile_index =
                (uint32_t)(in1_index_subblock_offset + (int)(inner * in1_per_core_w));
            matmul_block(
                tt::CBIndex::c_0,
                tt::CBIndex::c_1,
                in0_tile_index,
                in1_tile_index,
                0,
                transpose,
                out_subblock_w,
                out_subblock_h,
                in0_block_w);
          }
          tile_regs_commit();
          if (last_out) {
            cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
            tile_regs_wait();
            PACK((llk_pack_reconfig_l1_acc(0)));
#pragma GCC unroll 0
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
              pack_tile(i, tt::CBIndex::c_16);
            }
            tile_regs_release();
            cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
          } else {
            if (block == 0) {
              cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
              out_num_tiles_to_wait += out_subblock_num_tiles;
            }
            cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
            tile_regs_wait();
            if (block == 0) {
              PACK((llk_pack_reconfig_l1_acc(0)));
            } else if (block == 1) {
              PACK((llk_pack_reconfig_l1_acc(1)));
            }
#pragma GCC unroll 0
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
              pack_tile(i, tt::CBIndex::c_24);
            }
            tile_regs_release();
            cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
          }
          in1_index_subblock_offset += out_subblock_w;
        }
        in0_index_subblock_offset += in0_subblock_num_tiles;
      }
      if (num_blocks > 2 && block < num_blocks - 2) {
        cb_wait_front(tt::CBIndex::c_24, out_block_num_tiles);
        cb_pop_front(tt::CBIndex::c_24, out_block_num_tiles);
      }
      if (num_blocks >= 2 && block == num_blocks - 2) {
        enable_reload = true;
      }
      cb_pop_front(tt::CBIndex::c_0, in0_block_num_tiles);
      cb_pop_front(tt::CBIndex::c_1, in1_block_num_tiles);
    }
  }
}
}  // namespace NAMESPACE
