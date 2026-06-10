//! Width-sharded decode matmul: the RHS weight matrix is stored as one
//! contiguous shard of tile columns per DRAM bank, and each bank is read only
//! by a few worker cores placed next to it, so reads are long sequential runs
//! within a single bank instead of one interleaved 2 KB page per request (see
//! tt-metal's "Saturating DRAM bandwidth" tech report).

use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, MathFidelity, Program};
use crate::dram::{DType, DramBuffer};
use crate::hw::{CoreCoord, Dram, TensixL1};
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::env;
use std::io;

const MATMUL_WS_READER: &str = include_str!("../../kernels/matmul_ws_reader.cc");
const MATMUL_WS_WRITER: &str = include_str!("../../kernels/matmul_ws_writer.cc");

const CORES_PER_BANK: usize = 4;
const WEST_COLS: [u8; 2] = [1, 2];
const EAST_COLS: [u8; 2] = [10, 11];
const MAX_BLOCK_W: usize = 8;
const L1_BUDGET_BYTES: usize = TensixL1::SIZE as usize - TensixL1::DATA_BUFFER_SPACE_BASE as usize;

pub(crate) fn width_sharding_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env::var("LIBTT_MATMUL_WIDTH_SHARD").as_deref() != Ok("0"))
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct MatmulWsKey {
    pub(crate) kt: usize,
    pub(crate) logical_nt: usize,
    pub(crate) dtype: DType,
    pub(crate) math_fidelity: MathFidelity,
}

#[derive(Clone, Debug)]
pub(crate) struct MatmulWsPlan {
    pub(crate) kt: usize,
    pub(crate) logical_nt: usize,
    pub(crate) shard_nt: usize,
    pub(crate) per_core_n: usize,
    pub(crate) in0_block_w: usize,
    pub(crate) out_subblock_w: usize,
    /// (worker core, bank NOC xy, column offset inside the bank shard).
    pub(crate) cores: Vec<(CoreCoord, CoreCoord, usize)>,
}

impl MatmulWsPlan {
    pub(crate) fn padded_nt(&self) -> usize {
        self.shard_nt * Dram::BANK_COUNT
    }
}

fn divisors_le(n: usize, max: usize) -> Vec<usize> {
    (1..=max.min(n)).rev().filter(|d| n % d == 0).collect()
}

fn cb_bytes(bw: usize, pcn: usize, tile_bytes: usize) -> usize {
    (2 * bw + 2 * bw * pcn + pcn + pcn) * tile_bytes
}

pub(crate) fn plan_width_sharded(device: &Device, key: &MatmulWsKey) -> Option<MatmulWsPlan> {
    if device.active_dram_banks != Dram::BANK_COUNT {
        return None;
    }
    let tile_bytes = key.dtype.tile_size();
    // Round so out_subblock_w can be 4 even for awkward Nt (lm_head Nt is prime).
    let per_core_n = key
        .logical_nt
        .div_ceil(Dram::BANK_COUNT * CORES_PER_BANK)
        .next_multiple_of(4);
    let shard_nt = per_core_n * CORES_PER_BANK;
    let in0_block_w = divisors_le(key.kt, MAX_BLOCK_W)
        .into_iter()
        .find(|&bw| cb_bytes(bw, per_core_n, tile_bytes) <= L1_BUDGET_BYTES)?;
    let out_subblock_w = divisors_le(per_core_n, 8).into_iter().next()?;

    let workers: std::collections::BTreeSet<CoreCoord> =
        device.cores_ref().iter().copied().collect();
    let mut cores = Vec::with_capacity(Dram::BANK_COUNT * CORES_PER_BANK);
    for bank in 0..Dram::BANK_COUNT {
        let cols = if bank < 4 { WEST_COLS } else { EAST_COLS };
        let bank_tiles: Vec<&crate::hw::DramTile> = device
            .dram_tiles
            .iter()
            .filter(|tile| tile.bank == bank)
            .collect();
        if bank_tiles.is_empty() {
            return None;
        }
        for i in 0..CORES_PER_BANK {
            let core = CoreCoord {
                x: cols[i % 2],
                y: 2 + ((bank % 4) as u8) * 2 + (i / 2) as u8,
            };
            if !workers.contains(&core) {
                return None;
            }
            let tile = bank_tiles[i % bank_tiles.len()];
            cores.push((
                core,
                CoreCoord {
                    x: tile.x,
                    y: tile.y,
                },
                i * per_core_n,
            ));
        }
    }

    Some(MatmulWsPlan {
        kt: key.kt,
        logical_nt: key.logical_nt,
        shard_nt,
        per_core_n,
        in0_block_w,
        out_subblock_w,
        cores,
    })
}

/// Reorders canonical row-major tile pages (kt x logical_nt) so a standard
/// interleaved write lands each bank's shard contiguously in that bank: shard
/// slot s of bank b ends up at interleaved page s * BANK_COUNT + b.
pub(crate) fn shard_pages(data: &[u8], plan: &MatmulWsPlan, tile_bytes: usize) -> Vec<u8> {
    let padded_nt = plan.padded_nt();
    let mut out = vec![0u8; plan.kt * padded_nt * tile_bytes];
    for k in 0..plan.kt {
        for n in 0..plan.logical_nt {
            let bank = n / plan.shard_nt;
            let slot = k * plan.shard_nt + (n % plan.shard_nt);
            let dst = (slot * Dram::BANK_COUNT + bank) * tile_bytes;
            let src = (k * plan.logical_nt + n) * tile_bytes;
            out[dst..dst + tile_bytes].copy_from_slice(&data[src..src + tile_bytes]);
        }
    }
    out
}

pub(crate) fn sharded_rhs_for(
    device: &mut Device,
    rhs: &DramBuffer,
    plan: &MatmulWsPlan,
) -> io::Result<DramBuffer> {
    if let Some((source, buf)) = device.width_sharded_rhs.get(&rhs.addr) {
        if source == rhs {
            return Ok(buf.clone());
        }
    }
    let data = device.dram_read_raw(rhs)?;
    let tile_bytes = rhs.dtype.tile_size();
    let pages = shard_pages(&data, plan, tile_bytes);
    let num_tiles = plan.kt * plan.padded_nt();
    let buf = device.alloc(
        num_tiles,
        rhs.dtype,
        &[plan.kt * 32, plan.padded_nt() * 32],
        format!("{}_ws", rhs.name),
    )?;
    device.dram_write_raw(&buf, &pages)?;
    device
        .width_sharded_rhs
        .insert(rhs.addr, (rhs.clone(), buf.clone()));
    Ok(buf)
}

pub(crate) struct MatmulWsKernel {
    pub(crate) lhs_addr: u32,
    pub(crate) rhs_addr: u32,
    pub(crate) output_addr: u32,
    pub(crate) key: MatmulWsKey,
    pub(crate) plan: MatmulWsPlan,
}

impl Kernel<MatmulWsKey> for MatmulWsKernel {
    fn program_key(&self) -> MatmulWsKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        let plan = &self.plan;
        let num_blocks = plan.kt / plan.in0_block_w;
        let mut runtime_args = RuntimeArgsBuilder::new(0, vec![0], vec![0, 1], Vec::new());
        for (index, &(core, bank, col_offset)) in plan.cores.iter().enumerate() {
            let reader = vec![
                0,
                0,
                bank.x as u32,
                bank.y as u32,
                plan.kt as u32,
                plan.in0_block_w as u32,
                plan.per_core_n as u32,
                plan.shard_nt as u32,
                col_offset as u32,
            ];
            let global_col = (index / CORES_PER_BANK) * plan.shard_nt + col_offset;
            let writer = vec![
                0,
                global_col as u32,
                plan.per_core_n as u32,
                plan.out_subblock_w as u32,
                plan.logical_nt as u32,
            ];
            runtime_args.add_core(core, writer, reader, Vec::new())?;
        }

        let cbs = vec![
            CBConfig::new(0, self.key.dtype).with_tiles(2 * plan.in0_block_w),
            CBConfig::new(1, self.key.dtype).with_tiles(2 * plan.in0_block_w * plan.per_core_n),
            CBConfig::new(16, self.key.dtype).with_tiles(plan.per_core_n),
            CBConfig::new(24, self.key.dtype).with_tiles(plan.per_core_n),
        ];
        let compute = compute_src(plan, num_blocks);
        Ok(Program {
            reader_kernel: MATMUL_WS_READER.to_owned(),
            writer_kernel: MATMUL_WS_WRITER.to_owned(),
            compute_kernel: compute,
            name: format!(
                "matmul_ws_{:?}_{}x{}",
                self.key.dtype,
                plan.kt * 32,
                plan.logical_nt * 32
            ),
            compile: CompileConfig {
                cbs,
                math_fidelity: self.key.math_fidelity,
                ..CompileConfig::default()
            },
            grid: None,
            ..Program::new(runtime_args.build()?)
        })
    }

    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            0 => Some(self.lhs_addr),
            1 => Some(self.rhs_addr),
            _ => None,
        }
    }

    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        (index == 0).then_some(self.output_addr)
    }
}

fn compute_src(plan: &MatmulWsPlan, num_blocks: usize) -> String {
    let replacements = [
        ("@BATCH_COUNT@", 1usize),
        ("@IN1_TRANSPOSE@", 0),
        ("@IN0_BLOCK_W@", plan.in0_block_w),
        ("@IN0_NUM_SUBBLOCKS@", 1),
        ("@IN0_BLOCK_NUM_TILES@", plan.in0_block_w),
        ("@IN0_SUBBLOCK_NUM_TILES@", plan.in0_block_w),
        ("@IN1_NUM_SUBBLOCKS@", plan.per_core_n / plan.out_subblock_w),
        ("@IN1_BLOCK_NUM_TILES@", plan.in0_block_w * plan.per_core_n),
        ("@IN1_PER_CORE_W@", plan.per_core_n),
        ("@NUM_BLOCKS@", num_blocks),
        ("@OUT_SUBBLOCK_H@", 1),
        ("@OUT_SUBBLOCK_W@", plan.out_subblock_w),
        ("@OUT_SUBBLOCK_NUM_TILES@", plan.out_subblock_w),
        ("@OUT_BLOCK_NUM_TILES@", plan.per_core_n),
    ];
    let mut src = super::matmul::MATMUL_COMPUTE_TEMPLATE.to_owned();
    for (token, value) in replacements {
        src = src.replace(token, &value.to_string());
    }
    src
}
