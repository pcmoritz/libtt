use crate::device::Device;
use crate::dispatch::{CBConfig, CoreSelection, MathFidelity, Program};
use crate::dram::{DType, DramBuffer};
use crate::hw::{CoreCoord, TensixL1};
use crate::log::log;
use std::collections::HashMap;
use std::io;
use std::sync::{Mutex, OnceLock};

const BF16_READER_SENDER: &str = include_str!("../../kernels/matmul_reader_sender.cc");
const BF16_READER_RECV: &str = include_str!("../../kernels/matmul_reader_recv.cc");
const BF16_WRITER_SENDER: &str = include_str!("../../kernels/matmul_writer_sender.cc");
const BF16_WRITER_RECV: &str = include_str!("../../kernels/matmul_writer_recv.cc");
const BF16_COMPUTE_TEMPLATE: &str = include_str!("../../kernels/matmul_compute.cc");
const NUM_SEMAPHORES: usize = 4;
const SPLIT_N_STEP_TILES: usize = 8;

static ZERO_TILE_BY_DEVICE: OnceLock<Mutex<HashMap<usize, DramBuffer>>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq)]
struct MatmulPlan {
    rows: Vec<u8>,
    cols: Vec<u8>,
    mt: usize,
    kt: usize,
    nt: usize,
    per_core_m: usize,
    per_core_n: usize,
    in0_block_w: usize,
    out_subblock_h: usize,
    out_subblock_w: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MatmulChunkPlan {
    plan: MatmulPlan,
    col_offset_tiles: usize,
    logical_nt: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SplitMatmulPlan {
    west: MatmulChunkPlan,
    east: MatmulChunkPlan,
}

impl MatmulPlan {
    fn out_subblock_num_tiles(&self) -> usize {
        self.out_subblock_h * self.out_subblock_w
    }

    fn num_blocks(&self) -> usize {
        self.kt / self.in0_block_w
    }

    fn in0_num_subblocks(&self) -> usize {
        self.per_core_m / self.out_subblock_h
    }

    fn in1_num_subblocks(&self) -> usize {
        self.per_core_n / self.out_subblock_w
    }

    fn in0_block_num_tiles(&self) -> usize {
        self.per_core_m * self.in0_block_w
    }

    fn in0_subblock_num_tiles(&self) -> usize {
        self.out_subblock_h * self.in0_block_w
    }

    fn in1_block_num_tiles(&self) -> usize {
        self.per_core_n * self.in0_block_w
    }

    fn out_block_num_tiles(&self) -> usize {
        self.per_core_m * self.per_core_n
    }

    fn cb0_pages(&self) -> usize {
        2 * self.per_core_m * self.in0_block_w
    }

    fn cb1_pages(&self) -> usize {
        2 * self.per_core_n * self.in0_block_w
    }
}

pub(crate) fn matmul_bf16(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if lhs.dtype != DType::Float16B || rhs.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "matmul_bf16 requires bf16 inputs, got {:?} and {:?}",
            lhs.dtype, rhs.dtype
        )));
    }

    let (m, k) = shape_2d(lhs, "lhs")?;
    let (rhs_k, n) = shape_2d(rhs, "rhs")?;
    if k != rhs_k {
        return Err(invalid_input(format!(
            "matmul inner dimensions must match, got {k} and {rhs_k}"
        )));
    }
    if m % 32 != 0 || k % 32 != 0 || n % 32 != 0 {
        return Err(invalid_input(format!(
            "matmul_bf16 requires 32x32 tiled shapes, got lhs={:?} rhs={:?}",
            lhs.shape, rhs.shape
        )));
    }
    validate_tile_count(lhs, m / 32 * k / 32, "lhs")?;
    validate_tile_count(rhs, k / 32 * n / 32, "rhs")?;

    let output_name = name.into();
    let logical_mt = m / 32;
    let logical_nt = n / 32;
    let cores = device.cores();
    let plan = plan_matmul(m, k, n, &cores)?;
    log(format!(
        "matmul_bf16 plan: Mt={} Kt={} Nt={} grid={}x{} per_core_M={} per_core_N={} in0_block_w={} num_blocks={} subblock={}x{}",
        plan.mt,
        plan.kt,
        plan.nt,
        plan.rows.len(),
        plan.cols.len(),
        plan.per_core_m,
        plan.per_core_n,
        plan.in0_block_w,
        plan.num_blocks(),
        plan.out_subblock_h,
        plan.out_subblock_w
    ));

    let zero_tile = cached_zero_tile(device)?;
    let output = device.alloc(
        logical_mt * logical_nt,
        DType::Float16B,
        Some(&[m, n]),
        output_name,
    )?;
    let program = bf16_program(&plan, lhs, rhs, &output, &zero_tile, logical_mt, logical_nt)?;
    device.run_program(&program)?;
    Ok(output)
}

fn cached_zero_tile(device: &mut Device) -> io::Result<DramBuffer> {
    let cache = ZERO_TILE_BY_DEVICE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(buffer) = cache
        .lock()
        .map_err(|_| io::Error::other("zero tile cache is poisoned"))?
        .get(&device.local_hardware_id())
        .cloned()
    {
        return Ok(buffer);
    }

    let buffer = device.alloc_write(
        &vec![0u8; DType::Float16B.tile_size()],
        DType::Float16B,
        &[32, 32],
        "matmul_zero_tile",
    )?;
    cache
        .lock()
        .map_err(|_| io::Error::other("zero tile cache is poisoned"))?
        .insert(device.local_hardware_id(), buffer.clone());
    Ok(buffer)
}

fn shape_2d(buffer: &DramBuffer, name: &str) -> io::Result<(usize, usize)> {
    let shape = buffer
        .shape
        .as_ref()
        .ok_or_else(|| invalid_input(format!("{name} buffer is missing shape metadata")))?;
    if shape.len() != 2 {
        return Err(invalid_input(format!(
            "matmul_bf16 requires rank-2 {name} input, got shape {shape:?}"
        )));
    }
    Ok((shape[0], shape[1]))
}

fn validate_tile_count(buffer: &DramBuffer, expected: usize, name: &str) -> io::Result<()> {
    if buffer.num_tiles != expected {
        return Err(invalid_input(format!(
            "{name} tile count mismatch: got {}, expected {expected}",
            buffer.num_tiles
        )));
    }
    Ok(())
}

fn plan_matmul(m: usize, k: usize, n: usize, cores: &[CoreCoord]) -> io::Result<MatmulPlan> {
    let plan = plan_matmul_for_cores(m, k, n, cores)?;
    if crosses_column_gap(&plan) {
        let west_cores = cores
            .iter()
            .copied()
            .filter(|core| core.x < 8)
            .collect::<Vec<_>>();
        if let Ok(west_plan) = plan_matmul_for_cores_with_min_subblock(m, k, n, &west_cores, 2, 2)
            .or_else(|_| plan_matmul_for_cores(m, k, n, &west_cores))
        {
            log(format!(
                "matmul_bf16 avoiding cross-gap multicast: full grid={}x{} fallback grid={}x{}",
                plan.rows.len(),
                plan.cols.len(),
                west_plan.rows.len(),
                west_plan.cols.len()
            ));
            return Ok(west_plan);
        }
    }
    Ok(plan)
}

fn plan_split_matmul(
    m: usize,
    k: usize,
    n: usize,
    cores: &[CoreCoord],
) -> io::Result<Option<SplitMatmulPlan>> {
    let full_plan = plan_matmul_for_cores(m, k, n, cores)?;
    if !crosses_column_gap(&full_plan) {
        return Ok(None);
    }

    let logical_nt = n / 32;
    if logical_nt < 2 * SPLIT_N_STEP_TILES {
        return Ok(None);
    }

    let west_cores = cores
        .iter()
        .copied()
        .filter(|core| core.x < 7)
        .collect::<Vec<_>>();
    let east_cores = cores
        .iter()
        .copied()
        .filter(|core| core.x >= 10)
        .collect::<Vec<_>>();
    if west_cores.is_empty() || east_cores.is_empty() {
        return Ok(None);
    }

    let mut best = None;
    let mut best_score = f64::INFINITY;
    for west_nt in (SPLIT_N_STEP_TILES..logical_nt).step_by(SPLIT_N_STEP_TILES) {
        let east_nt = logical_nt - west_nt;
        if east_nt < SPLIT_N_STEP_TILES {
            continue;
        }
        let Ok(west_plan) =
            plan_matmul_for_cores_with_limits(m, k, west_nt * 32, &west_cores, 2, 2, 4)
        else {
            continue;
        };
        let Ok(east_plan) =
            plan_matmul_for_cores_with_limits(m, k, east_nt * 32, &east_cores, 2, 2, 4)
        else {
            continue;
        };
        if crosses_column_gap(&west_plan) || crosses_column_gap(&east_plan) {
            continue;
        }

        let score = chunk_time_score(west_nt, &west_plan) + chunk_time_score(east_nt, &east_plan);
        if score < best_score {
            best_score = score;
            best = Some(SplitMatmulPlan {
                west: MatmulChunkPlan {
                    plan: west_plan,
                    col_offset_tiles: 0,
                    logical_nt: west_nt,
                },
                east: MatmulChunkPlan {
                    plan: east_plan,
                    col_offset_tiles: west_nt,
                    logical_nt: east_nt,
                },
            });
        }
    }

    Ok(best)
}

fn chunk_time_score(logical_nt: usize, plan: &MatmulPlan) -> f64 {
    let active_cores = plan.rows.len() * plan.cols.len();
    let out_tiles = plan.per_core_m * plan.per_core_n;
    let bias = out_tiles.min(16);
    let throughput = active_cores * plan.in0_block_w * bias * bias;
    logical_nt as f64 / throughput as f64
}

fn plan_matmul_for_cores(
    m: usize,
    k: usize,
    n: usize,
    cores: &[CoreCoord],
) -> io::Result<MatmulPlan> {
    plan_matmul_for_cores_with_min_subblock(m, k, n, cores, 1, 1)
}

fn plan_matmul_for_cores_with_min_subblock(
    m: usize,
    k: usize,
    n: usize,
    cores: &[CoreCoord],
    min_out_subblock_num_tiles: usize,
    min_out_subblock_w: usize,
) -> io::Result<MatmulPlan> {
    plan_matmul_for_cores_with_limits(
        m,
        k,
        n,
        cores,
        min_out_subblock_num_tiles,
        min_out_subblock_w,
        usize::MAX,
    )
}

fn plan_matmul_for_cores_with_limits(
    m: usize,
    k: usize,
    n: usize,
    cores: &[CoreCoord],
    min_out_subblock_num_tiles: usize,
    min_out_subblock_w: usize,
    max_in0_block_w: usize,
) -> io::Result<MatmulPlan> {
    let mt_base = m / 32;
    let kt = k / 32;
    let nt_base = n / 32;
    let tile_bytes = DType::Float16B.tile_size();
    let l1_data_bytes = TensixL1::SIZE as usize - TensixL1::DATA_BUFFER_SPACE_BASE as usize;

    let mut ordered = cores.to_vec();
    ordered.sort_unstable();
    ordered.dedup();
    if ordered.is_empty() {
        return Err(invalid_input("no worker cores are available"));
    }

    let mut xs = ordered.iter().map(|core| core.x).collect::<Vec<_>>();
    xs.sort_unstable();
    xs.dedup();
    let mut ys = ordered.iter().map(|core| core.y).collect::<Vec<_>>();
    ys.sort_unstable();
    ys.dedup();

    let available = ordered.iter().copied().collect::<Vec<_>>();
    let kt_divs = divisors(kt);
    let mut best = None;
    let mut best_score = None;

    for y_start in 0..ys.len() {
        for y_stop in y_start + 1..=ys.len() {
            let rows = &ys[y_start..y_stop];
            let valid_cols = xs
                .iter()
                .copied()
                .filter(|&x| {
                    rows.iter()
                        .all(|&y| available.contains(&CoreCoord { x, y }))
                })
                .collect::<Vec<_>>();
            if valid_cols.is_empty() {
                continue;
            }
            for nc in 1..=valid_cols.len() {
                let cols = &valid_cols[..nc];
                let nr = rows.len();
                if nr * nc > ordered.len() {
                    continue;
                }
                let per_core_m = mt_base.div_ceil(nr);
                let per_core_n = nt_base.div_ceil(nc);
                let mt = nr * per_core_m;
                let nt = nc * per_core_n;
                let out_tiles = per_core_m * per_core_n;
                let bw_cap = if out_tiles <= 16 { 32 } else { 64 };
                for out_subblock_h in 1..=8 {
                    for out_subblock_w in 1..=8 {
                        let out_subblock_num_tiles = out_subblock_h * out_subblock_w;
                        if out_subblock_num_tiles > 8
                            || out_subblock_num_tiles < min_out_subblock_num_tiles
                            || out_subblock_w < min_out_subblock_w
                            || per_core_m % out_subblock_h != 0
                            || per_core_n % out_subblock_w != 0
                        {
                            continue;
                        }
                        for &in0_block_w in &kt_divs {
                            if in0_block_w > bw_cap
                                || in0_block_w > max_in0_block_w
                                || !fits_l1(
                                    per_core_m,
                                    per_core_n,
                                    in0_block_w,
                                    tile_bytes,
                                    l1_data_bytes,
                                )
                            {
                                continue;
                            }
                            let bias = out_tiles.min(16);
                            let active_cores = nr * nc;
                            let score = (
                                active_cores * in0_block_w * bias * bias,
                                usize::MAX - mt * nt,
                                active_cores * in0_block_w,
                                out_subblock_num_tiles,
                                active_cores,
                            );
                            if best_score.map_or(true, |current| score > current) {
                                best_score = Some(score);
                                best = Some((
                                    rows.to_vec(),
                                    cols.to_vec(),
                                    mt,
                                    nt,
                                    per_core_m,
                                    per_core_n,
                                    in0_block_w,
                                    out_subblock_h,
                                    out_subblock_w,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    let Some((
        rows,
        cols,
        mt,
        nt,
        per_core_m,
        per_core_n,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
    )) = best
    else {
        return Err(invalid_input(format!(
            "no valid matmul plan for M={m} K={k} N={n} on {} cores",
            ordered.len()
        )));
    };

    Ok(MatmulPlan {
        mt,
        kt,
        nt,
        rows,
        cols,
        per_core_m,
        per_core_n,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
    })
}

fn crosses_column_gap(plan: &MatmulPlan) -> bool {
    plan.cols.iter().any(|&x| x < 8) && plan.cols.iter().any(|&x| x >= 10)
}

fn fits_l1(
    per_core_m: usize,
    per_core_n: usize,
    in0_block_w: usize,
    tile_bytes: usize,
    l1_data_bytes: usize,
) -> bool {
    let cb0 = 2 * per_core_m * in0_block_w * tile_bytes;
    let cb1 = 2 * per_core_n * in0_block_w * tile_bytes;
    let cb_out = per_core_m * per_core_n * tile_bytes;
    cb0 + cb1 + cb_out <= l1_data_bytes
}

fn divisors(n: usize) -> Vec<usize> {
    let mut divs = Vec::new();
    let mut i = 1usize;
    while i * i <= n {
        if n % i == 0 {
            divs.push(i);
            if i != n / i {
                divs.push(n / i);
            }
        }
        i += 1;
    }
    divs.sort_unstable();
    divs
}

fn bf16_program(
    plan: &MatmulPlan,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    output: &DramBuffer,
    zero_tile: &DramBuffer,
    logical_mt: usize,
    logical_nt: usize,
) -> io::Result<Program> {
    bf16_program_for_columns(plan, lhs, rhs, output, zero_tile, logical_mt, logical_nt, 0)
}

fn bf16_program_for_columns(
    plan: &MatmulPlan,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    output: &DramBuffer,
    zero_tile: &DramBuffer,
    logical_mt: usize,
    logical_nt: usize,
    col_offset_tiles: usize,
) -> io::Result<Program> {
    let lhs_addr = u32_arg(lhs.addr, "lhs address")?;
    let rhs_addr = u32_arg(rhs.addr, "rhs address")?;
    let output_addr = u32_arg(output.addr, "output address")?;
    let zero_addr = u32_arg(zero_tile.addr, "zero tile address")?;
    let grid = plan_grid(plan);

    let mut per_core_reader_args = Vec::new();
    let mut per_core_writer_args = Vec::new();
    for (row_index, row) in grid.iter().enumerate() {
        for (col_index, &core) in row.iter().enumerate() {
            per_core_reader_args.push((
                (core.x, core.y),
                reader_args(
                    plan, lhs_addr, zero_addr, &grid, row_index, core, logical_mt,
                )?,
            ));
            per_core_writer_args.push((
                (core.x, core.y),
                writer_args(
                    plan,
                    rhs_addr,
                    output_addr,
                    &grid,
                    row_index,
                    col_index,
                    core,
                    zero_addr,
                    logical_mt,
                    logical_nt,
                    col_offset_tiles,
                )?,
            ));
        }
    }

    let reader_args = per_core_reader_args
        .first()
        .map(|(_, args)| args.clone())
        .unwrap_or_default();
    let writer_args = per_core_writer_args
        .first()
        .map(|(_, args)| args.clone())
        .unwrap_or_default();

    Ok(Program {
        cores: CoreSelection::All,
        reader_kernel: BF16_READER_SENDER.to_owned(),
        writer_kernel: BF16_WRITER_SENDER.to_owned(),
        compute_kernel: compute_src(plan),
        reader_recv_kernel: BF16_READER_RECV.to_owned(),
        writer_recv_kernel: BF16_WRITER_RECV.to_owned(),
        cbs: vec![
            CBConfig {
                index: 0,
                dtype: DType::Float16B,
                tiles: plan.cb0_pages(),
            },
            CBConfig {
                index: 1,
                dtype: DType::Float16B,
                tiles: plan.cb1_pages(),
            },
            CBConfig {
                index: 16,
                dtype: DType::Float16B,
                tiles: plan.out_block_num_tiles(),
            },
            CBConfig {
                index: 24,
                dtype: DType::Float16B,
                tiles: plan.out_block_num_tiles(),
            },
        ],
        name: format!(
            "matmul_bf16_{}x{}x{}",
            plan.mt * 32,
            plan.kt * 32,
            plan.nt * 32
        ),
        reader_args,
        writer_args,
        compute_args: Vec::new(),
        semaphores: NUM_SEMAPHORES,
        math_fidelity: MathFidelity::LoFi,
        grid: Some((plan.rows.clone(), plan.cols.clone())),
        per_core_reader_args,
        per_core_writer_args,
        ..Program::default()
    })
}

fn plan_grid(plan: &MatmulPlan) -> Vec<Vec<CoreCoord>> {
    plan.rows
        .iter()
        .map(|&y| plan.cols.iter().map(|&x| CoreCoord { x, y }).collect())
        .collect()
}

fn reader_args(
    plan: &MatmulPlan,
    lhs_addr: u32,
    zero_addr: u32,
    grid: &[Vec<CoreCoord>],
    row_index: usize,
    core: CoreCoord,
    logical_mt: usize,
) -> io::Result<Vec<u32>> {
    let west_cols = plan
        .cols
        .iter()
        .copied()
        .filter(|&x| x < 8)
        .collect::<Vec<_>>();
    let east_cols = plan
        .cols
        .iter()
        .copied()
        .filter(|&x| x >= 10)
        .collect::<Vec<_>>();
    let w_rect = mcast_rect_args(
        &west_cols
            .iter()
            .copied()
            .filter(|&x| x != core.x)
            .collect::<Vec<_>>(),
        core.y,
    );
    let e_rect = mcast_rect_args(
        &east_cols
            .iter()
            .copied()
            .filter(|&x| x != core.x)
            .collect::<Vec<_>>(),
        core.y,
    );
    let sender = grid[row_index][0];
    Ok(vec![
        lhs_addr,
        u32_value(row_index * plan.per_core_m * plan.kt, "lhs block offset")?,
        1,
        u32_value(plan.kt, "lhs row stride")?,
        u32_value(plan.in0_block_w, "lhs block advance")?,
        u32_value(plan.in0_block_w, "lhs block width")?,
        u32_value(plan.per_core_m, "lhs block height")?,
        u32_value(plan.in0_block_num_tiles(), "lhs block tiles")?,
        u32_value(plan.num_blocks(), "num blocks")?,
        w_rect[0],
        w_rect[1],
        w_rect[2],
        w_rect[3],
        w_rect[4],
        e_rect[0],
        e_rect[1],
        e_rect[2],
        e_rect[3],
        e_rect[4],
        sender.x as u32,
        sender.y as u32,
        0,
        1,
        zero_addr,
        u32_value(logical_mt, "logical M tiles")?,
    ])
}

fn writer_args(
    plan: &MatmulPlan,
    rhs_addr: u32,
    output_addr: u32,
    grid: &[Vec<CoreCoord>],
    row_index: usize,
    col_index: usize,
    core: CoreCoord,
    zero_addr: u32,
    logical_mt: usize,
    logical_nt: usize,
    col_offset_tiles: usize,
) -> io::Result<Vec<u32>> {
    let recv_ys = plan.rows.iter().copied().skip(1).collect::<Vec<_>>();
    let mcast = if recv_ys.is_empty() {
        [0, 0, 0, 0, 0]
    } else {
        [
            core.x as u32,
            *recv_ys.iter().max().expect("recv_ys is non-empty") as u32,
            core.x as u32,
            *recv_ys.iter().min().expect("recv_ys is non-empty") as u32,
            recv_ys.len() as u32,
        ]
    };
    let sender = grid[0][col_index];
    let column_start = col_offset_tiles + col_index * plan.per_core_n;
    let out_start = row_index * plan.per_core_m * plan.nt + col_index * plan.per_core_n;
    Ok(vec![
        rhs_addr,
        u32_value(column_start, "rhs block offset")?,
        1,
        u32_value(logical_nt, "rhs row stride")?,
        u32_value(plan.in0_block_w * logical_nt, "rhs block advance")?,
        u32_value(plan.per_core_n, "rhs block width")?,
        u32_value(plan.in0_block_w, "rhs block height")?,
        u32_value(plan.in1_block_num_tiles(), "rhs block tiles")?,
        u32_value(plan.num_blocks(), "num blocks")?,
        mcast[0],
        mcast[1],
        mcast[2],
        mcast[3],
        mcast[4],
        sender.x as u32,
        sender.y as u32,
        2,
        3,
        output_addr,
        u32_value(out_start, "output tile offset")?,
        1,
        u32_value(plan.nt, "output row stride")?,
        u32_value(plan.out_subblock_w, "output next subblock w")?,
        u32_value(plan.out_subblock_h * plan.nt, "output next subblock h")?,
        u32_value(plan.out_subblock_w, "output subblock width")?,
        u32_value(plan.out_subblock_h, "output subblock height")?,
        u32_value(plan.out_subblock_num_tiles(), "output subblock tiles")?,
        u32_value(plan.in1_num_subblocks(), "output num subblocks w")?,
        u32_value(plan.in0_num_subblocks(), "output num subblocks h")?,
        u32_value(logical_mt, "logical M tiles")?,
        u32_value(logical_nt, "logical N tiles")?,
        zero_addr,
        u32_value(col_offset_tiles, "output column offset")?,
    ])
}

fn mcast_rect_args(cols: &[u8], y: u8) -> [u32; 5] {
    if cols.is_empty() {
        [0, 0, 0, 0, 0]
    } else {
        [
            *cols.iter().min().expect("cols is non-empty") as u32,
            y as u32,
            *cols.iter().max().expect("cols is non-empty") as u32,
            y as u32,
            cols.len() as u32,
        ]
    }
}

fn compute_src(plan: &MatmulPlan) -> String {
    let replacements = [
        ("@IN0_BLOCK_W@", plan.in0_block_w),
        ("@IN0_NUM_SUBBLOCKS@", plan.in0_num_subblocks()),
        ("@IN0_BLOCK_NUM_TILES@", plan.in0_block_num_tiles()),
        ("@IN0_SUBBLOCK_NUM_TILES@", plan.in0_subblock_num_tiles()),
        ("@IN1_NUM_SUBBLOCKS@", plan.in1_num_subblocks()),
        ("@IN1_BLOCK_NUM_TILES@", plan.in1_block_num_tiles()),
        ("@IN1_PER_CORE_W@", plan.per_core_n),
        ("@NUM_BLOCKS@", plan.num_blocks()),
        ("@OUT_SUBBLOCK_H@", plan.out_subblock_h),
        ("@OUT_SUBBLOCK_W@", plan.out_subblock_w),
        ("@OUT_SUBBLOCK_NUM_TILES@", plan.out_subblock_num_tiles()),
        ("@OUT_BLOCK_NUM_TILES@", plan.out_block_num_tiles()),
    ];

    let mut src = BF16_COMPUTE_TEMPLATE.to_owned();
    for (token, value) in replacements {
        src = src.replace(token, &value.to_string());
    }
    src
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}

fn u32_value(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_input(format!("{name} does not fit in u32: {value}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cores(cols: &[u8], rows: &[u8]) -> Vec<CoreCoord> {
        cols.iter()
            .flat_map(|&x| rows.iter().map(move |&y| CoreCoord { x, y }))
            .collect()
    }

    fn p100_worker_cores() -> Vec<CoreCoord> {
        cores(
            &[1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14],
            &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )
        .into_iter()
        .filter(|core| *core != CoreCoord { x: 14, y: 2 })
        .filter(|core| *core != CoreCoord { x: 14, y: 3 })
        .collect()
    }

    #[test]
    fn plan_matmul_uses_exact_tiling() {
        let plan = plan_matmul(64, 64, 64, &cores(&[1, 2], &[2, 3])).expect("plan");
        assert_eq!(plan.mt, 2);
        assert_eq!(plan.kt, 2);
        assert_eq!(plan.nt, 2);
        assert_eq!(plan.per_core_m * plan.rows.len(), plan.mt);
        assert_eq!(plan.per_core_n * plan.cols.len(), plan.nt);
    }

    #[test]
    fn compute_source_contains_plan_constants() {
        let plan = plan_matmul(64, 64, 64, &cores(&[1], &[2])).expect("plan");
        let source = compute_src(&plan);
        assert!(source.contains("constexpr uint32_t in0_block_w = 2;"));
        assert!(source.contains("#include \"compute_kernel_api/matmul.h\""));
    }

    #[test]
    fn plan_matmul_prefers_square_exact_grid() {
        let plan = plan_matmul(512, 512, 512, &p100_worker_cores()).expect("plan");
        assert_eq!(plan.per_core_m * plan.rows.len(), plan.mt);
        assert_eq!(plan.per_core_n * plan.cols.len(), plan.nt);
        assert!(plan.mt >= 16);
        assert!(plan.nt >= 16);
    }

    #[test]
    fn plan_matmul_prefers_throughput_for_large_shapes() {
        let plan = plan_matmul(4096, 8192, 4096, &p100_worker_cores()).expect("plan");
        assert!(!crosses_column_gap(&plan));
        assert_eq!(plan.rows, vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(plan.cols, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(plan.mt, 130);
        assert_eq!(plan.nt, 132);
        assert_eq!(plan.per_core_m, 13);
        assert_eq!(plan.per_core_n, 22);
        assert_eq!(plan.in0_block_w, 4);
        assert_eq!(plan.out_subblock_h, 1);
        assert_eq!(plan.out_subblock_w, 2);
    }

    #[test]
    fn split_planner_uses_both_sides_without_cross_gap_multicast() {
        let split = plan_split_matmul(4096, 8192, 4096, &p100_worker_cores()).expect("split plan");
        let split = split.expect("large p100 matmul should split across the column gap");

        assert_eq!(split.west.col_offset_tiles, 0);
        assert_eq!(split.east.col_offset_tiles, split.west.logical_nt);
        assert_eq!(split.west.logical_nt + split.east.logical_nt, 128);
        assert!(split.west.plan.cols.iter().all(|&x| x < 7));
        assert!(split.east.plan.cols.iter().all(|&x| x >= 10));
        assert!(!crosses_column_gap(&split.west.plan));
        assert!(!crosses_column_gap(&split.east.plan));
        assert_eq!(split.west.plan.in0_block_w, 4);
        assert_eq!(split.east.plan.in0_block_w, 4);
    }

    #[test]
    fn reader_args_exclude_east_sender_from_multicast_receivers() {
        let plan = plan_matmul_for_cores(
            4096,
            8192,
            1536,
            &p100_worker_cores()
                .into_iter()
                .filter(|core| core.x >= 10)
                .collect::<Vec<_>>(),
        )
        .expect("east plan");
        let grid = plan_grid(&plan);
        let sender = grid[0][0];
        let args = reader_args(&plan, 1, 2, &grid, 0, sender, 128).expect("reader args");
        assert_eq!(args[18] as usize, plan.cols.len() - 1);
    }

    #[test]
    fn unrestricted_planner_matches_blackhole_large_shape() {
        let plan = plan_matmul_for_cores(4096, 8192, 4096, &p100_worker_cores()).expect("plan");
        assert_eq!(plan.rows, vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(plan.cols, vec![1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]);
        assert_eq!(plan.mt, 130);
        assert_eq!(plan.nt, 132);
        assert_eq!(plan.per_core_m, 13);
        assert_eq!(plan.per_core_n, 12);
        assert_eq!(plan.in0_block_w, 8);
    }
}
