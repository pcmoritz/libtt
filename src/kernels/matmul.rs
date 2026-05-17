use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, MathFidelity, Program};
use crate::dram::{DType, DramBuffer};
use crate::hw::{CoreCoord, TensixL1};
use crate::kernels::kernel::{Kernel, RuntimeArgs, RuntimeArgsBuilder};
use crate::log::log;
use std::env;
use std::io;
use std::sync::Arc;

const BF16_READER_SENDER: &str = include_str!("../../kernels/matmul_reader_sender.cc");
const BF16_READER_RECV: &str = include_str!("../../kernels/matmul_reader_recv.cc");
const BF16_WRITER_SENDER: &str = include_str!("../../kernels/matmul_writer_sender.cc");
const BF16_WRITER_RECV: &str = include_str!("../../kernels/matmul_writer_recv.cc");
const BF16_COMPUTE_TEMPLATE: &str = include_str!("../../kernels/matmul_compute.cc");
const NUM_SEMAPHORES: usize = 4;
const READER_LHS_ADDR_INDEX: usize = 0;
const WRITER_RHS_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 18;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MatmulProgramKey {
    logical_mt: usize,
    logical_kt: usize,
    logical_nt: usize,
    cores: Arc<[CoreCoord]>,
    math_fidelity: MathFidelity,
    output_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MatmulPlan {
    rows: Vec<u8>,
    cols: Vec<u8>,
    direct_grid: Option<Vec<Vec<CoreCoord>>>,
    mt: usize,
    kt: usize,
    nt: usize,
    per_core_m: usize,
    per_core_n: usize,
    in0_block_w: usize,
    out_subblock_h: usize,
    out_subblock_w: usize,
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

struct MatmulBf16Kernel {
    lhs_addr: u32,
    rhs_addr: u32,
    output_addr: u32,
    key: MatmulProgramKey,
}

impl Kernel<MatmulProgramKey> for MatmulBf16Kernel {
    fn program_key(&self) -> MatmulProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        let plan = plan_matmul(
            self.key.logical_mt * 32,
            self.key.logical_kt * 32,
            self.key.logical_nt * 32,
            &self.key.cores,
        )?;
        log_matmul_plan(&plan);
        bf16_program(
            &plan,
            self.key.logical_mt,
            self.key.logical_nt,
            self.key.math_fidelity,
            self.key.output_dtype,
        )
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_LHS_ADDR_INDEX => Some(self.lhs_addr),
            _ => None,
        }
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            WRITER_RHS_ADDR_INDEX => Some(self.rhs_addr),
            WRITER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            _ => None,
        }
    }
}

pub(crate) fn matmul_bf16(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    matmul_bf16_with_output_dtype(device, lhs, rhs, DType::Float16B, name)
}

pub(crate) fn matmul_bf16_with_output_dtype(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    output_dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if lhs.dtype != DType::Float16B || rhs.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "matmul_bf16 requires bf16 inputs, got {:?} and {:?}",
            lhs.dtype, rhs.dtype
        )));
    }
    if !matches!(output_dtype, DType::Float16B | DType::Float32) {
        return Err(invalid_input(format!(
            "matmul_bf16 output must be Float16B or Float32, got {output_dtype:?}"
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
    let logical_kt = k / 32;
    let logical_nt = n / 32;
    let math_fidelity = matmul_math_fidelity()?;
    let cores = device.cores_arc();
    let output = device.alloc(
        logical_mt * logical_nt,
        output_dtype,
        &[m, n],
        output_name,
    )?;
    let key = MatmulProgramKey {
        logical_mt,
        logical_kt,
        logical_nt,
        cores,
        math_fidelity,
        output_dtype,
    };
    let kernel = MatmulBf16Kernel {
        lhs_addr: u32_arg(lhs.addr, "lhs address")?,
        rhs_addr: u32_arg(rhs.addr, "rhs address")?,
        output_addr: u32_arg(output.addr, "output address")?,
        key,
    };
    kernel.run(device)?;
    Ok(output)
}

fn log_matmul_plan(plan: &MatmulPlan) {
    let grid = plan_grid(plan);
    let grid_rows = grid.len();
    let grid_cols = grid.first().map_or(0, Vec::len);
    log(format!(
        "matmul_bf16 plan: Mt={} Kt={} Nt={} grid={}x{} mode={} per_core_M={} per_core_N={} in0_block_w={} num_blocks={} subblock={}x{}",
        plan.mt,
        plan.kt,
        plan.nt,
        grid_rows,
        grid_cols,
        if plan.direct_grid.is_some() { "direct" } else { "mcast" },
        plan.per_core_m,
        plan.per_core_n,
        plan.in0_block_w,
        plan.num_blocks(),
        plan.out_subblock_h,
        plan.out_subblock_w
    ));
}

fn shape_2d(buffer: &DramBuffer, name: &str) -> io::Result<(usize, usize)> {
    let shape = &buffer.shape;
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
    let mt_base = ceil32(m) / 32;
    let kt = ceil32(k) / 32;
    let nt_base = (ceil32(n) / 32).max(1);
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
                            || per_core_m % out_subblock_h != 0
                            || per_core_n % out_subblock_w != 0
                        {
                            continue;
                        }
                        for &in0_block_w in &kt_divs {
                            if in0_block_w > bw_cap
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
                                    None,
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
        direct_grid,
        mt,
        nt,
        per_core_m,
        per_core_n,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
    )) = best.or_else(|| {
        plan_direct_matmul(
            mt_base,
            nt_base,
            &ordered,
            &kt_divs,
            tile_bytes,
            l1_data_bytes,
        )
    })
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
        direct_grid,
        per_core_m,
        per_core_n,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
    })
}

#[allow(clippy::type_complexity)]
fn plan_direct_matmul(
    mt_base: usize,
    nt_base: usize,
    cores: &[CoreCoord],
    kt_divs: &[usize],
    tile_bytes: usize,
    l1_data_bytes: usize,
) -> Option<(
    Vec<u8>,
    Vec<u8>,
    Option<Vec<Vec<CoreCoord>>>,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
)> {
    if mt_base == 0 || nt_base == 0 {
        return None;
    }

    let mut best = None;
    let mut best_score = None;
    let max_rows = mt_base.min(cores.len());
    for logical_rows in 1..=max_rows {
        let max_cols = nt_base.min(cores.len() / logical_rows);
        for logical_cols in 1..=max_cols {
            let active_cores = logical_rows * logical_cols;
            let per_core_m = mt_base.div_ceil(logical_rows);
            let per_core_n = nt_base.div_ceil(logical_cols);
            let mt = logical_rows * per_core_m;
            let nt = logical_cols * per_core_n;
            let out_tiles = per_core_m * per_core_n;
            let bw_cap = if out_tiles <= 16 { 32 } else { 64 };

            for out_subblock_h in 1..=8 {
                for out_subblock_w in 1..=8 {
                    let out_subblock_num_tiles = out_subblock_h * out_subblock_w;
                    if out_subblock_num_tiles > 8
                        || per_core_m % out_subblock_h != 0
                        || per_core_n % out_subblock_w != 0
                    {
                        continue;
                    }
                    for &in0_block_w in kt_divs {
                        if in0_block_w > bw_cap
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
                        let padding = mt * nt - mt_base * nt_base;
                        let bias = out_tiles.min(16);
                        // Direct plans do not amortize narrow output blocks with multicast.
                        // Prefer wider subblocks even if that leaves a few cores idle.
                        let score = (
                            out_subblock_num_tiles,
                            active_cores * in0_block_w * bias * bias,
                            usize::MAX - padding,
                            active_cores * in0_block_w,
                            active_cores,
                        );
                        if best_score.map_or(true, |current| score > current) {
                            best_score = Some(score);
                            best = Some((
                                Vec::new(),
                                Vec::new(),
                                Some(
                                    cores[..active_cores]
                                        .chunks(logical_cols)
                                        .map(|row| row.to_vec())
                                        .collect(),
                                ),
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

    best
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

fn ceil32(value: usize) -> usize {
    (value + 31) & !31
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
    logical_mt: usize,
    logical_nt: usize,
    math_fidelity: MathFidelity,
    output_dtype: DType,
) -> io::Result<Program> {
    let cbs = vec![
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
            dtype: output_dtype,
            tiles: plan.out_block_num_tiles(),
        },
        CBConfig {
            index: 24,
            dtype: output_dtype,
            tiles: plan.out_block_num_tiles(),
        },
    ];
    let runtime_args = lower_runtime_args(plan, logical_mt, logical_nt)?;
    Ok(Program {
        reader_kernel: BF16_READER_SENDER.to_owned(),
        writer_kernel: BF16_WRITER_SENDER.to_owned(),
        compute_kernel: compute_src(plan),
        reader_recv_kernel: BF16_READER_RECV.to_owned(),
        writer_recv_kernel: BF16_WRITER_RECV.to_owned(),
        name: format!(
            "matmul_bf16_{:?}_{}x{}x{}",
            output_dtype,
            plan.mt * 32,
            plan.kt * 32,
            plan.nt * 32
        ),
        compile: CompileConfig {
            cbs,
            math_fidelity,
            dst_accum_mode: output_dtype == DType::Float32,
            ..CompileConfig::default()
        },
        grid: plan
            .direct_grid
            .is_none()
            .then(|| (plan.rows.clone(), plan.cols.clone())),
        ..Program::new(runtime_args)
    })
}

fn lower_runtime_args(
    plan: &MatmulPlan,
    logical_mt: usize,
    logical_nt: usize,
) -> io::Result<RuntimeArgs> {
    let grid = plan_grid(plan);
    let mut runtime_args = RuntimeArgsBuilder::new(
        NUM_SEMAPHORES,
        vec![WRITER_RHS_ADDR_INDEX, WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_LHS_ADDR_INDEX],
        Vec::new(),
    );
    for (row_index, row) in grid.iter().enumerate() {
        for (col_index, &core) in row.iter().enumerate() {
            let reader = reader_args(plan, &grid, row_index, core, logical_mt)?;
            let writer = writer_args(
                plan, &grid, row_index, col_index, core, logical_mt, logical_nt,
            )?;
            runtime_args.add_core(core, writer, reader, Vec::new())?;
        }
    }
    runtime_args.build()
}

fn matmul_math_fidelity() -> io::Result<MathFidelity> {
    match env::var("LIBTT_MATMUL_FIDELITY") {
        Ok(value) => parse_matmul_math_fidelity(&value),
        Err(env::VarError::NotPresent) => Ok(MathFidelity::HiFi2),
        Err(env::VarError::NotUnicode(_)) => {
            Err(invalid_input("LIBTT_MATMUL_FIDELITY must be valid Unicode"))
        }
    }
}

fn parse_matmul_math_fidelity(value: &str) -> io::Result<MathFidelity> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "hifi2" | "hi2" | "2" => Ok(MathFidelity::HiFi2),
        "lofi" | "lo" | "0" => Ok(MathFidelity::LoFi),
        other => Err(invalid_input(format!(
            "invalid LIBTT_MATMUL_FIDELITY={other:?}; expected lofi or hifi2"
        ))),
    }
}

fn plan_grid(plan: &MatmulPlan) -> Vec<Vec<CoreCoord>> {
    if let Some(grid) = &plan.direct_grid {
        grid.clone()
    } else {
        plan.rows
            .iter()
            .map(|&y| plan.cols.iter().map(|&x| CoreCoord { x, y }).collect())
            .collect()
    }
}

fn reader_args(
    plan: &MatmulPlan,
    grid: &[Vec<CoreCoord>],
    row_index: usize,
    core: CoreCoord,
    logical_mt: usize,
) -> io::Result<Vec<u32>> {
    let (w_rect, e_rect, sender) = if plan.direct_grid.is_some() {
        ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], core)
    } else {
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
        (
            mcast_rect_args(
                &west_cols
                    .iter()
                    .copied()
                    .filter(|&x| x != core.x)
                    .collect::<Vec<_>>(),
                core.y,
            ),
            mcast_rect_args(
                &east_cols
                    .iter()
                    .copied()
                    .filter(|&x| x != core.x)
                    .collect::<Vec<_>>(),
                core.y,
            ),
            grid[row_index][0],
        )
    };
    let mut args = vec![
        0,
        u32_value(row_index * plan.per_core_m * plan.kt, "lhs block offset")?,
        1,
        u32_value(plan.kt, "lhs row stride")?,
        u32_value(plan.in0_block_w, "lhs block advance")?,
        u32_value(plan.in0_block_w, "lhs block width")?,
        u32_value(plan.per_core_m, "lhs block height")?,
        u32_value(plan.in0_block_num_tiles(), "lhs block tiles")?,
        u32_value(plan.num_blocks(), "num blocks")?,
    ];
    for value in w_rect {
        args.push(value);
    }
    for value in e_rect {
        args.push(value);
    }
    args.extend([
        sender.x as u32,
        sender.y as u32,
        0,
        1,
        u32_value(logical_mt, "logical M tiles")?,
    ]);
    Ok(args)
}

fn writer_args(
    plan: &MatmulPlan,
    grid: &[Vec<CoreCoord>],
    row_index: usize,
    col_index: usize,
    core: CoreCoord,
    logical_mt: usize,
    logical_nt: usize,
) -> io::Result<Vec<u32>> {
    let mcast = if plan.direct_grid.is_some() || plan.rows.len() <= 1 {
        [0, 0, 0, 0, 0]
    } else {
        let recv_ys = &plan.rows[1..];
        [
            core.x as u32,
            *recv_ys.last().expect("recv_ys is non-empty") as u32,
            core.x as u32,
            recv_ys[0] as u32,
            recv_ys.len() as u32,
        ]
    };
    let sender = if plan.direct_grid.is_some() {
        core
    } else {
        grid[0][col_index]
    };
    let column_start = col_index * plan.per_core_n;
    let out_start = row_index * plan.per_core_m * plan.nt + col_index * plan.per_core_n;
    let mut args = vec![
        0,
        u32_value(column_start, "rhs block offset")?,
        1,
        u32_value(logical_nt, "rhs row stride")?,
        u32_value(plan.in0_block_w * logical_nt, "rhs block advance")?,
        u32_value(plan.per_core_n, "rhs block width")?,
        u32_value(plan.in0_block_w, "rhs block height")?,
        u32_value(plan.in1_block_num_tiles(), "rhs block tiles")?,
        u32_value(plan.num_blocks(), "num blocks")?,
    ];
    for value in mcast {
        args.push(value);
    }
    args.extend([
        sender.x as u32,
        sender.y as u32,
        2,
        3,
        0,
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
        0,
    ]);
    Ok(args)
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
    fn matmul_fidelity_parser_defaults_empty_value_to_hifi2() {
        assert_eq!(
            parse_matmul_math_fidelity("").expect("empty value should use the default"),
            MathFidelity::HiFi2
        );
        assert_eq!(
            parse_matmul_math_fidelity("hifi2").expect("hifi2 should parse"),
            MathFidelity::HiFi2
        );
    }

    #[test]
    fn matmul_fidelity_parser_accepts_explicit_lofi_override() {
        assert_eq!(
            parse_matmul_math_fidelity("lofi").expect("lofi should parse"),
            MathFidelity::LoFi
        );
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
        assert_eq!(plan.rows, vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(plan.cols, vec![1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]);
        assert_eq!(plan.mt, 130);
        assert_eq!(plan.nt, 132);
        assert_eq!(plan.per_core_m, 13);
        assert_eq!(plan.per_core_n, 12);
        assert_eq!(plan.in0_block_w, 8);
        assert_eq!(plan.out_subblock_h, 1);
        assert_eq!(plan.out_subblock_w, 6);
    }

    #[test]
    fn plan_matmul_uses_ceiled_tile_shape() {
        let plan = plan_matmul(33, 65, 1, &cores(&[1], &[2])).expect("plan");
        assert_eq!(plan.mt, 2);
        assert_eq!(plan.kt, 3);
        assert_eq!(plan.nt, 1);
    }

    #[test]
    fn plan_matmul_uses_direct_grid_for_wide_projection() {
        let plan = plan_matmul(32, 1024, 151936, &p100_worker_cores()).expect("plan");
        let grid = plan.direct_grid.as_ref().expect("direct plan");
        assert_eq!(grid.len(), 1);
        assert_eq!(grid[0].len(), 101);
        assert_eq!(plan.mt, 1);
        assert_eq!(plan.kt, 32);
        assert_eq!(plan.per_core_m, 1);
        assert_eq!(plan.per_core_n, 48);
        assert_eq!(plan.out_subblock_w, 8);
    }

    #[test]
    fn plan_matmul_direct_grid_is_not_limited_to_single_m_tile() {
        let plan = plan_matmul(64, 1024, 151936, &p100_worker_cores()).expect("plan");
        let grid = plan.direct_grid.as_ref().expect("direct plan");
        assert_eq!(grid.iter().map(Vec::len).sum::<usize>(), 110);
        assert_eq!(plan.mt, 2);
        assert_eq!(plan.kt, 32);
        assert_eq!(plan.per_core_m * grid.len(), plan.mt);
        assert_eq!(plan.out_subblock_h * plan.out_subblock_w, 8);
    }

    #[test]
    fn reader_args_exclude_east_sender_from_multicast_receivers() {
        let plan = plan_matmul(
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
        let mut builder = RuntimeArgsBuilder::new(0, Vec::new(), Vec::new(), Vec::new());
        let reader = reader_args(&plan, &grid, 0, sender, 128).expect("reader args");
        builder
            .add_core(sender, Vec::new(), reader, Vec::new())
            .expect("add core");
        let runtime_args = builder.build().expect("lower runtime args");
        let offset = 18 * 4;
        let value = u32::from_le_bytes(
            runtime_args.blobs()[0][offset..offset + 4]
                .try_into()
                .expect("u32 runtime arg"),
        );
        assert_eq!(value as usize, plan.cols.len() - 1);
    }
}
