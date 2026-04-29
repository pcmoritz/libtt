use crate::device::Device;
use crate::dispatch::{CBConfig, CoreSelection, MathFidelity, Program};
use crate::dram::{DType, DramBuffer};
use crate::hw::{CoreCoord, TensixL1};
use crate::kernels::cache::{PerCoreRuntimeArgs, ProgramCache, RuntimeArgList};
use crate::kernels::kernel::Kernel;
use crate::log::{enabled as log_enabled, log};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::hash::{Hash, Hasher};
use std::io;
use std::sync::{Mutex, OnceLock};

const BF16_READER_SENDER: &str = include_str!("../../kernels/matmul_reader_sender.cc");
const BF16_READER_RECV: &str = include_str!("../../kernels/matmul_reader_recv.cc");
const BF16_WRITER_SENDER: &str = include_str!("../../kernels/matmul_writer_sender.cc");
const BF16_WRITER_RECV: &str = include_str!("../../kernels/matmul_writer_recv.cc");
const BF16_COMPUTE_TEMPLATE: &str = include_str!("../../kernels/matmul_compute.cc");
const NUM_SEMAPHORES: usize = 4;
const READER_LHS_ADDR_INDEX: usize = 0;
const READER_ZERO_ADDR_INDEX: usize = 23;
const WRITER_RHS_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 18;
const WRITER_ZERO_ADDR_INDEX: usize = 31;

static ZERO_TILE_BY_DEVICE: OnceLock<Mutex<HashMap<usize, DramBuffer>>> = OnceLock::new();
static PLAN_CACHE: OnceLock<Mutex<HashMap<MatmulPlanKey, MatmulPlan>>> = OnceLock::new();
static PROGRAM_CACHE: ProgramCache<MatmulProgramKey> = ProgramCache::new("matmul program");

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MatmulPlanKey {
    m: usize,
    k: usize,
    n: usize,
    cores: Vec<CoreCoord>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MatmulProgramKey {
    static_key: u64,
    logical_mt: usize,
    logical_nt: usize,
    col_offset_tiles: usize,
}

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

struct MatmulBf16Kernel<'a> {
    lhs: &'a DramBuffer,
    rhs: &'a DramBuffer,
    output: &'a DramBuffer,
    zero_tile: &'a DramBuffer,
    logical_mt: usize,
    logical_nt: usize,
    math_fidelity: MathFidelity,
}

impl Kernel for MatmulBf16Kernel<'_> {
    fn program(&self, device: &Device) -> io::Result<Program> {
        let cores = device.cores();
        let m = self.logical_mt * 32;
        let k = self.lhs.shape.as_ref().expect("validated lhs shape")[1];
        let n = self.logical_nt * 32;
        let plan = cached_plan_matmul(m, k, n, &cores)?;
        log_matmul_plan(&plan);
        let static_key = matmul_static_key(&plan, self.math_fidelity);
        let key = MatmulProgramKey {
            static_key,
            logical_mt: self.logical_mt,
            logical_nt: self.logical_nt,
            col_offset_tiles: 0,
        };
        let base_program = PROGRAM_CACHE.get_or_insert_with(key, || {
            bf16_program(
                &plan,
                self.logical_mt,
                self.logical_nt,
                0,
                static_key,
                self.math_fidelity,
            )
        })?;
        base_program.update_runtime_args_from_kernel(self)
    }

    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> io::Result<Option<u32>> {
        match index {
            READER_LHS_ADDR_INDEX => Ok(Some(u32_arg(self.lhs.addr, "lhs address")?)),
            READER_ZERO_ADDR_INDEX => Ok(Some(u32_arg(self.zero_tile.addr, "zero tile address")?)),
            _ => Ok(None),
        }
    }

    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> io::Result<Option<u32>> {
        match index {
            WRITER_RHS_ADDR_INDEX => Ok(Some(u32_arg(self.rhs.addr, "rhs address")?)),
            WRITER_OUTPUT_ADDR_INDEX => Ok(Some(u32_arg(self.output.addr, "output address")?)),
            WRITER_ZERO_ADDR_INDEX => Ok(Some(u32_arg(self.zero_tile.addr, "zero tile address")?)),
            _ => Ok(None),
        }
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
    let math_fidelity = matmul_math_fidelity()?;
    let zero_tile = cached_zero_tile(device)?;
    let output = device.alloc(
        logical_mt * logical_nt,
        DType::Float16B,
        Some(&[m, n]),
        output_name,
    )?;
    let kernel = MatmulBf16Kernel {
        lhs,
        rhs,
        output: &output,
        zero_tile: &zero_tile,
        logical_mt,
        logical_nt,
        math_fidelity,
    };
    let program = kernel.program(device)?;
    device.run_program(&program)?;
    Ok(output)
}

fn log_matmul_plan(plan: &MatmulPlan) {
    if log_enabled() {
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
    }
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
    plan_matmul_for_cores(m, k, n, cores)
}

fn cached_plan_matmul(m: usize, k: usize, n: usize, cores: &[CoreCoord]) -> io::Result<MatmulPlan> {
    let mut ordered = cores.to_vec();
    ordered.sort_unstable();
    ordered.dedup();
    let key = MatmulPlanKey {
        m,
        k,
        n,
        cores: ordered,
    };
    let cache = PLAN_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(plan) = cache
        .lock()
        .map_err(|_| io::Error::other("matmul plan cache is poisoned"))?
        .get(&key)
        .cloned()
    {
        return Ok(plan);
    }

    let plan = plan_matmul(m, k, n, cores)?;
    cache
        .lock()
        .map_err(|_| io::Error::other("matmul plan cache is poisoned"))?
        .insert(key, plan.clone());
    Ok(plan)
}

fn plan_matmul_for_cores(
    m: usize,
    k: usize,
    n: usize,
    cores: &[CoreCoord],
) -> io::Result<MatmulPlan> {
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
    col_offset_tiles: usize,
    static_key: u64,
    math_fidelity: MathFidelity,
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
            dtype: DType::Float16B,
            tiles: plan.out_block_num_tiles(),
        },
        CBConfig {
            index: 24,
            dtype: DType::Float16B,
            tiles: plan.out_block_num_tiles(),
        },
    ];
    let runtime = lower_runtime_args(plan, logical_mt, logical_nt, col_offset_tiles)?;

    Ok(Program {
        static_key: Some(static_key),
        cores: CoreSelection::All,
        reader_kernel: BF16_READER_SENDER.to_owned(),
        writer_kernel: BF16_WRITER_SENDER.to_owned(),
        compute_kernel: compute_src(plan),
        reader_recv_kernel: BF16_READER_RECV.to_owned(),
        writer_recv_kernel: BF16_WRITER_RECV.to_owned(),
        cbs,
        name: format!(
            "matmul_bf16_{}x{}x{}",
            plan.mt * 32,
            plan.kt * 32,
            plan.nt * 32
        ),
        reader_args: runtime.reader_args,
        writer_args: runtime.writer_args,
        compute_args: runtime.compute_args,
        semaphores: NUM_SEMAPHORES,
        math_fidelity,
        grid: Some((plan.rows.clone(), plan.cols.clone())),
        runtime_args: Some(runtime.runtime_args),
        per_core_reader_args: Vec::new(),
        per_core_writer_args: Vec::new(),
        ..Program::default()
    })
}

fn lower_runtime_args(
    plan: &MatmulPlan,
    logical_mt: usize,
    logical_nt: usize,
    col_offset_tiles: usize,
) -> io::Result<crate::kernels::cache::PackedRuntimeArgs> {
    let grid = plan_grid(plan);
    let mut runtime_args = Vec::new();
    for (row_index, row) in grid.iter().enumerate() {
        for (col_index, &core) in row.iter().enumerate() {
            let reader = reader_args(plan, &grid, row_index, core, logical_mt)?;
            let writer = writer_args(
                plan,
                &grid,
                row_index,
                col_index,
                core,
                logical_mt,
                logical_nt,
                col_offset_tiles,
            )?;
            runtime_args.push(PerCoreRuntimeArgs {
                core,
                writer,
                reader,
                compute: RuntimeArgList::new(),
            });
        }
    }
    crate::dispatch::RuntimeArgs::from_per_core(runtime_args, NUM_SEMAPHORES)
}

fn matmul_static_key(plan: &MatmulPlan, math_fidelity: MathFidelity) -> u64 {
    let mut hasher = DefaultHasher::new();
    "matmul_bf16_v2".hash(&mut hasher);
    plan.rows.hash(&mut hasher);
    plan.cols.hash(&mut hasher);
    plan.mt.hash(&mut hasher);
    plan.kt.hash(&mut hasher);
    plan.nt.hash(&mut hasher);
    plan.per_core_m.hash(&mut hasher);
    plan.per_core_n.hash(&mut hasher);
    plan.in0_block_w.hash(&mut hasher);
    plan.out_subblock_h.hash(&mut hasher);
    plan.out_subblock_w.hash(&mut hasher);
    math_fidelity.hash(&mut hasher);
    hasher.finish()
}

fn matmul_math_fidelity() -> io::Result<MathFidelity> {
    match env::var("LIBTT_MATMUL_FIDELITY") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "" | "lofi" | "lo" | "0" => Ok(MathFidelity::LoFi),
            "hifi2" | "hi2" | "2" => Ok(MathFidelity::HiFi2),
            other => Err(invalid_input(format!(
                "invalid LIBTT_MATMUL_FIDELITY={other:?}; expected lofi or hifi2"
            ))),
        },
        Err(env::VarError::NotPresent) => Ok(MathFidelity::LoFi),
        Err(env::VarError::NotUnicode(_)) => {
            Err(invalid_input("LIBTT_MATMUL_FIDELITY must be valid Unicode"))
        }
    }
}

fn plan_grid(plan: &MatmulPlan) -> Vec<Vec<CoreCoord>> {
    plan.rows
        .iter()
        .map(|&y| plan.cols.iter().map(|&x| CoreCoord { x, y }).collect())
        .collect()
}

fn reader_args(
    plan: &MatmulPlan,
    grid: &[Vec<CoreCoord>],
    row_index: usize,
    core: CoreCoord,
    logical_mt: usize,
) -> io::Result<RuntimeArgList> {
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
    let mut args = RuntimeArgList::new();
    args.push_dynamic();
    args.push(u32_value(
        row_index * plan.per_core_m * plan.kt,
        "lhs block offset",
    )?);
    args.push(1);
    args.push(u32_value(plan.kt, "lhs row stride")?);
    args.push(u32_value(plan.in0_block_w, "lhs block advance")?);
    args.push(u32_value(plan.in0_block_w, "lhs block width")?);
    args.push(u32_value(plan.per_core_m, "lhs block height")?);
    args.push(u32_value(plan.in0_block_num_tiles(), "lhs block tiles")?);
    args.push(u32_value(plan.num_blocks(), "num blocks")?);
    for value in w_rect {
        args.push(value);
    }
    for value in e_rect {
        args.push(value);
    }
    args.push(sender.x as u32);
    args.push(sender.y as u32);
    args.push(0);
    args.push(1);
    args.push_dynamic();
    args.push(u32_value(logical_mt, "logical M tiles")?);
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
    col_offset_tiles: usize,
) -> io::Result<RuntimeArgList> {
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
    let mut args = RuntimeArgList::new();
    args.push_dynamic();
    args.push(u32_value(column_start, "rhs block offset")?);
    args.push(1);
    args.push(u32_value(logical_nt, "rhs row stride")?);
    args.push(u32_value(
        plan.in0_block_w * logical_nt,
        "rhs block advance",
    )?);
    args.push(u32_value(plan.per_core_n, "rhs block width")?);
    args.push(u32_value(plan.in0_block_w, "rhs block height")?);
    args.push(u32_value(plan.in1_block_num_tiles(), "rhs block tiles")?);
    args.push(u32_value(plan.num_blocks(), "num blocks")?);
    for value in mcast {
        args.push(value);
    }
    args.push(sender.x as u32);
    args.push(sender.y as u32);
    args.push(2);
    args.push(3);
    args.push_dynamic();
    args.push(u32_value(out_start, "output tile offset")?);
    args.push(1);
    args.push(u32_value(plan.nt, "output row stride")?);
    args.push(u32_value(
        plan.out_subblock_w,
        "output next subblock w",
    )?);
    args.push(u32_value(
        plan.out_subblock_h * plan.nt,
        "output next subblock h",
    )?);
    args.push(u32_value(plan.out_subblock_w, "output subblock width")?);
    args.push(u32_value(plan.out_subblock_h, "output subblock height")?);
    args.push(u32_value(
        plan.out_subblock_num_tiles(),
        "output subblock tiles",
    )?);
    args.push(u32_value(
        plan.in1_num_subblocks(),
        "output num subblocks w",
    )?);
    args.push(u32_value(
        plan.in0_num_subblocks(),
        "output num subblocks h",
    )?);
    args.push(u32_value(logical_mt, "logical M tiles")?);
    args.push(u32_value(logical_nt, "logical N tiles")?);
    args.push_dynamic();
    args.push(u32_value(col_offset_tiles, "output column offset")?);
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
        let plan = plan_matmul_for_cores(33, 65, 1, &cores(&[1], &[2])).expect("plan");
        assert_eq!(plan.mt, 2);
        assert_eq!(plan.kt, 3);
        assert_eq!(plan.nt, 1);
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
        let args = reader_args(&plan, &grid, 0, sender, 128).expect("reader args");
        assert_eq!(args.values()[18] as usize, plan.cols.len() - 1);
    }
}
