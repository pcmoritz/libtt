use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, MathFidelity, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::hw::{CoreCoord, TensixL1};
use crate::kernels::kernel::{Kernel, RuntimeArgs, RuntimeArgsBuilder};
use crate::log::log;
use std::env;
use std::io;
use std::sync::Arc;

const BF16_READER_SENDER: &str = concat!(
    include_str!("../../kernels/matmul_common.cc"),
    include_str!("../../kernels/matmul_reader_sender.cc")
);
const BF16_READER_RECV: &str = include_str!("../../kernels/matmul_reader_recv.cc");
const BF16_WRITER_SENDER: &str = concat!(
    include_str!("../../kernels/matmul_common.cc"),
    include_str!("../../kernels/matmul_writer_common.cc"),
    include_str!("../../kernels/matmul_writer_sender.cc")
);
const BF16_WRITER_RECV: &str = concat!(
    include_str!("../../kernels/matmul_common.cc"),
    include_str!("../../kernels/matmul_writer_common.cc"),
    include_str!("../../kernels/matmul_writer_recv.cc")
);
const BF16_MATMUL_TOP1_WRITER: &str = concat!(
    include_str!("../../kernels/matmul_common.cc"),
    include_str!("../../kernels/matmul_top1_writer_sender.cc")
);
const BF16_COMPUTE_TEMPLATE: &str = include_str!("../../kernels/matmul_compute.cc");
const NUM_SEMAPHORES: usize = 4;
const READER_LHS_ADDR_INDEX: usize = 0;
const WRITER_RHS_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 18;
const WRITER_PARTIAL_VALUES_ADDR_INDEX: usize = 18;
const WRITER_PARTIAL_INDICES_ADDR_INDEX: usize = 19;
const MAX_RANK: usize = 8;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum MatmulEpilogueKind {
    Store,
    Top1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MatmulEpilogue {
    Store { output_dtype: DType },
    Top1,
}

impl MatmulEpilogue {
    fn kind(self) -> MatmulEpilogueKind {
        match self {
            Self::Store { .. } => MatmulEpilogueKind::Store,
            Self::Top1 => MatmulEpilogueKind::Top1,
        }
    }
}

#[derive(Debug)]
pub(crate) enum MatmulOutput {
    Store(DramBuffer),
    Top1 {
        values: DramBuffer,
        indices: DramBuffer,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatmulRuntimeEpilogue {
    Store {
        output_addr: u32,
    },
    Top1 {
        partial_values_addr: u32,
        partial_indices_addr: u32,
    },
}

impl MatmulRuntimeEpilogue {
    fn kind(self) -> MatmulEpilogueKind {
        match self {
            Self::Store { .. } => MatmulEpilogueKind::Store,
            Self::Top1 { .. } => MatmulEpilogueKind::Top1,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MatmulProgramKey {
    logical_mt: usize,
    logical_kt: usize,
    logical_nt: usize,
    batch_count: usize,
    lhs_view: MatmulOperandView,
    rhs_view: MatmulOperandView,
    output_view: MatmulOperandView,
    cores: Arc<[CoreCoord]>,
    math_fidelity: MathFidelity,
    output_dtype: DType,
    epilogue: MatmulEpilogueKind,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum MatmulViewKind {
    Contiguous = 0,
    Generic = 2,
    TiledIndexMap = 4,
}

impl MatmulViewKind {
    fn runtime_value(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct MatmulOperandView {
    kind: MatmulViewKind,
    rank: u32,
    batch_rank: u32,
    row_rank: u32,
    col_rank: u32,
    logical_rows: u32,
    logical_cols: u32,
    tile_rows: u32,
    tiles_per_row: u32,
    shape: [u32; MAX_RANK],
    batch_dims: [u32; MAX_RANK],
    row_dims: [u32; MAX_RANK],
    col_dims: [u32; MAX_RANK],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MatmulPlan {
    rows: Vec<u8>,
    cols: Vec<u8>,
    direct_grid: Option<Vec<Vec<CoreCoord>>>,
    batch_groups: usize,
    batches_per_group: usize,
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

    fn direct_grid_rows_per_batch(&self) -> Option<usize> {
        self.direct_grid.as_ref().map(|grid| {
            debug_assert!(self.batch_groups > 0);
            debug_assert_eq!(grid.len() % self.batch_groups, 0);
            grid.len() / self.batch_groups
        })
    }
}

struct MatmulBf16Kernel {
    lhs_addr: u32,
    rhs_addr: u32,
    epilogue: MatmulRuntimeEpilogue,
    key: MatmulProgramKey,
}

impl Kernel<MatmulProgramKey> for MatmulBf16Kernel {
    fn program_key(&self) -> MatmulProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        let plan = plan_for_key(&self.key)?;
        log_matmul_plan(&plan);
        bf16_program(
            &plan,
            self.key.logical_mt,
            self.key.logical_nt,
            self.key.batch_count,
            &self.key.lhs_view,
            &self.key.rhs_view,
            &self.key.output_view,
            self.key.math_fidelity,
            self.key.output_dtype,
            self.key.epilogue,
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
        debug_assert_eq!(self.key.epilogue, self.epilogue.kind());
        match self.epilogue {
            MatmulRuntimeEpilogue::Store { output_addr } => match index {
                WRITER_RHS_ADDR_INDEX => Some(self.rhs_addr),
                WRITER_OUTPUT_ADDR_INDEX => Some(output_addr),
                _ => None,
            },
            MatmulRuntimeEpilogue::Top1 {
                partial_values_addr,
                partial_indices_addr,
            } => match index {
                WRITER_RHS_ADDR_INDEX => Some(self.rhs_addr),
                WRITER_PARTIAL_VALUES_ADDR_INDEX => Some(partial_values_addr),
                WRITER_PARTIAL_INDICES_ADDR_INDEX => Some(partial_indices_addr),
                _ => None,
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_bf16_dot_general(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    lhs_logical_shape: &[usize],
    rhs_logical_shape: &[usize],
    output_logical_shape: &[usize],
    lhs_batching_dimensions: &[i64],
    rhs_batching_dimensions: &[i64],
    lhs_contracting_dimensions: &[i64],
    rhs_contracting_dimensions: &[i64],
    epilogue: MatmulEpilogue,
    name: impl Into<String>,
) -> io::Result<MatmulOutput> {
    if lhs.dtype != DType::Float16B || rhs.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "matmul_bf16 requires bf16 inputs, got {:?} and {:?}",
            lhs.dtype, rhs.dtype
        )));
    }
    let output_dtype = match epilogue {
        MatmulEpilogue::Store { output_dtype } => {
            if !matches!(output_dtype, DType::Float16B | DType::Float32) {
                return Err(invalid_input(format!(
                    "matmul_bf16 output must be Float16B or Float32, got {output_dtype:?}"
                )));
            }
            output_dtype
        }
        MatmulEpilogue::Top1 => DType::Float16B,
    };
    let epilogue_kind = epilogue.kind();

    let shape = dot_general_shape(
        lhs_logical_shape,
        rhs_logical_shape,
        output_logical_shape,
        lhs_batching_dimensions,
        rhs_batching_dimensions,
        lhs_contracting_dimensions,
        rhs_contracting_dimensions,
    )?;
    if epilogue_kind == MatmulEpilogueKind::Top1 && (shape.batch_count != 1 || shape.m != 1) {
        return Err(invalid_input(format!(
            "matmul_top1 expects a single row output, got batch_count={} M={}",
            shape.batch_count, shape.m
        )));
    }
    validate_tile_count(lhs, tiled_shape_tile_count(lhs_logical_shape)?, "lhs")?;
    validate_tile_count(rhs, tiled_shape_tile_count(rhs_logical_shape)?, "rhs")?;

    let logical_mt = ceil32(shape.m) / 32;
    let logical_kt = ceil32(shape.k) / 32;
    let logical_nt = ceil32(shape.n) / 32;
    let math_fidelity = matmul_math_fidelity()?;
    let cores = device.cores_arc();
    let name = name.into();
    let key = MatmulProgramKey {
        logical_mt,
        logical_kt,
        logical_nt,
        batch_count: shape.batch_count,
        lhs_view: shape.lhs_view,
        rhs_view: shape.rhs_view,
        output_view: shape.output_view,
        cores,
        math_fidelity,
        output_dtype,
        epilogue: epilogue_kind,
    };

    match epilogue {
        MatmulEpilogue::Store { output_dtype } => {
            let output_tiles = tiled_shape_tile_count(output_logical_shape)?;
            let output_shape = tiled_allocation_shape(output_logical_shape)?;
            let output = device.alloc(output_tiles, output_dtype, &output_shape, name)?;
            let kernel = MatmulBf16Kernel {
                lhs_addr: u32_arg(lhs.addr, "lhs address")?,
                rhs_addr: u32_arg(rhs.addr, "rhs address")?,
                epilogue: MatmulRuntimeEpilogue::Store {
                    output_addr: u32_arg(output.addr, "output address")?,
                },
                key,
            };
            kernel.run(device)?;
            Ok(MatmulOutput::Store(output))
        }
        MatmulEpilogue::Top1 => {
            let plan = plan_for_key(&key)?;
            let partial_count = plan_grid(&plan).iter().map(Vec::len).sum::<usize>();
            let partial_shape = [partial_count * 32, 32];
            let partial_values = device.alloc(
                partial_count,
                DType::Float16B,
                &partial_shape,
                format!("{name}_partial_values"),
            )?;
            let partial_indices = device.alloc(
                partial_count,
                DType::Int32,
                &partial_shape,
                format!("{name}_partial_indices"),
            )?;
            let kernel = MatmulBf16Kernel {
                lhs_addr: u32_arg(lhs.addr, "lhs address")?,
                rhs_addr: u32_arg(rhs.addr, "rhs address")?,
                epilogue: MatmulRuntimeEpilogue::Top1 {
                    partial_values_addr: u32_arg(partial_values.addr, "partial values address")?,
                    partial_indices_addr: u32_arg(partial_indices.addr, "partial indices address")?,
                },
                key,
            };
            log("matmul_top1: running partial matmul".to_owned());
            kernel.run(device)?;
            log("matmul_top1: partial matmul complete".to_owned());
            log("matmul_top1: finalizing partials".to_owned());
            let (values, indices) = crate::kernels::topk::top1_finalize_partials(
                device,
                &partial_values,
                &partial_indices,
                partial_count,
                format!("{name}_top1"),
            )?;
            log("matmul_top1: finalize complete".to_owned());
            Ok(MatmulOutput::Top1 { values, indices })
        }
    }
}

fn log_matmul_plan(plan: &MatmulPlan) {
    let grid = plan_grid(plan);
    let grid_rows = grid.len();
    let grid_cols = grid.first().map_or(0, Vec::len);
    log(format!(
        "matmul_bf16 plan: Mt={} Kt={} Nt={} grid={}x{} batch_groups={} batches_per_group={} mode={} per_core_M={} per_core_N={} in0_block_w={} num_blocks={} subblock={}x{}",
        plan.mt,
        plan.kt,
        plan.nt,
        grid_rows,
        grid_cols,
        plan.batch_groups,
        plan.batches_per_group,
        if plan.direct_grid.is_some() {
            "direct"
        } else {
            "mcast"
        },
        plan.per_core_m,
        plan.per_core_n,
        plan.in0_block_w,
        plan.num_blocks(),
        plan.out_subblock_h,
        plan.out_subblock_w
    ));
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DotGeneralMatmulShape {
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
    lhs_view: MatmulOperandView,
    rhs_view: MatmulOperandView,
    output_view: MatmulOperandView,
}

fn dot_general_shape(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    lhs_batching_dimensions: &[i64],
    rhs_batching_dimensions: &[i64],
    lhs_contracting_dimensions: &[i64],
    rhs_contracting_dimensions: &[i64],
) -> io::Result<DotGeneralMatmulShape> {
    if !(2..=MAX_RANK).contains(&lhs_shape.len())
        || !(2..=MAX_RANK).contains(&rhs_shape.len())
        || output_shape.len() < 2
    {
        return Err(invalid_input(format!(
            "dot_general matmul requires lhs/rhs ranks in 2..={MAX_RANK} and output rank >= 2, got lhs={lhs_shape:?} rhs={rhs_shape:?} output={output_shape:?}"
        )));
    }
    if lhs_shape.contains(&0) || rhs_shape.contains(&0) || output_shape.contains(&0) {
        return Err(invalid_input(
            "dot_general matmul zero-sized dimensions are not currently supported",
        ));
    }
    if lhs_batching_dimensions.len() != rhs_batching_dimensions.len() {
        return Err(invalid_input(format!(
            "dot_general lhs/rhs batching dimension counts must match, got {} and {}",
            lhs_batching_dimensions.len(),
            rhs_batching_dimensions.len()
        )));
    }
    if lhs_contracting_dimensions.len() != rhs_contracting_dimensions.len() {
        return Err(invalid_input(format!(
            "dot_general lhs/rhs contracting dimension counts must match, got {} and {}",
            lhs_contracting_dimensions.len(),
            rhs_contracting_dimensions.len()
        )));
    }

    let lhs_dims = dot_general_dims(
        lhs_shape.len(),
        lhs_batching_dimensions,
        lhs_contracting_dimensions,
        "lhs",
    )?;
    let rhs_dims = dot_general_dims(
        rhs_shape.len(),
        rhs_batching_dimensions,
        rhs_contracting_dimensions,
        "rhs",
    )?;

    let mut batch_shape = Vec::with_capacity(lhs_dims.batch.len());
    for (&lhs_dim, &rhs_dim) in lhs_dims.batch.iter().zip(&rhs_dims.batch) {
        if lhs_shape[lhs_dim] != rhs_shape[rhs_dim] {
            return Err(invalid_input(format!(
                "dot_general batch dimensions must match, got lhs dim {lhs_dim}={} and rhs dim {rhs_dim}={}",
                lhs_shape[lhs_dim], rhs_shape[rhs_dim]
            )));
        }
        batch_shape.push(lhs_shape[lhs_dim]);
    }
    for (&lhs_dim, &rhs_dim) in lhs_dims.contract.iter().zip(&rhs_dims.contract) {
        if lhs_shape[lhs_dim] != rhs_shape[rhs_dim] {
            return Err(invalid_input(format!(
                "dot_general contracting dimensions must match, got lhs dim {lhs_dim}={} and rhs dim {rhs_dim}={}",
                lhs_shape[lhs_dim], rhs_shape[rhs_dim]
            )));
        }
    }

    let batch_count = checked_product(&batch_shape, "dot_general batch dimensions")?;
    let m = checked_product_of_dims(lhs_shape, &lhs_dims.free, "lhs free dimensions")?;
    let k = checked_product_of_dims(lhs_shape, &lhs_dims.contract, "lhs contracting dimensions")?;
    let rhs_k =
        checked_product_of_dims(rhs_shape, &rhs_dims.contract, "rhs contracting dimensions")?;
    let n = checked_product_of_dims(rhs_shape, &rhs_dims.free, "rhs free dimensions")?;
    if k != rhs_k {
        return Err(invalid_input(format!(
            "dot_general contracting dimension products must match, got {k} and {rhs_k}"
        )));
    }

    let mut expected_output = batch_shape.clone();
    for &dim in &lhs_dims.free {
        expected_output.push(lhs_shape[dim]);
    }
    for &dim in &rhs_dims.free {
        expected_output.push(rhs_shape[dim]);
    }
    if output_shape != expected_output {
        return Err(invalid_input(format!(
            "dot_general matmul output shape mismatch: expected shape {:?}, got {output_shape:?}",
            expected_output
        )));
    }
    let output_batch_dims = (0..batch_shape.len()).collect::<Vec<_>>();
    let output_row_dims =
        (batch_shape.len()..batch_shape.len() + lhs_dims.free.len()).collect::<Vec<_>>();
    let output_col_dims =
        (batch_shape.len() + lhs_dims.free.len()..output_shape.len()).collect::<Vec<_>>();

    Ok(DotGeneralMatmulShape {
        batch_count,
        m,
        k,
        n,
        lhs_view: operand_view(
            lhs_shape,
            &lhs_dims.batch,
            &lhs_dims.free,
            &lhs_dims.contract,
            m,
            k,
        )?,
        rhs_view: operand_view(
            rhs_shape,
            &rhs_dims.batch,
            &rhs_dims.contract,
            &rhs_dims.free,
            k,
            n,
        )?,
        output_view: operand_view(
            output_shape,
            &output_batch_dims,
            &output_row_dims,
            &output_col_dims,
            m,
            n,
        )?,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DotGeneralDims {
    batch: Vec<usize>,
    contract: Vec<usize>,
    free: Vec<usize>,
}

fn dot_general_dims(
    rank: usize,
    batch_dimensions: &[i64],
    contracting_dimensions: &[i64],
    name: &str,
) -> io::Result<DotGeneralDims> {
    let mut used = vec![false; rank];
    let mut parse_dims = |dims: &[i64], kind: &str| -> io::Result<Vec<usize>> {
        let mut parsed = Vec::with_capacity(dims.len());
        for &dim in dims {
            let dim = usize::try_from(dim).map_err(|_| {
                invalid_input(format!("dot_general {name} {kind} dimensions must be >= 0"))
            })?;
            if dim >= rank {
                return Err(invalid_input(format!(
                    "dot_general {name} {kind} dimension {dim} is out of bounds for rank {rank}"
                )));
            }
            if std::mem::replace(&mut used[dim], true) {
                return Err(invalid_input(format!(
                    "dot_general {name} dimension {dim} is used more than once"
                )));
            }
            parsed.push(dim);
        }
        Ok(parsed)
    };

    let batch = parse_dims(batch_dimensions, "batching")?;
    let contract = parse_dims(contracting_dimensions, "contracting")?;
    let free = (0..rank).filter(|&dim| !used[dim]).collect::<Vec<_>>();
    Ok(DotGeneralDims {
        batch,
        contract,
        free,
    })
}

fn checked_product_of_dims(shape: &[usize], dims: &[usize], name: &str) -> io::Result<usize> {
    dims.iter()
        .map(|&dim| shape[dim])
        .try_fold(1usize, |acc, value| {
            acc.checked_mul(value)
                .ok_or_else(|| invalid_input(format!("{name} product overflow")))
        })
}

fn operand_view(
    shape: &[usize],
    batch_dims: &[usize],
    row_dims: &[usize],
    col_dims: &[usize],
    logical_rows: usize,
    logical_cols: usize,
) -> io::Result<MatmulOperandView> {
    let allocation_shape = tiled_allocation_shape(shape)?;
    let rank = shape.len();
    let kind = operand_view_kind(rank, batch_dims, row_dims, col_dims);
    Ok(MatmulOperandView {
        kind,
        rank: u32_value(rank, "matmul operand rank")?,
        batch_rank: u32_value(batch_dims.len(), "matmul operand batch rank")?,
        row_rank: u32_value(row_dims.len(), "matmul operand row rank")?,
        col_rank: u32_value(col_dims.len(), "matmul operand column rank")?,
        logical_rows: u32_value(logical_rows, "matmul operand logical rows")?,
        logical_cols: u32_value(logical_cols, "matmul operand logical columns")?,
        tile_rows: u32_value(
            allocation_shape[rank - 2] / 32,
            "matmul operand source tile rows",
        )?,
        tiles_per_row: u32_value(
            allocation_shape[rank - 1] / 32,
            "matmul operand source tiles per row",
        )?,
        shape: padded_u32_array(shape, "matmul operand shape")?,
        batch_dims: padded_u32_array(batch_dims, "matmul operand batch dimensions")?,
        row_dims: padded_u32_array(row_dims, "matmul operand row dimensions")?,
        col_dims: padded_u32_array(col_dims, "matmul operand column dimensions")?,
    })
}

fn operand_view_kind(
    rank: usize,
    batch_dims: &[usize],
    row_dims: &[usize],
    col_dims: &[usize],
) -> MatmulViewKind {
    let leading_batch = batch_dims.iter().copied().eq(0..batch_dims.len());
    if leading_batch
        && rank == batch_dims.len() + 2
        && row_dims == [rank - 2]
        && col_dims == [rank - 1]
    {
        MatmulViewKind::Contiguous
    } else if is_tiled_index_map_view(rank, batch_dims, row_dims, col_dims) {
        MatmulViewKind::TiledIndexMap
    } else {
        MatmulViewKind::Generic
    }
}

fn is_tiled_index_map_view(
    rank: usize,
    batch_dims: &[usize],
    row_dims: &[usize],
    col_dims: &[usize],
) -> bool {
    // Example: [batch, token, head, dim] viewed as [batch, head, dim, token].
    // The matmul row dim is the physical innermost dim, and the matmul column
    // dim is a prefix dim, so each output column maps to one source tile.
    if rank < 3 || row_dims != [rank - 1] || col_dims.len() != 1 || col_dims[0] >= rank - 2 {
        return false;
    }
    (0..rank)
        .filter(|&dim| dim != col_dims[0] && dim != row_dims[0])
        .eq(batch_dims.iter().copied())
}

fn padded_u32_array(values: &[usize], name: &str) -> io::Result<[u32; MAX_RANK]> {
    if values.len() > MAX_RANK {
        return Err(invalid_input(format!(
            "{name} rank {} exceeds maximum rank {MAX_RANK}",
            values.len()
        )));
    }
    let mut result = [0u32; MAX_RANK];
    for (index, &value) in values.iter().enumerate() {
        result[index] = u32_value(value, name)?;
    }
    Ok(result)
}

fn checked_product(values: &[usize], name: &str) -> io::Result<usize> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or_else(|| invalid_input(format!("{name} product overflow")))
    })
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

fn plan_for_key(key: &MatmulProgramKey) -> io::Result<MatmulPlan> {
    match key.epilogue {
        MatmulEpilogueKind::Store => plan_matmul(
            key.logical_mt * 32,
            key.logical_kt * 32,
            key.logical_nt * 32,
            key.batch_count,
            &key.cores,
            key.output_view.kind == MatmulViewKind::Contiguous,
        ),
        MatmulEpilogueKind::Top1 => plan_matmul(
            32,
            key.logical_kt * 32,
            key.output_view.logical_cols as usize,
            1,
            &key.cores,
            true,
        ),
    }
}

fn plan_matmul(
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    cores: &[CoreCoord],
    allow_column_split: bool,
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
            let max_nc = if allow_column_split {
                valid_cols.len()
            } else {
                1
            };
            for nc in 1..=max_nc {
                let cols = &valid_cols[..nc];
                let nr = rows.len();
                if nr > mt_base {
                    continue;
                }
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
                                best = Some(MatmulPlan {
                                    rows: rows.to_vec(),
                                    cols: cols.to_vec(),
                                    direct_grid: None,
                                    batch_groups: 1,
                                    batches_per_group: batch_count,
                                    mt,
                                    kt,
                                    nt,
                                    per_core_m,
                                    per_core_n,
                                    in0_block_w,
                                    out_subblock_h,
                                    out_subblock_w,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    let direct = || {
        plan_direct_matmul(
            mt_base,
            kt,
            nt_base,
            &ordered,
            &kt_divs,
            tile_bytes,
            l1_data_bytes,
            batch_count,
            allow_column_split,
            1,
            1,
            None,
        )
    };
    let candidate = if mt_base == 1 {
        direct().or(best)
    } else {
        best.or_else(direct)
    };
    let Some(candidate) = candidate else {
        return Err(invalid_input(format!(
            "no valid matmul plan for M={m} K={k} N={n} on {} cores",
            ordered.len()
        )));
    };

    let mut plan = candidate;
    let baseline_work = plan_work(&plan);
    if let Some(batched) = plan_direct_matmul(
        mt_base,
        kt,
        nt_base,
        &ordered,
        &kt_divs,
        tile_bytes,
        l1_data_bytes,
        batch_count,
        allow_column_split,
        2,
        batch_count.min(ordered.len()),
        Some(baseline_work),
    ) {
        plan = batched;
    }

    Ok(plan)
}

fn plan_direct_matmul(
    mt_base: usize,
    kt: usize,
    nt_base: usize,
    cores: &[CoreCoord],
    kt_divs: &[usize],
    tile_bytes: usize,
    l1_data_bytes: usize,
    batch_count: usize,
    allow_column_split: bool,
    min_batch_groups: usize,
    max_batch_groups: usize,
    baseline_work: Option<usize>,
) -> Option<MatmulPlan> {
    if mt_base == 0 || nt_base == 0 {
        return None;
    }

    let mut best = None;
    let mut best_score = None;
    let min_batch_groups = min_batch_groups.max(1);
    let max_batch_groups = max_batch_groups.min(batch_count).min(cores.len());
    if min_batch_groups > max_batch_groups {
        return None;
    }

    for batch_groups in min_batch_groups..=max_batch_groups {
        let max_group_cores = cores.len() / batch_groups;
        if max_group_cores == 0 {
            continue;
        }
        let batches_per_group = batch_count.div_ceil(batch_groups);
        for logical_rows in 1..=mt_base.min(max_group_cores) {
            let max_cols = if allow_column_split {
                nt_base.min(max_group_cores / logical_rows)
            } else {
                1
            };
            for logical_cols in 1..=max_cols {
                let active_cores_per_group = logical_rows * logical_cols;
                let active_cores = active_cores_per_group * batch_groups;
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
                            let num_blocks = kt / in0_block_w;
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
                            let per_core_work =
                                batches_per_group * num_blocks * per_core_m * per_core_n;
                            if batch_groups > 1 {
                                let Some(baseline) = baseline_work else {
                                    continue;
                                };
                                if per_core_work >= baseline {
                                    continue;
                                }
                            }
                            let padding = (mt * nt - mt_base * nt_base) * batch_groups;
                            let bias = out_tiles.min(16);
                            let score = if batch_groups > 1 {
                                (
                                    usize::MAX - per_core_work,
                                    active_cores,
                                    usize::MAX - padding,
                                    in0_block_w,
                                    out_subblock_num_tiles,
                                )
                            } else if mt_base == 1 {
                                (
                                    usize::MAX - per_core_work,
                                    usize::MAX - padding,
                                    active_cores,
                                    out_subblock_num_tiles,
                                    in0_block_w,
                                )
                            } else {
                                (
                                    out_subblock_num_tiles,
                                    active_cores * in0_block_w * bias * bias,
                                    usize::MAX - padding,
                                    active_cores * in0_block_w,
                                    active_cores,
                                )
                            };
                            if best_score.map_or(true, |current| score > current) {
                                best_score = Some(score);
                                best = Some(MatmulPlan {
                                    rows: Vec::new(),
                                    cols: Vec::new(),
                                    direct_grid: Some(
                                        cores[..active_cores]
                                            .chunks(logical_cols)
                                            .map(|row| row.to_vec())
                                            .collect(),
                                    ),
                                    batch_groups,
                                    batches_per_group,
                                    mt,
                                    kt,
                                    nt,
                                    per_core_m,
                                    per_core_n,
                                    in0_block_w,
                                    out_subblock_h,
                                    out_subblock_w,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    best
}

fn plan_work(plan: &MatmulPlan) -> usize {
    plan.batches_per_group
        .saturating_mul(plan.num_blocks())
        .saturating_mul(plan.per_core_m)
        .saturating_mul(plan.per_core_n)
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
    let transpose_scratch = 2 * tile_bytes;
    cb0 + cb1 + cb_out + transpose_scratch <= l1_data_bytes
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
    batch_count: usize,
    lhs_view: &MatmulOperandView,
    rhs_view: &MatmulOperandView,
    output_view: &MatmulOperandView,
    math_fidelity: MathFidelity,
    output_dtype: DType,
    epilogue: MatmulEpilogueKind,
) -> io::Result<Program> {
    let mut cbs = vec![
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
            index: 2,
            dtype: DType::Float16B,
            tiles: 1,
        },
        CBConfig {
            index: 3,
            dtype: DType::Float16B,
            tiles: 1,
        },
        CBConfig {
            index: 4,
            dtype: output_dtype,
            tiles: 1,
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
    if epilogue == MatmulEpilogueKind::Top1 {
        cbs.push(CBConfig {
            index: 17,
            dtype: DType::Int32,
            tiles: 1,
        });
    }
    let runtime_args = lower_runtime_args(
        plan,
        logical_mt,
        logical_nt,
        batch_count,
        lhs_view,
        rhs_view,
        output_view,
        epilogue,
    )?;
    let top1_epilogue = epilogue == MatmulEpilogueKind::Top1;
    Ok(Program {
        reader_kernel: BF16_READER_SENDER.to_owned(),
        writer_kernel: if top1_epilogue {
            BF16_MATMUL_TOP1_WRITER.to_owned()
        } else {
            BF16_WRITER_SENDER.to_owned()
        },
        compute_kernel: compute_src(plan, plan.batches_per_group),
        reader_recv_kernel: BF16_READER_RECV.to_owned(),
        writer_recv_kernel: if top1_epilogue {
            BF16_MATMUL_TOP1_WRITER.to_owned()
        } else {
            BF16_WRITER_RECV.to_owned()
        },
        name: format!(
            "matmul{}_bf16_{:?}_{}x{}x{}",
            if top1_epilogue { "_top1" } else { "" },
            output_dtype,
            plan.mt * 32,
            plan.kt * 32,
            plan.nt * 32
        ),
        compile: CompileConfig {
            cbs,
            math_fidelity,
            dst_accum_mode: true,
            dst_full_sync: true,
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
    batch_count: usize,
    lhs_view: &MatmulOperandView,
    rhs_view: &MatmulOperandView,
    output_view: &MatmulOperandView,
    epilogue: MatmulEpilogueKind,
) -> io::Result<RuntimeArgs> {
    let grid = plan_grid(plan);
    let writer_dynamic_indices = match epilogue {
        MatmulEpilogueKind::Store => vec![WRITER_RHS_ADDR_INDEX, WRITER_OUTPUT_ADDR_INDEX],
        MatmulEpilogueKind::Top1 => vec![
            WRITER_RHS_ADDR_INDEX,
            WRITER_PARTIAL_VALUES_ADDR_INDEX,
            WRITER_PARTIAL_INDICES_ADDR_INDEX,
        ],
    };
    let mut runtime_args = RuntimeArgsBuilder::new(
        NUM_SEMAPHORES,
        writer_dynamic_indices,
        vec![READER_LHS_ADDR_INDEX],
        Vec::new(),
    );
    let direct_grid_rows_per_batch = plan.direct_grid_rows_per_batch();
    let mut partial_tile_id = 0usize;
    for (flat_row_index, row) in grid.iter().enumerate() {
        let (batch_group, row_index) = if let Some(rows_per_batch) = direct_grid_rows_per_batch {
            (
                flat_row_index / rows_per_batch,
                flat_row_index % rows_per_batch,
            )
        } else {
            (0, flat_row_index)
        };
        let batch_start = batch_group * plan.batches_per_group;
        for (col_index, &core) in row.iter().enumerate() {
            let reader = reader_args(
                plan,
                &grid,
                row_index,
                core,
                logical_mt,
                plan.batches_per_group,
                batch_start,
                batch_count,
                lhs_view,
            )?;
            let writer_epilogue = match epilogue {
                MatmulEpilogueKind::Store => WriterEpilogue::Store,
                MatmulEpilogueKind::Top1 => WriterEpilogue::Top1 { partial_tile_id },
            };
            let writer = writer_args(
                plan,
                &grid,
                row_index,
                col_index,
                core,
                logical_mt,
                logical_nt,
                plan.batches_per_group,
                batch_start,
                batch_count,
                rhs_view,
                output_view,
                writer_epilogue,
            )?;
            runtime_args.add_core(core, writer, reader, Vec::new())?;
            if epilogue == MatmulEpilogueKind::Top1 {
                partial_tile_id += 1;
            }
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
    local_batch_count: usize,
    batch_start: usize,
    total_batch_count: usize,
    lhs_view: &MatmulOperandView,
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
    let lhs_block_offset = row_index * plan.per_core_m * plan.kt;
    let mut args = vec![
        0,
        u32_value(lhs_block_offset, "lhs block offset")?,
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
        u32_value(local_batch_count, "local batch count")?,
        u32_value(batch_start, "batch start")?,
        u32_value(total_batch_count, "total batch count")?,
        u32_value(logical_mt * plan.kt, "lhs batch stride")?,
    ]);
    append_view_args(&mut args, lhs_view);
    Ok(args)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriterEpilogue {
    Store,
    Top1 { partial_tile_id: usize },
}

fn writer_args(
    plan: &MatmulPlan,
    grid: &[Vec<CoreCoord>],
    row_index: usize,
    col_index: usize,
    core: CoreCoord,
    logical_mt: usize,
    logical_nt: usize,
    local_batch_count: usize,
    batch_start: usize,
    total_batch_count: usize,
    rhs_view: &MatmulOperandView,
    output_view: &MatmulOperandView,
    epilogue: WriterEpilogue,
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
    args.extend([sender.x as u32, sender.y as u32, 2, 3, 0]);
    if let WriterEpilogue::Top1 { .. } = epilogue {
        args.push(0);
    }
    args.extend([
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
    if let WriterEpilogue::Top1 { partial_tile_id } = epilogue {
        args.push(u32_value(partial_tile_id, "partial tile id")?);
    }
    args.extend([
        u32_value(local_batch_count, "local batch count")?,
        u32_value(batch_start, "batch start")?,
        u32_value(total_batch_count, "total batch count")?,
        u32_value(plan.kt * logical_nt, "rhs batch stride")?,
        u32_value(logical_mt * logical_nt, "output batch stride")?,
    ]);
    append_view_args(&mut args, rhs_view);
    append_view_args(&mut args, output_view);
    Ok(args)
}

fn append_view_args(args: &mut Vec<u32>, view: &MatmulOperandView) {
    args.extend([
        view.kind.runtime_value(),
        view.rank,
        view.batch_rank,
        view.row_rank,
        view.col_rank,
        view.logical_rows,
        view.logical_cols,
        view.tile_rows,
        view.tiles_per_row,
    ]);
    args.extend(view.shape);
    args.extend(view.batch_dims);
    args.extend(view.row_dims);
    args.extend(view.col_dims);
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

fn compute_src(plan: &MatmulPlan, batch_count: usize) -> String {
    let replacements = [
        ("@BATCH_COUNT@", batch_count),
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
        let plan = plan_matmul(64, 64, 64, 1, &cores(&[1, 2], &[2, 3]), true).expect("plan");
        let grid = plan_grid(&plan);
        assert_eq!(plan.mt, 2);
        assert_eq!(plan.kt, 2);
        assert_eq!(plan.nt, 2);
        assert_eq!(plan.per_core_m * grid.len(), plan.mt);
        assert_eq!(plan.per_core_n * grid[0].len(), plan.nt);
    }

    #[test]
    fn plan_matmul_prefers_square_exact_grid() {
        let plan = plan_matmul(512, 512, 512, 1, &p100_worker_cores(), true).expect("plan");
        assert_eq!(plan.per_core_m * plan.rows.len(), plan.mt);
        assert_eq!(plan.per_core_n * plan.cols.len(), plan.nt);
        assert!(plan.mt >= 16);
        assert!(plan.nt >= 16);
    }

    #[test]
    fn plan_matmul_prefers_throughput_for_large_shapes() {
        let plan = plan_matmul(4096, 8192, 4096, 1, &p100_worker_cores(), true).expect("plan");
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
        let plan = plan_matmul(33, 65, 1, 1, &cores(&[1], &[2]), true).expect("plan");
        assert_eq!(plan.mt, 2);
        assert_eq!(plan.kt, 3);
        assert_eq!(plan.nt, 1);
    }

    #[test]
    fn plan_matmul_uses_direct_grid_for_wide_projection() {
        let plan = plan_matmul(32, 1024, 151936, 1, &p100_worker_cores(), true).expect("plan");
        let grid = plan.direct_grid.as_ref().expect("direct plan");
        assert_eq!(grid.len(), 1);
        assert!(grid[0].len() >= 100);
        assert_eq!(plan.mt, 1);
        assert_eq!(plan.kt, 32);
        assert_eq!(plan.per_core_m, 1);
        assert!(plan.per_core_n <= 48);
    }

    #[test]
    fn plan_matmul_direct_grid_is_not_limited_to_single_m_tile() {
        let plan = plan_matmul(64, 1024, 151936, 1, &p100_worker_cores(), true).expect("plan");
        let grid = plan.direct_grid.as_ref().expect("direct plan");
        assert_eq!(grid.iter().map(Vec::len).sum::<usize>(), 110);
        assert_eq!(plan.mt, 2);
        assert_eq!(plan.kt, 32);
        assert_eq!(plan.per_core_m * grid.len(), plan.mt);
        assert_eq!(plan.out_subblock_h * plan.out_subblock_w, 8);
    }

    #[test]
    fn plan_matmul_splits_batches_across_direct_grid() {
        let plan = plan_matmul(32, 1024, 1024, 16, &p100_worker_cores(), true).expect("plan");
        let grid = plan.direct_grid.as_ref().expect("direct plan");
        assert!(plan.batch_groups > 1);
        assert_eq!(grid.len() % plan.batch_groups, 0);
        assert_eq!(
            plan.direct_grid_rows_per_batch(),
            Some(grid.len() / plan.batch_groups)
        );
    }

    #[test]
    fn reader_args_exclude_east_sender_from_multicast_receivers() {
        let plan = plan_matmul(
            4096,
            8192,
            1536,
            1,
            &p100_worker_cores()
                .into_iter()
                .filter(|core| core.x >= 10)
                .collect::<Vec<_>>(),
            true,
        )
        .expect("east plan");
        let grid = plan_grid(&plan);
        let sender = grid[0][0];
        let mut builder = RuntimeArgsBuilder::new(0, Vec::new(), Vec::new(), Vec::new());
        let lhs_view = operand_view(&[4096, 8192], &[], &[0], &[1], 4096, 8192).expect("lhs view");
        let reader =
            reader_args(&plan, &grid, 0, sender, 128, 1, 0, 1, &lhs_view).expect("reader args");
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
