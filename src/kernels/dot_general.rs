use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{self, DType, DramBuffer, TILE_C, TILE_R};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const DPA_SCORE_READER: &str = include_str!("../../kernels/dpa_score_reader.cc");
const DPA_VALUE_READER: &str = include_str!("../../kernels/dpa_value_reader.cc");
const DPA_VALUE_WRITER: &str = include_str!("../../kernels/dpa_value_writer.cc");
const MATMUL_TILE_COMPUTE: &str = include_str!("../../kernels/matmul_tile_compute.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_LHS_ADDR_INDEX: usize = 0;
const READER_RHS_ADDR_INDEX: usize = 1;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug)]
pub(crate) struct DotGeneralSpec {
    pub(crate) lhs_shape: Vec<usize>,
    pub(crate) rhs_shape: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) lhs_batching_dimensions: Vec<i64>,
    pub(crate) rhs_batching_dimensions: Vec<i64>,
    pub(crate) lhs_contracting_dimensions: Vec<i64>,
    pub(crate) rhs_contracting_dimensions: Vec<i64>,
    pub(crate) output_dtype: DType,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DpaScoreShape {
    output_tile_count: u32,
    query_tokens: u32,
    key_tokens: u32,
    kv_heads: u32,
    groups: u32,
    head_dim: u32,
    lhs_tiles_per_prefix: u32,
    rhs_tiles_per_prefix: u32,
    output_tiles_per_row: u32,
    kt: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DpaScoreProgramKey {
    cores: Vec<CoreCoord>,
    shape: DpaScoreShape,
}

struct DpaScoreKernel {
    lhs_addr: u32,
    rhs_addr: u32,
    output_addr: u32,
    key: DpaScoreProgramKey,
}

impl Kernel<DpaScoreProgramKey> for DpaScoreKernel {
    fn program_key(&self) -> DpaScoreProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        dpa_score_program(&self.key)
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_LHS_ADDR_INDEX => Some(self.lhs_addr),
            READER_RHS_ADDR_INDEX => Some(self.rhs_addr),
            _ => None,
        }
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            WRITER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DpaValueShape {
    work_tile_count: u32,
    batch: u32,
    key_tokens: u32,
    query_tokens: u32,
    kv_heads: u32,
    groups: u32,
    head_dim: u32,
    head_tiles: u32,
    lhs_tiles_per_prefix: u32,
    rhs_tile_rows: u32,
    rhs_tiles_per_row: u32,
    output_tiles_per_row: u32,
    kt: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DpaValueProgramKey {
    cores: Vec<CoreCoord>,
    shape: DpaValueShape,
}

struct DpaValueKernel {
    lhs_addr: u32,
    rhs_addr: u32,
    output_addr: u32,
    key: DpaValueProgramKey,
}

impl Kernel<DpaValueProgramKey> for DpaValueKernel {
    fn program_key(&self) -> DpaValueProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        dpa_value_program(&self.key)
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_LHS_ADDR_INDEX => Some(self.lhs_addr),
            READER_RHS_ADDR_INDEX => Some(self.rhs_addr),
            _ => None,
        }
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            WRITER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            _ => None,
        }
    }
}

pub(crate) fn dot_general(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    spec: DotGeneralSpec,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let output_name = name.into();
    validate_dtype(lhs.dtype, "dot_general lhs")?;
    validate_dtype(rhs.dtype, "dot_general rhs")?;
    validate_dtype(spec.output_dtype, "dot_general output")?;
    validate_buffer(lhs, &spec.lhs_shape, "dot_general lhs")?;
    validate_buffer(rhs, &spec.rhs_shape, "dot_general rhs")?;
    if let Some(output) = try_dpa_score_dot(device, lhs, rhs, &spec, output_name.as_str())? {
        return Ok(output);
    }
    if let Some(output) = try_dpa_value_dot(device, lhs, rhs, &spec, output_name.as_str())? {
        return Ok(output);
    }
    Err(invalid_input(format!(
        "unsupported dot_general shape: only DPA score/value specializations are currently supported, got lhs={:?} rhs={:?} output={:?} lhs_batch={:?} rhs_batch={:?} lhs_contract={:?} rhs_contract={:?}",
        spec.lhs_shape,
        spec.rhs_shape,
        spec.output_shape,
        spec.lhs_batching_dimensions,
        spec.rhs_batching_dimensions,
        spec.lhs_contracting_dimensions,
        spec.rhs_contracting_dimensions
    )))
}

fn try_dpa_score_dot(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    spec: &DotGeneralSpec,
    name: &str,
) -> io::Result<Option<DramBuffer>> {
    if lhs.dtype != DType::Float16B
        || rhs.dtype != DType::Float16B
        || spec.output_dtype != DType::Float32
        || spec.lhs_shape.len() != 5
        || spec.rhs_shape.len() != 4
        || spec.output_shape.len() != 5
        || spec.lhs_batching_dimensions != [0, 2]
        || spec.rhs_batching_dimensions != [0, 2]
        || spec.lhs_contracting_dimensions != [4]
        || spec.rhs_contracting_dimensions != [3]
    {
        return Ok(None);
    }

    let batch = spec.lhs_shape[0];
    let query_tokens = spec.lhs_shape[1];
    let kv_heads = spec.lhs_shape[2];
    let groups = spec.lhs_shape[3];
    let head_dim = spec.lhs_shape[4];
    let key_tokens = spec.rhs_shape[1];
    if spec.rhs_shape[0] != batch
        || spec.rhs_shape[2] != kv_heads
        || spec.rhs_shape[3] != head_dim
        || spec.output_shape != [batch, kv_heads, query_tokens, groups, key_tokens]
    {
        return Ok(None);
    }
    if groups > TILE_R || kv_heads > TILE_R {
        return Ok(None);
    }

    let lhs_allocation = dram::tiled_allocation_shape(&spec.lhs_shape)?;
    let rhs_allocation = dram::tiled_allocation_shape(&spec.rhs_shape)?;
    let output_allocation = dram::tiled_allocation_shape(&spec.output_shape)?;
    let lhs_tile_rows = lhs_allocation[3] / TILE_R;
    let rhs_tile_rows = rhs_allocation[2] / TILE_R;
    let output_tile_rows = output_allocation[3] / TILE_R;
    if lhs_tile_rows != 1 || rhs_tile_rows != 1 || output_tile_rows != 1 {
        return Ok(None);
    }

    let kt = rhs_allocation[3] / TILE_C;
    let output_tile_count = dram::tiled_shape_tile_count(&spec.output_shape)?;
    let shape = DpaScoreShape {
        output_tile_count: u32_value(output_tile_count, "dpa score output tile count")?,
        query_tokens: u32_value(query_tokens, "dpa score query tokens")?,
        key_tokens: u32_value(key_tokens, "dpa score key tokens")?,
        kv_heads: u32_value(kv_heads, "dpa score kv heads")?,
        groups: u32_value(groups, "dpa score groups")?,
        head_dim: u32_value(head_dim, "dpa score head dim")?,
        lhs_tiles_per_prefix: u32_value(
            lhs_tile_rows * (lhs_allocation[4] / TILE_C),
            "dpa score lhs tiles per prefix",
        )?,
        rhs_tiles_per_prefix: u32_value(
            rhs_tile_rows * (rhs_allocation[3] / TILE_C),
            "dpa score rhs tiles per prefix",
        )?,
        output_tiles_per_row: u32_value(
            output_allocation[4] / TILE_C,
            "dpa score output tiles per row",
        )?,
        kt: u32_value(kt, "dpa score kt")?,
    };

    let output = device.alloc(
        output_tile_count,
        DType::Float32,
        &output_allocation,
        name.to_owned(),
    )?;
    let cores = select_worker_cores(device.cores_ref(), output.num_tiles)?;
    let kernel = DpaScoreKernel {
        lhs_addr: u32_addr(lhs.addr, "lhs address")?,
        rhs_addr: u32_addr(rhs.addr, "rhs address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: DpaScoreProgramKey { cores, shape },
    };
    kernel.run(device)?;
    Ok(Some(output))
}

fn try_dpa_value_dot(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    spec: &DotGeneralSpec,
    name: &str,
) -> io::Result<Option<DramBuffer>> {
    if lhs.dtype != DType::Float16B
        || rhs.dtype != DType::Float16B
        || spec.output_dtype != DType::Float16B
        || spec.lhs_shape.len() != 4
        || spec.rhs_shape.len() != 5
        || spec.output_shape.len() != 5
        || spec.lhs_batching_dimensions != [0, 2]
        || spec.rhs_batching_dimensions != [1, 2]
        || spec.lhs_contracting_dimensions != [1]
        || spec.rhs_contracting_dimensions != [4]
    {
        return Ok(None);
    }

    let batch = spec.lhs_shape[0];
    let key_tokens = spec.lhs_shape[1];
    let kv_heads = spec.lhs_shape[2];
    let head_dim = spec.lhs_shape[3];
    let groups = spec.rhs_shape[0];
    let query_tokens = spec.rhs_shape[3];
    if spec.rhs_shape[1] != batch
        || spec.rhs_shape[2] != kv_heads
        || spec.rhs_shape[4] != key_tokens
        || spec.output_shape != [batch, kv_heads, head_dim, groups, query_tokens]
    {
        return Ok(None);
    }
    if groups > TILE_R || kv_heads > TILE_R {
        return Ok(None);
    }

    let lhs_allocation = dram::tiled_allocation_shape(&spec.lhs_shape)?;
    let rhs_allocation = dram::tiled_allocation_shape(&spec.rhs_shape)?;
    let output_allocation = dram::tiled_allocation_shape(&spec.output_shape)?;
    let lhs_tile_rows = lhs_allocation[2] / TILE_R;
    let output_tile_rows = output_allocation[3] / TILE_R;
    if lhs_tile_rows != 1 || output_tile_rows != 1 {
        return Ok(None);
    }

    let kt = rhs_allocation[4] / TILE_C;
    let output_tile_count = dram::tiled_shape_tile_count(&spec.output_shape)?;
    let head_tiles = lhs_allocation[3] / TILE_C;
    let work_tile_count = batch
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_tiles))
        .and_then(|value| value.checked_mul(output_allocation[4] / TILE_C))
        .ok_or_else(|| invalid_input("dpa value work tile count overflow"))?;
    let shape = DpaValueShape {
        work_tile_count: u32_value(work_tile_count, "dpa value work tile count")?,
        batch: u32_value(batch, "dpa value batch")?,
        key_tokens: u32_value(key_tokens, "dpa value key tokens")?,
        query_tokens: u32_value(query_tokens, "dpa value query tokens")?,
        kv_heads: u32_value(kv_heads, "dpa value kv heads")?,
        groups: u32_value(groups, "dpa value groups")?,
        head_dim: u32_value(head_dim, "dpa value head dim")?,
        head_tiles: u32_value(head_tiles, "dpa value head tiles")?,
        lhs_tiles_per_prefix: u32_value(
            lhs_tile_rows * (lhs_allocation[3] / TILE_C),
            "dpa value lhs tiles per prefix",
        )?,
        rhs_tile_rows: u32_value(rhs_allocation[3] / TILE_R, "dpa value rhs tile rows")?,
        rhs_tiles_per_row: u32_value(rhs_allocation[4] / TILE_C, "dpa value rhs tiles per row")?,
        output_tiles_per_row: u32_value(
            output_allocation[4] / TILE_C,
            "dpa value output tiles per row",
        )?,
        kt: u32_value(kt, "dpa value kt")?,
    };

    let output = device.alloc(
        output_tile_count,
        DType::Float16B,
        &output_allocation,
        name.to_owned(),
    )?;
    let cores = select_worker_cores(device.cores_ref(), work_tile_count)?;
    let kernel = DpaValueKernel {
        lhs_addr: u32_addr(lhs.addr, "lhs address")?,
        rhs_addr: u32_addr(rhs.addr, "rhs address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: DpaValueProgramKey { cores, shape },
    };
    kernel.run(device)?;
    Ok(Some(output))
}

fn dpa_score_program(key: &DpaScoreProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_LHS_ADDR_INDEX, READER_RHS_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.output_tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            dpa_score_reader_args(&key.shape, offset, n_tiles),
            vec![key.shape.kt, n_tiles],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: DPA_SCORE_READER.to_owned(),
        writer_kernel: WRITER.to_owned(),
        compute_kernel: MATMUL_TILE_COMPUTE.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(2, DType::Float16B),
                CBConfig {
                    index: 3,
                    dtype: DType::Float16B,
                    tiles: usize::try_from(key.shape.kt)
                        .map_err(|_| invalid_input("dpa score kt does not fit in usize"))?,
                },
                CBConfig::new(16, DType::Float32),
            ],
            dst_accum_mode: true,
            ..CompileConfig::default()
        },
        name: format!(
            "dpa_score_dot_bf16_f32_t{}_s{}_kv{}_g{}_h{}",
            key.shape.query_tokens,
            key.shape.key_tokens,
            key.shape.kv_heads,
            key.shape.groups,
            key.shape.head_dim
        ),
        ..Program::new(runtime_args)
    })
}

fn dpa_value_program(key: &DpaValueProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_LHS_ADDR_INDEX, READER_RHS_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.work_tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            dpa_value_writer_args(&key.shape, offset, n_tiles),
            dpa_value_reader_args(&key.shape, offset, n_tiles),
            vec![
                key.shape.kt,
                n_tiles
                    .checked_mul(key.shape.groups)
                    .ok_or_else(|| invalid_input("dpa value compute tile count overflow"))?,
            ],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: DPA_VALUE_READER.to_owned(),
        writer_kernel: DPA_VALUE_WRITER.to_owned(),
        compute_kernel: MATMUL_TILE_COMPUTE.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(2, DType::Float16B),
                CBConfig {
                    index: 3,
                    dtype: DType::Float16B,
                    tiles: usize::try_from(key.shape.kt)
                        .map_err(|_| invalid_input("dpa value kt does not fit in usize"))?,
                },
                CBConfig {
                    index: 16,
                    dtype: DType::Float16B,
                    tiles: usize::try_from(key.shape.groups.max(2))
                        .map_err(|_| invalid_input("dpa value groups does not fit in usize"))?,
                },
                CBConfig::new(17, DType::Float16B),
            ],
            ..CompileConfig::default()
        },
        name: format!(
            "dpa_value_dot_bf16_t{}_s{}_kv{}_g{}_h{}",
            key.shape.query_tokens,
            key.shape.key_tokens,
            key.shape.kv_heads,
            key.shape.groups,
            key.shape.head_dim
        ),
        ..Program::new(runtime_args)
    })
}

fn dpa_score_reader_args(shape: &DpaScoreShape, offset: u32, n_tiles: u32) -> Vec<u32> {
    vec![
        0,
        0,
        offset,
        n_tiles,
        shape.query_tokens,
        shape.key_tokens,
        shape.kv_heads,
        shape.groups,
        shape.head_dim,
        shape.lhs_tiles_per_prefix,
        shape.rhs_tiles_per_prefix,
        shape.output_tiles_per_row,
        shape.kt,
    ]
}

fn dpa_value_reader_args(shape: &DpaValueShape, offset: u32, n_tiles: u32) -> Vec<u32> {
    vec![
        0,
        0,
        offset,
        n_tiles,
        shape.key_tokens,
        shape.query_tokens,
        shape.kv_heads,
        shape.groups,
        shape.head_dim,
        shape.head_tiles,
        shape.lhs_tiles_per_prefix,
        shape.rhs_tile_rows,
        shape.rhs_tiles_per_row,
        shape.output_tiles_per_row,
        shape.kt,
        shape.batch,
    ]
}

fn dpa_value_writer_args(shape: &DpaValueShape, offset: u32, n_tiles: u32) -> Vec<u32> {
    vec![
        0,
        offset,
        n_tiles,
        shape.groups,
        shape.query_tokens,
        shape.kv_heads,
        shape.head_dim,
        shape.head_tiles,
        shape.output_tiles_per_row,
    ]
}

fn validate_buffer(buffer: &DramBuffer, logical_shape: &[usize], label: &str) -> io::Result<()> {
    let expected_shape = dram::tiled_allocation_shape(logical_shape)?;
    if buffer.shape != expected_shape {
        return Err(invalid_input(format!(
            "{label} allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            buffer.shape, expected_shape, logical_shape
        )));
    }
    let expected_tiles = dram::tiled_shape_tile_count(logical_shape)?;
    if buffer.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "{label} tile count mismatch: got {}, expected {expected_tiles}",
            buffer.num_tiles
        )));
    }
    Ok(())
}

fn validate_dtype(dtype: DType, label: &str) -> io::Result<()> {
    if matches!(dtype, DType::Float16B | DType::Float32) {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "{label} currently supports Float16B and Float32, got {dtype:?}"
        )))
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_value(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_input(format!("{name} does not fit in u32: {value}")))
}

fn u32_addr(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}
