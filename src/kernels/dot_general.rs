use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{self, DType, DramBuffer, TILE_C, TILE_R};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{
    select_worker_cores, split_tile_range, DramKernel, Kernel, RuntimeArgsBuilder,
};
use std::io;

const DPA_SCORE_READER: &str = include_str!("../../kernels/dpa_score_reader.cc");
const DPA_VALUE_READER: &str = include_str!("../../kernels/dpa_value_reader.cc");
const DPA_VALUE_WRITER: &str = include_str!("../../kernels/dpa_value_writer.cc");
const DPA_COMMON: &str = include_str!("../../kernels/dpa_common.cc");
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
    head_dim: u32,
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
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DpaProgramKey<S> {
    cores: Vec<CoreCoord>,
    shape: S,
}

type DpaScoreProgramKey = DpaProgramKey<DpaScoreShape>;
type DpaValueProgramKey = DpaProgramKey<DpaValueShape>;

pub(crate) fn dot_general(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    spec: DotGeneralSpec,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let output_name = name.into();
    if let Some(output) = try_dpa_score_dot(device, lhs, rhs, &spec, output_name.as_str())? {
        return Ok(output);
    }
    if let Some(output) = try_dpa_value_dot(device, lhs, rhs, &spec, output_name.as_str())? {
        return Ok(output);
    }
    Err(invalid_input("unsupported dot_general shape"))
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

    let output_allocation = dram::tiled_allocation_shape(&spec.output_shape)?;
    let output_tile_count = dram::tiled_shape_tile_count(&spec.output_shape)?;
    let shape = DpaScoreShape {
        output_tile_count: u32_value(output_tile_count)?,
        query_tokens: u32_value(query_tokens)?,
        key_tokens: u32_value(key_tokens)?,
        kv_heads: u32_value(kv_heads)?,
        head_dim: u32_value(head_dim)?,
    };

    let output = device.alloc(
        output_tile_count,
        DType::Float32,
        &output_allocation,
        name.to_owned(),
    )?;
    let cores = select_worker_cores(device.cores_ref(), output.num_tiles)?;
    let kernel = DramKernel {
        reader_addrs: [u32_addr(lhs.addr)?, u32_addr(rhs.addr)?],
        output_addr: u32_addr(output.addr)?,
        key: DpaScoreProgramKey { cores, shape },
        build: dpa_score_program,
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

    let output_allocation = dram::tiled_allocation_shape(&spec.output_shape)?;
    let output_tile_count = dram::tiled_shape_tile_count(&spec.output_shape)?;
    let head_tiles = head_dim.div_ceil(TILE_C);
    let output_tiles_per_row = query_tokens.div_ceil(TILE_C);
    let work_tile_count = batch
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_tiles))
        .and_then(|value| value.checked_mul(output_tiles_per_row))
        .ok_or_else(|| invalid_input("dpa value work tile count overflow"))?;
    let shape = DpaValueShape {
        work_tile_count: u32_value(work_tile_count)?,
        batch: u32_value(batch)?,
        key_tokens: u32_value(key_tokens)?,
        query_tokens: u32_value(query_tokens)?,
        kv_heads: u32_value(kv_heads)?,
        groups: u32_value(groups)?,
        head_dim: u32_value(head_dim)?,
    };

    let output = device.alloc(
        output_tile_count,
        DType::Float16B,
        &output_allocation,
        name.to_owned(),
    )?;
    let cores = select_worker_cores(device.cores_ref(), work_tile_count)?;
    let kernel = DramKernel {
        reader_addrs: [u32_addr(lhs.addr)?, u32_addr(rhs.addr)?],
        output_addr: u32_addr(output.addr)?,
        key: DpaValueProgramKey { cores, shape },
        build: dpa_value_program,
    };
    kernel.run(device)?;
    Ok(Some(output))
}

fn dpa_score_program(key: DpaScoreProgramKey) -> io::Result<Program> {
    let kt = key.shape.head_dim.div_ceil(TILE_C as u32);
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
            vec![kt, n_tiles],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: dpa_source(DPA_SCORE_READER),
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
                    tiles: usize::try_from(kt)
                        .map_err(|_| invalid_input("dpa score kt does not fit in usize"))?,
                },
                CBConfig::new(16, DType::Float32),
            ],
            dst_accum_mode: true,
            ..CompileConfig::default()
        },
        name: "dpa_score_dot".to_owned(),
        ..Program::new(runtime_args)
    })
}

fn dpa_value_program(key: DpaValueProgramKey) -> io::Result<Program> {
    let kt = key.shape.key_tokens.div_ceil(TILE_C as u32);
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
                kt,
                n_tiles
                    .checked_mul(key.shape.groups)
                    .ok_or_else(|| invalid_input("dpa value compute tile count overflow"))?,
            ],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: dpa_source(DPA_VALUE_READER),
        writer_kernel: dpa_source(DPA_VALUE_WRITER),
        compute_kernel: MATMUL_TILE_COMPUTE.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(2, DType::Float16B),
                CBConfig {
                    index: 3,
                    dtype: DType::Float16B,
                    tiles: usize::try_from(kt)
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
        name: "dpa_value_dot".to_owned(),
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
        shape.head_dim,
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
    ]
}

fn dpa_source(source: &str) -> String {
    format!("{DPA_COMMON}\n{source}")
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_value(value: usize) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_input("value does not fit in u32"))
}

fn u32_addr(value: u64) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_input("address does not fit in u32"))
}
