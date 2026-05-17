use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::executable::ReduceReducer;
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/reduce_reader.cc");
const GENERAL_READER: &str = include_str!("../../kernels/reduce_general_reader.cc");
const DPA_REDUCE_READER: &str = include_str!("../../kernels/dpa_reduce_reader.cc");
const DPA_REDUCE_WRITER: &str = include_str!("../../kernels/dpa_reduce_writer.cc");
const COMPUTE: &str = include_str!("../../kernels/reduce_compute.cc");
const WRITER: &str = include_str!("../../kernels/reduce_writer.cc");
const SIMPLE_WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;
const MAX_RANK: usize = 8;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ReduceOp {
    Sum,
    Max,
}

impl ReduceOp {
    fn from_reducer(reducer: ReduceReducer) -> io::Result<Self> {
        match reducer {
            ReduceReducer::Add => Ok(Self::Sum),
            ReduceReducer::Max => Ok(Self::Max),
            ReduceReducer::Mul => Err(invalid_input(
                "reduce kernel currently supports add and max reducers",
            )),
        }
    }

    fn cpp_pool_type(self) -> &'static str {
        match self {
            Self::Sum => "ckernel::PoolType::SUM",
            Self::Max => "ckernel::PoolType::MAX",
        }
    }

    fn is_sum(self) -> bool {
        matches!(self, Self::Sum)
    }

    fn padding_identity_bits(self) -> u32 {
        match self {
            Self::Sum => 0.0f32.to_bits(),
            Self::Max => f32::NEG_INFINITY.to_bits(),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum OutputRank {
    One,
    Two,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct ReduceKernelShape {
    input_width_tiles: u32,
    valid_last_width: u32,
    output_tiles: u32,
    inner_output_tiles: u32,
    output_rank: OutputRank,
    output_dim0: u32,
    output_dim1: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReducePlan {
    input_shape: Vec<usize>,
    output_allocation_shape: Vec<usize>,
    shape: ReduceKernelShape,
    op: ReduceOp,
    dtype: DType,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GeneralReduceKernelShape {
    output_rank: u32,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    reduce_dim_size: u32,
    output_rows: u32,
    output_cols: u32,
    output_tiles_per_row: u32,
    output_matrix_tiles: u32,
    output_tile_count: u32,
    input_shape: [u32; MAX_RANK],
    output_shape: [u32; MAX_RANK],
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct DpaReduceShape {
    output_tile_count: u32,
    batch: u32,
    query_tokens: u32,
    kv_heads: u32,
    input_tiles_per_row: u32,
    output_tiles_per_row: u32,
    valid_last_width: u32,
    padding_identity_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GeneralReducePlan {
    input_shape: Vec<usize>,
    output_allocation_shape: Vec<usize>,
    shape: GeneralReduceKernelShape,
    op: ReduceOp,
    dtype: DType,
}

impl ReducePlan {
    pub(crate) fn new(
        dtype: DType,
        input_shape: &[usize],
        output_shape: &[usize],
        dimensions: &[i64],
        reducer: ReduceReducer,
    ) -> io::Result<Self> {
        if dtype != DType::Float32 {
            return Err(invalid_input(format!(
                "reduce kernel currently supports Float32 inputs, got {dtype:?}"
            )));
        }
        if input_shape.len() < 2 {
            return Err(invalid_input(format!(
                "reduce kernel requires rank >= 2 input, got {input_shape:?}"
            )));
        }
        if output_shape.len() > 2 {
            return Err(invalid_input(format!(
                "reduce kernel currently supports rank <= 2 outputs, got {output_shape:?}"
            )));
        }

        let reduce_dim = input_shape.len() - 1;
        if dimensions != [reduce_dim as i64] {
            return Err(invalid_input(format!(
                "reduce kernel currently supports only the last dimension, got dimensions {dimensions:?} for shape {input_shape:?}"
            )));
        }
        let expected_output = &input_shape[..input_shape.len() - 1];
        if output_shape != expected_output {
            return Err(invalid_input(format!(
                "reduce output shape mismatch: expected {:?}, got {:?}",
                expected_output, output_shape
            )));
        }

        let input_allocation_shape = tiled_allocation_shape(input_shape)?;
        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let rank = input_allocation_shape.len();
        let input_width_tiles = input_allocation_shape[rank - 1] / TILE_C;
        let valid_last_width = valid_last_tile_width(input_shape[input_shape.len() - 1])?;
        let inner_output_tiles = input_allocation_shape[rank - 2] / TILE_R;
        let output_tiles = tiled_shape_tile_count(output_shape)?;
        let output_inner_tiles =
            output_allocation_shape[output_allocation_shape.len() - 1] / TILE_C;
        debug_assert_eq!(inner_output_tiles, output_inner_tiles);
        let (output_rank, output_dim0, output_dim1) = match output_shape {
            [dim] => (OutputRank::One, 1, *dim),
            [dim0, dim1] => (OutputRank::Two, *dim0, *dim1),
            _ => {
                return Err(invalid_input(format!(
                    "reduce kernel currently supports rank 1 or 2 outputs, got {output_shape:?}"
                )))
            }
        };

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            shape: ReduceKernelShape {
                input_width_tiles: u32_arg(input_width_tiles, "input width tile count")?,
                valid_last_width,
                output_tiles: u32_arg(output_tiles, "output tile count")?,
                inner_output_tiles: u32_arg(inner_output_tiles, "inner output tile count")?,
                output_rank,
                output_dim0: u32_arg(output_dim0, "output dim0")?,
                output_dim1: u32_arg(output_dim1, "output dim1")?,
            },
            op: ReduceOp::from_reducer(reducer)?,
            dtype,
        })
    }
}

impl GeneralReducePlan {
    pub(crate) fn new(
        dtype: DType,
        input_shape: &[usize],
        output_shape: &[usize],
        dimensions: &[i64],
        reducer: ReduceReducer,
    ) -> io::Result<Self> {
        if dtype != DType::Float32 {
            return Err(invalid_input(format!(
                "general reduce kernel currently supports Float32 inputs, got {dtype:?}"
            )));
        }
        if !(3..=MAX_RANK).contains(&output_shape.len()) {
            return Err(invalid_input(format!(
                "general reduce output rank must be in 3..={MAX_RANK}, got {output_shape:?}"
            )));
        }
        if input_shape.len() != output_shape.len() + 1 {
            return Err(invalid_input(format!(
                "general reduce expects input rank to be output rank + 1, got input={input_shape:?} output={output_shape:?}"
            )));
        }
        let reduce_dim = input_shape.len() - 1;
        if dimensions != [reduce_dim as i64] {
            return Err(invalid_input(format!(
                "general reduce currently supports only the last dimension, got dimensions {dimensions:?} for shape {input_shape:?}"
            )));
        }
        let expected_output = &input_shape[..input_shape.len() - 1];
        if output_shape != expected_output {
            return Err(invalid_input(format!(
                "general reduce output shape mismatch: expected {:?}, got {:?}",
                expected_output, output_shape
            )));
        }
        if input_shape.contains(&0) || output_shape.contains(&0) {
            return Err(invalid_input(
                "general reduce zero-sized dimensions are not currently supported",
            ));
        }

        let input_allocation_shape = tiled_allocation_shape(input_shape)?;
        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let output_rank = output_shape.len();
        let output_tiles_per_row = output_allocation_shape[output_rank - 1] / TILE_C;
        let output_tile_rows = output_allocation_shape[output_rank - 2] / TILE_R;
        let output_matrix_tiles = output_tile_rows
            .checked_mul(output_tiles_per_row)
            .ok_or_else(|| invalid_input("general reduce output matrix tile count overflow"))?;
        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            shape: GeneralReduceKernelShape {
                output_rank: u32_arg(output_rank, "output rank")?,
                input_tile_rows: u32_arg(
                    input_allocation_shape[input_shape.len() - 2] / TILE_R,
                    "input tile rows",
                )?,
                input_tiles_per_row: u32_arg(
                    input_allocation_shape[input_shape.len() - 1] / TILE_C,
                    "input tiles per row",
                )?,
                reduce_dim_size: u32_arg(input_shape[input_shape.len() - 1], "reduce dim size")?,
                output_rows: u32_arg(output_shape[output_rank - 2], "output rows")?,
                output_cols: u32_arg(output_shape[output_rank - 1], "output cols")?,
                output_tiles_per_row: u32_arg(output_tiles_per_row, "output tiles per row")?,
                output_matrix_tiles: u32_arg(output_matrix_tiles, "output matrix tiles")?,
                output_tile_count: u32_arg(
                    tiled_shape_tile_count(output_shape)?,
                    "output tile count",
                )?,
                input_shape: padded_array(input_shape, "input shape")?,
                output_shape: padded_array(output_shape, "output shape")?,
            },
            op: ReduceOp::from_reducer(reducer)?,
            dtype,
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReduceProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    op: ReduceOp,
    shape: ReduceKernelShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReduceCoreRange {
    group_offset: u32,
    reduce_groups: u32,
    output_tile_offset: u32,
    output_tiles: u32,
}

struct ReduceKernel {
    input_addr: u32,
    output_addr: u32,
    key: ReduceProgramKey,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct GeneralReduceProgramKey {
    cores: Vec<CoreCoord>,
    op: ReduceOp,
    shape: GeneralReduceKernelShape,
}

struct GeneralReduceKernel {
    input_addr: u32,
    output_addr: u32,
    key: GeneralReduceProgramKey,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DpaReduceProgramKey {
    cores: Vec<CoreCoord>,
    op: ReduceOp,
    shape: DpaReduceShape,
}

struct DpaReduceKernel {
    input_addr: u32,
    output_addr: u32,
    key: DpaReduceProgramKey,
}

impl Kernel<ReduceProgramKey> for ReduceKernel {
    fn program_key(&self) -> ReduceProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        reduce_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_INPUT_ADDR_INDEX => Some(self.input_addr),
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

impl Kernel<GeneralReduceProgramKey> for GeneralReduceKernel {
    fn program_key(&self) -> GeneralReduceProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        general_reduce_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_INPUT_ADDR_INDEX => Some(self.input_addr),
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

impl Kernel<DpaReduceProgramKey> for DpaReduceKernel {
    fn program_key(&self) -> DpaReduceProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        dpa_reduce_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_INPUT_ADDR_INDEX => Some(self.input_addr),
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

pub(crate) fn reduce(
    device: &mut Device,
    input: &DramBuffer,
    plan: &ReducePlan,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_input(input, plan)?;
    let output_tiles = usize::try_from(plan.shape.output_tiles).map_err(|_| {
        invalid_input(format!(
            "output tile count does not fit in usize: {}",
            plan.shape.output_tiles
        ))
    })?;
    let partition_count = usize::try_from(reduce_partition_count(plan.shape)?).map_err(|_| {
        invalid_input(format!(
            "reduce partition count does not fit in usize for shape {:?}",
            plan.shape
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), partition_count)?;
    let output = device.alloc(
        output_tiles,
        plan.dtype,
        &plan.output_allocation_shape,
        name,
    )?;
    let kernel = ReduceKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: ReduceProgramKey {
            cores,
            dtype: plan.dtype,
            op: plan.op,
            shape: plan.shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

pub(crate) fn reduce_general(
    device: &mut Device,
    input: &DramBuffer,
    plan: &GeneralReducePlan,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let output_name = name.into();
    validate_general_input(input, plan)?;
    if let Some(output) = try_dpa_attention_reduce(device, input, plan, output_name.as_str())? {
        return Ok(output);
    }
    let output_tiles = usize::try_from(plan.shape.output_tile_count).map_err(|_| {
        invalid_input(format!(
            "output tile count does not fit in usize: {}",
            plan.shape.output_tile_count
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(
        output_tiles,
        plan.dtype,
        &plan.output_allocation_shape,
        output_name,
    )?;
    let kernel = GeneralReduceKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: GeneralReduceProgramKey {
            cores,
            op: plan.op,
            shape: plan.shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn try_dpa_attention_reduce(
    device: &mut Device,
    input: &DramBuffer,
    plan: &GeneralReducePlan,
    name: &str,
) -> io::Result<Option<DramBuffer>> {
    let Some(shape) = dpa_reduce_shape_from_plan(plan)? else {
        return Ok(None);
    };
    let output_tiles = usize::try_from(shape.output_tile_count).map_err(|_| {
        invalid_input(format!(
            "output tile count does not fit in usize: {}",
            shape.output_tile_count
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(
        output_tiles,
        plan.dtype,
        &plan.output_allocation_shape,
        name.to_owned(),
    )?;
    let kernel = DpaReduceKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: DpaReduceProgramKey {
            cores,
            op: plan.op,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(Some(output))
}

fn dpa_reduce_shape_from_plan(plan: &GeneralReducePlan) -> io::Result<Option<DpaReduceShape>> {
    let input_shape = plan.input_shape.as_slice();
    if plan.dtype != DType::Float32 || input_shape.len() != 5 || plan.shape.output_rank != 4 {
        return Ok(None);
    }

    let batch = input_shape[1];
    let kv_heads = input_shape[2];
    let query_tokens = input_shape[3];
    if kv_heads > TILE_R
        || plan.shape.output_matrix_tiles != plan.shape.output_tiles_per_row
        || plan.shape.input_tile_rows != plan.shape.output_tiles_per_row
        || plan.shape.output_rows as usize != kv_heads
        || plan.shape.output_cols as usize != query_tokens
    {
        return Ok(None);
    }

    Ok(Some(DpaReduceShape {
        output_tile_count: plan.shape.output_tile_count,
        batch: u32_arg(batch, "dpa reduce batch")?,
        query_tokens: u32_arg(query_tokens, "dpa reduce query tokens")?,
        kv_heads: u32_arg(kv_heads, "dpa reduce kv heads")?,
        input_tiles_per_row: plan.shape.input_tiles_per_row,
        output_tiles_per_row: plan.shape.output_tiles_per_row,
        valid_last_width: valid_last_tile_width(input_shape[4])?,
        padding_identity_bits: plan.op.padding_identity_bits(),
    }))
}

fn validate_input(input: &DramBuffer, plan: &ReducePlan) -> io::Result<()> {
    if input.dtype != plan.dtype {
        return Err(invalid_input(format!(
            "reduce input requires {:?}, got {:?}",
            plan.dtype, input.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "reduce input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, plan.input_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "reduce input tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn validate_general_input(input: &DramBuffer, plan: &GeneralReducePlan) -> io::Result<()> {
    if input.dtype != plan.dtype {
        return Err(invalid_input(format!(
            "general reduce input requires {:?}, got {:?}",
            plan.dtype, input.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "general reduce input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, plan.input_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "general reduce input tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn reduce_program(key: ReduceProgramKey) -> io::Result<Program> {
    let shape = key.shape;
    let ranges = reduce_core_ranges(shape, key.cores.len())?;
    let max_core_output_tiles = ranges
        .iter()
        .map(|range| range.output_tiles)
        .max()
        .unwrap_or(1);
    let output_tiles = usize::try_from(max_core_output_tiles).map_err(|_| {
        invalid_input(format!(
            "per-core output tile count does not fit in usize: {max_core_output_tiles}"
        ))
    })?;
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX],
        Vec::new(),
    );
    for (&core, range) in key.cores.iter().zip(ranges.iter()) {
        runtime_args.add_core(
            core,
            vec![
                0,
                range.group_offset,
                range.reduce_groups,
                shape.inner_output_tiles,
                range.output_tile_offset,
                range.output_tiles,
                shape.output_dim0,
                shape.output_dim1,
            ],
            vec![
                0,
                range.group_offset,
                range.reduce_groups,
                shape.input_width_tiles,
                shape.valid_last_width,
                key.op.padding_identity_bits(),
            ],
            vec![range.reduce_groups, shape.input_width_tiles],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: READER.to_owned(),
        compute_kernel: reduce_compute_source(key.op),
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.dtype),
                CBConfig::new(16, key.dtype),
                CBConfig {
                    index: 17,
                    dtype: key.dtype,
                    tiles: output_tiles,
                },
            ],
            dst_accum_mode: true,
            ..CompileConfig::default()
        },
        name: format!("reduce_{:?}_{:?}", key.op, key.dtype),
        ..Program::new(runtime_args)
    })
}

fn general_reduce_program(key: GeneralReduceProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.output_tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            general_reader_args(&key.shape, offset, n_tiles),
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: general_reduce_reader_source(key.op),
        writer_kernel: SIMPLE_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float32),
                CBConfig::new(16, DType::Float32),
            ],
            ..CompileConfig::default()
        },
        name: format!("reduce_general_{:?}", key.op),
        ..Program::new(runtime_args)
    })
}

fn dpa_reduce_program(key: DpaReduceProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.output_tile_count, core_index, key.cores.len())?;
        let reduce_groups = n_tiles
            .checked_mul(TILE_R as u32)
            .ok_or_else(|| invalid_input("dpa reduce group count overflow"))?;
        runtime_args.add_core(
            core,
            dpa_reduce_writer_args(&key.shape, offset, n_tiles),
            dpa_reduce_reader_args(&key.shape, offset, n_tiles),
            vec![reduce_groups, key.shape.input_tiles_per_row],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: DPA_REDUCE_READER.to_owned(),
        compute_kernel: reduce_compute_source(key.op),
        writer_kernel: DPA_REDUCE_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float32),
                CBConfig::new(16, DType::Float32),
                CBConfig::new(17, DType::Float32),
            ],
            dst_accum_mode: true,
            ..CompileConfig::default()
        },
        name: format!(
            "dpa_reduce_{:?}_t{}_kv{}",
            key.op, key.shape.query_tokens, key.shape.kv_heads
        ),
        ..Program::new(runtime_args)
    })
}

fn general_reader_args(shape: &GeneralReduceKernelShape, offset: u32, n_tiles: u32) -> Vec<u32> {
    let mut args = vec![
        0,
        offset,
        n_tiles,
        shape.output_rank,
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.reduce_dim_size,
        shape.output_rows,
        shape.output_cols,
        shape.output_tiles_per_row,
        shape.output_matrix_tiles,
    ];
    args.extend(shape.output_shape);
    args.extend(shape.input_shape);
    args
}

fn dpa_reduce_reader_args(shape: &DpaReduceShape, offset: u32, n_tiles: u32) -> Vec<u32> {
    vec![
        0,
        offset,
        n_tiles,
        shape.query_tokens,
        shape.batch,
        shape.kv_heads,
        shape.input_tiles_per_row,
        shape.output_tiles_per_row,
        shape.valid_last_width,
        shape.padding_identity_bits,
    ]
}

fn dpa_reduce_writer_args(shape: &DpaReduceShape, offset: u32, n_tiles: u32) -> Vec<u32> {
    vec![
        0,
        offset,
        n_tiles,
        shape.query_tokens,
        shape.kv_heads,
        shape.output_tiles_per_row,
    ]
}

fn general_reduce_reader_source(op: ReduceOp) -> String {
    format!(
        "#define REDUCE_GENERAL_MAX_RANK {MAX_RANK}\n#define REDUCE_GENERAL_IS_SUM {}\n{GENERAL_READER}",
        bool_define(op.is_sum())
    )
}

fn reduce_compute_source(op: ReduceOp) -> String {
    COMPUTE
        .replace("REDUCE_POOL_TYPE", op.cpp_pool_type())
        .replace("REDUCE_IS_SUM", bool_define(op.is_sum()))
}

fn bool_define(value: bool) -> &'static str {
    if value {
        "1"
    } else {
        "0"
    }
}

fn reduce_partition_count(shape: ReduceKernelShape) -> io::Result<u32> {
    match shape.output_rank {
        OutputRank::One => Ok(shape.output_tiles),
        OutputRank::Two => output_tile_rows(shape),
    }
}

fn reduce_core_ranges(
    shape: ReduceKernelShape,
    core_count: usize,
) -> io::Result<Vec<ReduceCoreRange>> {
    let partition_count = reduce_partition_count(shape)?;
    (0..core_count)
        .map(|core_index| {
            let (partition_offset, partitions) =
                split_tile_range(partition_count, core_index, core_count)?;
            reduce_core_range(shape, partition_offset, partitions)
        })
        .collect()
}

fn reduce_core_range(
    shape: ReduceKernelShape,
    partition_offset: u32,
    partitions: u32,
) -> io::Result<ReduceCoreRange> {
    match shape.output_rank {
        OutputRank::One => Ok(ReduceCoreRange {
            group_offset: partition_offset,
            reduce_groups: partitions,
            output_tile_offset: partition_offset,
            output_tiles: partitions,
        }),
        OutputRank::Two => reduce_matrix_core_range(shape, partition_offset, partitions),
    }
}

fn reduce_matrix_core_range(
    shape: ReduceKernelShape,
    tile_row_offset: u32,
    tile_rows: u32,
) -> io::Result<ReduceCoreRange> {
    let output_tile_offset = checked_mul_u32(
        tile_row_offset,
        shape.inner_output_tiles,
        "output tile offset",
    )?;
    let output_tiles = checked_mul_u32(tile_rows, shape.inner_output_tiles, "output tiles")?;
    let output_row_offset = checked_mul_u32(tile_row_offset, TILE_R as u32, "output row offset")?;
    let max_rows = checked_mul_u32(tile_rows, TILE_R as u32, "output rows")?;
    let output_rows = shape
        .output_dim0
        .saturating_sub(output_row_offset)
        .min(max_rows);
    let group_offset = checked_mul_u32(
        output_row_offset,
        shape.inner_output_tiles,
        "reduce group offset",
    )?;
    let reduce_groups =
        checked_mul_u32(output_rows, shape.inner_output_tiles, "reduce group count")?;
    Ok(ReduceCoreRange {
        group_offset,
        reduce_groups,
        output_tile_offset,
        output_tiles,
    })
}

fn output_tile_rows(shape: ReduceKernelShape) -> io::Result<u32> {
    if shape.inner_output_tiles == 0 {
        return Err(invalid_input("inner output tile count must be nonzero"));
    }
    Ok(shape.output_tiles / shape.inner_output_tiles)
}

fn valid_last_tile_width(logical_width: usize) -> io::Result<u32> {
    let width = logical_width % TILE_C;
    let width = if width == 0 && logical_width != 0 {
        TILE_C
    } else {
        width
    };
    u32_arg(width, "valid last reduction tile width")
}

fn padded_array(values: &[usize], label: &str) -> io::Result<[u32; MAX_RANK]> {
    if values.len() > MAX_RANK {
        return Err(invalid_input(format!(
            "general reduce {label} rank exceeds {MAX_RANK}: {}",
            values.len()
        )));
    }
    let mut out = [0u32; MAX_RANK];
    for (index, &value) in values.iter().enumerate() {
        out[index] = u32_arg(value, label)?;
    }
    Ok(out)
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_input(format!("{name} does not fit in u32: {value}")))
}

fn u32_addr(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}

fn checked_mul_u32(lhs: u32, rhs: u32, name: &str) -> io::Result<u32> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| invalid_input(format!("{name} overflow: {lhs} * {rhs}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_plan_tracks_partial_last_width_tile() {
        let plan =
            ReducePlan::new(DType::Float32, &[2, 30], &[2], &[1], ReduceReducer::Max).unwrap();
        assert_eq!(plan.shape.input_width_tiles, 1);
        assert_eq!(plan.shape.valid_last_width, 30);
        assert_eq!(plan.op.padding_identity_bits(), f32::NEG_INFINITY.to_bits());
    }

    #[test]
    fn reduce_plan_keeps_aligned_last_width_tile_unmasked() {
        let plan =
            ReducePlan::new(DType::Float32, &[2, 64], &[2], &[1], ReduceReducer::Add).unwrap();
        assert_eq!(plan.shape.input_width_tiles, 2);
        assert_eq!(plan.shape.valid_last_width, TILE_C as u32);
        assert_eq!(plan.op.padding_identity_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn general_reduce_plan_supports_rank4_output() {
        let plan = GeneralReducePlan::new(
            DType::Float32,
            &[2, 1, 8, 27, 27],
            &[2, 1, 8, 27],
            &[4],
            ReduceReducer::Max,
        )
        .unwrap();

        assert_eq!(plan.shape.output_rank, 4);
        assert_eq!(plan.shape.reduce_dim_size, 27);
        assert_eq!(plan.shape.output_tile_count, 2);
        assert_eq!(plan.op.padding_identity_bits(), f32::NEG_INFINITY.to_bits());
    }

    #[test]
    fn dpa_reduce_shape_matches_dot_product_attention_layout() {
        let plan = GeneralReducePlan::new(
            DType::Float32,
            &[2, 1, 8, 65, 65],
            &[2, 1, 8, 65],
            &[4],
            ReduceReducer::Max,
        )
        .unwrap();
        let shape = dpa_reduce_shape_from_plan(&plan).unwrap().unwrap();

        assert_eq!(shape.batch, 1);
        assert_eq!(shape.kv_heads, 8);
        assert_eq!(shape.query_tokens, 65);
        assert_eq!(shape.input_tiles_per_row, 3);
        assert_eq!(shape.output_tiles_per_row, 3);
        assert_eq!(shape.output_tile_count, 6);
        assert_eq!(shape.valid_last_width, 1);
    }

    #[test]
    fn dpa_reduce_shape_rejects_multi_tile_kv_heads() {
        let plan = GeneralReducePlan::new(
            DType::Float32,
            &[2, 1, 33, 65, 65],
            &[2, 1, 33, 65],
            &[4],
            ReduceReducer::Max,
        )
        .unwrap();

        assert!(dpa_reduce_shape_from_plan(&plan).unwrap().is_none());
    }
}
