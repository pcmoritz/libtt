use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::executable::ReduceReducer;
use crate::hw::CoreCoord;
use crate::kernels::kernel::{
    select_worker_cores, split_tile_range, DramKernel, Kernel, RuntimeArgsBuilder,
};
use std::io;

const READER: &str = include_str!("../../kernels/reduce_reader.cc");
const COMPUTE: &str = include_str!("../../kernels/reduce_compute.cc");
const WRITER: &str = include_str!("../../kernels/reduce_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ReduceOp {
    Sum,
    Max,
    Min,
    And,
    Or,
}

impl ReduceOp {
    fn from_reducer(reducer: ReduceReducer) -> io::Result<Self> {
        match reducer {
            ReduceReducer::Add => Ok(Self::Sum),
            ReduceReducer::Max => Ok(Self::Max),
            ReduceReducer::Min => Ok(Self::Min),
            ReduceReducer::And => Ok(Self::And),
            ReduceReducer::Or => Ok(Self::Or),
            ReduceReducer::Mul => Err(invalid_input(
                "reduce kernel currently supports add, min, max, and bitwise and/or reducers",
            )),
        }
    }

    fn is_bitwise(self) -> bool {
        matches!(self, Self::And | Self::Or)
    }

    fn is_sum(self) -> bool {
        matches!(self, Self::Sum)
    }

    fn cpp_pool_type(self) -> &'static str {
        match self {
            Self::Sum => "ckernel::PoolType::SUM",
            Self::Max => "ckernel::PoolType::MAX",
            Self::Min => "ckernel::PoolType::MIN",
            Self::And | Self::Or => "ckernel::PoolType::SUM",
        }
    }

    fn padding_identity_bits(self) -> u32 {
        match self {
            Self::Sum => 0.0f32.to_bits(),
            Self::Max => f32::NEG_INFINITY.to_bits(),
            Self::Min => f32::INFINITY.to_bits(),
            Self::And | Self::Or => 0,
        }
    }

    fn is_min(self) -> bool {
        matches!(self, Self::Min)
    }

    fn identity_literal(self, dtype: DType) -> io::Result<&'static str> {
        match (self, dtype) {
            (Self::Sum, DType::Float32) => Ok("0.0f"),
            (Self::Sum, _) => Ok("0"),
            (Self::Max, DType::Float32) => Ok("(-3.4028234663852886e+38F)"),
            (Self::Min, DType::Float32) => Ok("3.4028234663852886e+38F"),
            (Self::Max, DType::Int32) => Ok("(-2147483647 - 1)"),
            (Self::Min, DType::Int32) => Ok("2147483647"),
            (Self::Max, DType::Float16B) => Ok("0xff80u"),
            (Self::Min, DType::Float16B) => Ok("0x7f80u"),
            (Self::Max, DType::UInt32 | DType::UInt16) => Ok("0"),
            (Self::Min, DType::UInt32) => Ok("0xffffffffu"),
            (Self::Min, DType::UInt16) => Ok("0xffffu"),
            (Self::And | Self::Or, _) => Err(invalid_input(
                "bitwise reduce does not use arithmetic identity literals",
            )),
            _ => Err(invalid_input(format!(
                "reduce kernel does not support {:?} with dtype {dtype:?}",
                self
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct ReduceKernelShape {
    reduce_dim: u32,
    reduce_count: u32,
    valid_last_width: u32,
    output_tiles: u32,
    inner_output_tiles: u32,
    output_tile_rows_per_prefix: u32,
    output_dim0: u32,
    output_dim1: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReducePlan {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    output_allocation_shape: Vec<usize>,
    shape: ReduceKernelShape,
    op: ReduceOp,
    dtype: DType,
    identity: Option<u32>,
}

impl ReducePlan {
    pub(crate) fn new(
        dtype: DType,
        input_shape: &[usize],
        output_shape: &[usize],
        dimensions: &[i64],
        reducer: ReduceReducer,
        identity: Option<u32>,
    ) -> io::Result<Self> {
        let op = ReduceOp::from_reducer(reducer)?;
        validate_reduce_dtype(dtype, op)?;
        if input_shape.len() < 2 {
            return Err(invalid_input(format!(
                "reduce kernel requires rank >= 2 input, got {input_shape:?}"
            )));
        }
        if dimensions.len() != 1 {
            return Err(invalid_input(format!(
                "reduce kernel currently supports exactly one reduction dimension, got {dimensions:?}"
            )));
        }
        let reduce_dim = usize::try_from(dimensions[0]).map_err(|_| {
            invalid_input(format!(
                "reduce dimension must be nonnegative, got {}",
                dimensions[0]
            ))
        })?;
        if reduce_dim >= input_shape.len() {
            return Err(invalid_input(format!(
                "reduce dimension {reduce_dim} is out of range for shape {input_shape:?}"
            )));
        }
        let expected_output = input_shape
            .iter()
            .enumerate()
            .filter_map(|(dim, &size)| (dim != reduce_dim).then_some(size))
            .collect::<Vec<_>>();
        if output_shape != expected_output {
            return Err(invalid_input(format!(
                "reduce output shape mismatch: expected {:?}, got {:?}",
                expected_output, output_shape
            )));
        }

        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let reduce_count = input_shape[reduce_dim];
        let valid_last_width = valid_last_tile_width(input_shape[input_shape.len() - 1])?;
        let output_tiles = tiled_shape_tile_count(output_shape)?;
        let output_inner_tiles =
            output_allocation_shape[output_allocation_shape.len() - 1] / TILE_C;
        let inner_output_tiles = output_inner_tiles;
        let (output_dim0, output_dim1, output_tile_rows_per_prefix) = match output_shape {
            [dim] => (1, *dim, 1),
            [] => {
                return Err(invalid_input(
                    "reduce kernel currently requires rank >= 1 output",
                ))
            }
            _ => {
                let rank = output_shape.len();
                (
                    output_shape[rank - 2],
                    output_shape[rank - 1],
                    output_allocation_shape[rank - 2] / TILE_R,
                )
            }
        };
        if output_tile_rows_per_prefix == 0 {
            return Err(invalid_input(format!(
                "reduce output tile rows per prefix must be nonzero for shape {output_shape:?}"
            )));
        };
        if op.is_bitwise() && identity.is_none() {
            return Err(invalid_input("bitwise reduce requires an identity value"));
        }

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_shape: output_shape.to_vec(),
            output_allocation_shape,
            shape: ReduceKernelShape {
                reduce_dim: u32_arg(reduce_dim, "reduce dimension")?,
                reduce_count: u32_arg(reduce_count, "reduce element count")?,
                valid_last_width,
                output_tiles: u32_arg(output_tiles, "output tile count")?,
                inner_output_tiles: u32_arg(inner_output_tiles, "inner output tile count")?,
                output_tile_rows_per_prefix: u32_arg(
                    output_tile_rows_per_prefix,
                    "output tile rows per prefix",
                )?,
                output_dim0: u32_arg(output_dim0, "output dim0")?,
                output_dim1: u32_arg(output_dim1, "output dim1")?,
            },
            op,
            dtype,
            identity,
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReduceProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    op: ReduceOp,
    input_shape: Vec<u32>,
    output_shape: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    shape: ReduceKernelShape,
    identity: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReduceCoreRange {
    group_offset: u32,
    reduce_groups: u32,
    output_tile_offset: u32,
    output_tiles: u32,
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
    let input_allocation_shape = tiled_allocation_shape(&plan.input_shape)?;
    let input_rank = input_allocation_shape.len();
    let kernel = DramKernel {
        reader_addrs: [u32_addr(input.addr, "input address")?],
        output_addr: u32_addr(output.addr, "output address")?,
        key: ReduceProgramKey {
            cores,
            dtype: plan.dtype,
            op: plan.op,
            input_shape: u32_shape(&plan.input_shape, "reduce input shape")?,
            output_shape: u32_shape(&plan.output_shape, "reduce output shape")?,
            input_tile_rows: u32_arg(
                input_allocation_shape[input_rank - 2] / TILE_R,
                "reduce input tile rows",
            )?,
            input_tiles_per_row: u32_arg(
                input_allocation_shape[input_rank - 1] / TILE_C,
                "reduce input tiles per row",
            )?,
            shape: plan.shape,
            identity: plan.identity,
        },
        build: reduce_program,
    };
    kernel.run(device)?;
    Ok(output)
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

fn validate_reduce_dtype(dtype: DType, op: ReduceOp) -> io::Result<()> {
    if op.is_bitwise() {
        if matches!(
            dtype,
            DType::Int32 | DType::UInt32 | DType::UInt16 | DType::UInt8
        ) {
            return Ok(());
        }
        return Err(invalid_input(format!(
            "bitwise reduce does not support dtype {dtype:?}"
        )));
    }
    if matches!(
        dtype,
        DType::Float32 | DType::Float16B | DType::Int32 | DType::UInt32 | DType::UInt16
    ) {
        return Ok(());
    }
    Err(invalid_input(format!(
        "reduce kernel currently supports Float32, BF16, UInt16, UInt32, and Int32 inputs, got {dtype:?}"
    )))
}

fn reduce_program(key: ReduceProgramKey) -> io::Result<Program> {
    let shape = key.shape;
    let use_tiled_last_dim = use_tiled_last_dim(&key);
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
        let reader_args = if use_tiled_last_dim {
            vec![
                0,
                range.group_offset,
                range.reduce_groups,
                key.input_tiles_per_row,
                shape.valid_last_width,
                key.op.padding_identity_bits(),
            ]
        } else {
            vec![
                0,
                range.group_offset,
                range.reduce_groups,
                shape.reduce_count,
            ]
        };
        let compute_args = if use_tiled_last_dim {
            vec![range.reduce_groups, key.input_tiles_per_row]
        } else {
            vec![range.reduce_groups, shape.reduce_count]
        };
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
                shape.output_tile_rows_per_prefix,
            ],
            reader_args,
            compute_args,
        )?;
    }
    let runtime_args = runtime_args.build()?;
    let compute_dtype = reduce_compute_dtype(key.dtype)?;
    Ok(Program {
        reader_kernel: reduce_reader_source(&key, use_tiled_last_dim)?,
        compute_kernel: reduce_compute_source(key.op, compute_dtype, use_tiled_last_dim),
        writer_kernel: reduce_writer_source(key.dtype, key.op)?,
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.dtype).with_compute_dtype(compute_dtype),
                CBConfig::new(1, key.dtype),
                CBConfig::new(16, key.dtype).with_compute_dtype(compute_dtype),
                CBConfig {
                    index: 17,
                    dtype: key.dtype,
                    compute_dtype: key.dtype,
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

fn use_tiled_last_dim(key: &ReduceProgramKey) -> bool {
    key.dtype == DType::Float32
        && !key.op.is_min()
        && key.shape.reduce_dim as usize == key.input_shape.len().saturating_sub(1)
}

fn reduce_reader_source(key: &ReduceProgramKey, use_tiled_last_dim: bool) -> io::Result<String> {
    Ok(format!(
        "#define REDUCE_LAST_DIM_TILED {}\n\
         #define REDUCE_RANK {}\n\
         #define REDUCE_DIMENSION {}\n\
         #define REDUCE_INPUT_SHAPE {}\n\
         #define REDUCE_OUTPUT_SHAPE {}\n\
         #define REDUCE_INPUT_TILE_ROWS {}\n\
         #define REDUCE_INPUT_TILES_PER_ROW {}\n\
         #define REDUCE_INNER_OUTPUT_TILES {}\n\
         #define REDUCE_IDENTITY {}\n\
         #define REDUCE_ELEMENT_TYPE {}\n\
         {READER}",
        bool_define(use_tiled_last_dim),
        key.input_shape.len(),
        key.shape.reduce_dim,
        cpp_u32_array(&key.input_shape),
        cpp_u32_array(&key.output_shape),
        key.input_tile_rows,
        key.input_tiles_per_row,
        key.shape.inner_output_tiles,
        reduce_identity_literal(key)?,
        reduce_element_type(key.dtype, key.op)?,
    ))
}

fn reduce_compute_source(op: ReduceOp, compute_dtype: DType, use_tiled_last_dim: bool) -> String {
    COMPUTE
        .replace("REDUCE_LAST_DIM_TILED", bool_define(use_tiled_last_dim))
        .replace("REDUCE_DATA_FORMAT", data_format(compute_dtype))
        .replace("REDUCE_POOL_TYPE", op.cpp_pool_type())
        .replace("REDUCE_IS_SUM", bool_define(op.is_sum()))
        .replace("REDUCE_IS_MIN", bool_define(op.is_min()))
        .replace("REDUCE_IS_OR", bool_define(matches!(op, ReduceOp::Or)))
        .replace("REDUCE_IS_BITWISE", bool_define(op.is_bitwise()))
}

fn reduce_writer_source(dtype: DType, op: ReduceOp) -> io::Result<String> {
    Ok(format!(
        "#define REDUCE_ELEMENT_TYPE {}\n{WRITER}",
        reduce_element_type(dtype, op)?
    ))
}

fn reduce_compute_dtype(dtype: DType) -> io::Result<DType> {
    match dtype {
        DType::Float16B => Ok(DType::Float32),
        DType::Float32 | DType::Int32 | DType::UInt32 | DType::UInt16 | DType::UInt8 => Ok(dtype),
        _ => Err(invalid_input(format!(
            "reduce compute kernel does not support dtype {dtype:?}"
        ))),
    }
}

fn data_format(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "DataFormat::Float32",
        DType::Float16B => "DataFormat::Float16_b",
        DType::Int32 => "DataFormat::Int32",
        DType::UInt32 => "DataFormat::UInt32",
        DType::UInt16 => "DataFormat::UInt16",
        DType::UInt8 => "DataFormat::UInt8",
        _ => "DataFormat::Invalid",
    }
}

fn bool_define(value: bool) -> &'static str {
    if value {
        "1"
    } else {
        "0"
    }
}

fn reduce_partition_count(shape: ReduceKernelShape) -> io::Result<u32> {
    if shape.output_dim0 == 1 && shape.output_tile_rows_per_prefix == 1 {
        Ok(shape.output_tiles)
    } else {
        output_tile_rows(shape)
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
    if shape.output_dim0 == 1 && shape.output_tile_rows_per_prefix == 1 {
        Ok(ReduceCoreRange {
            group_offset: partition_offset,
            reduce_groups: partitions,
            output_tile_offset: partition_offset,
            output_tiles: partitions,
        })
    } else {
        reduce_matrix_core_range(shape, partition_offset, partitions)
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
    let group_row_offset = output_rows_before_tile_row(shape, tile_row_offset)?;
    let group_offset = checked_mul_u32(
        group_row_offset,
        shape.inner_output_tiles,
        "reduce group offset",
    )?;
    let end_tile_row = tile_row_offset
        .checked_add(tile_rows)
        .ok_or_else(|| invalid_input("reduce tile row range overflow"))?;
    let reduce_rows = output_rows_before_tile_row(shape, end_tile_row)?
        .checked_sub(group_row_offset)
        .ok_or_else(|| invalid_input("reduce row count underflow"))?;
    let reduce_groups =
        checked_mul_u32(reduce_rows, shape.inner_output_tiles, "reduce group count")?;
    Ok(ReduceCoreRange {
        group_offset,
        reduce_groups,
        output_tile_offset,
        output_tiles,
    })
}

fn output_rows_before_tile_row(shape: ReduceKernelShape, tile_row: u32) -> io::Result<u32> {
    let tile_rows_per_prefix = shape.output_tile_rows_per_prefix;
    if tile_rows_per_prefix == 0 {
        return Err(invalid_input(
            "matrix reduce requires nonzero output tile rows per prefix",
        ));
    }
    let prefix = tile_row / tile_rows_per_prefix;
    let row_tile = tile_row % tile_rows_per_prefix;
    let prefix_rows = checked_mul_u32(prefix, shape.output_dim0, "reduce prefix row offset")?;
    let row_offset =
        checked_mul_u32(row_tile, TILE_R as u32, "output row offset")?.min(shape.output_dim0);
    prefix_rows
        .checked_add(row_offset)
        .ok_or_else(|| invalid_input("reduce output row offset overflow"))
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

fn u32_shape(shape: &[usize], name: &str) -> io::Result<Vec<u32>> {
    shape
        .iter()
        .enumerate()
        .map(|(index, &dim)| u32_arg(dim, &format!("{name} dimension {index}")))
        .collect()
}

fn cpp_u32_array(values: &[u32]) -> String {
    if values.is_empty() {
        return "{1u}".to_owned();
    }
    let values = values
        .iter()
        .map(|value| format!("{value}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{values}}}")
}

fn reduce_identity_literal(key: &ReduceProgramKey) -> io::Result<String> {
    if key.op.is_bitwise() {
        let identity = key
            .identity
            .ok_or_else(|| invalid_input("bitwise reduce requires an identity value"))?;
        return Ok(format!("{identity}u"));
    }
    Ok(key.op.identity_literal(key.dtype)?.to_owned())
}

fn reduce_element_type(dtype: DType, op: ReduceOp) -> io::Result<&'static str> {
    if op.is_bitwise() {
        return bitwise_element_type(dtype);
    }
    arithmetic_element_type(dtype)
}

fn arithmetic_element_type(dtype: DType) -> io::Result<&'static str> {
    match dtype {
        DType::Float32 => Ok("float"),
        DType::Float16B => Ok("uint16_t"),
        DType::Int32 => Ok("int32_t"),
        DType::UInt32 => Ok("uint32_t"),
        DType::UInt16 => Ok("uint16_t"),
        _ => Err(invalid_input(format!(
            "reduce kernel does not support dtype {dtype:?}"
        ))),
    }
}

fn bitwise_element_type(dtype: DType) -> io::Result<&'static str> {
    match dtype {
        DType::Int32 | DType::UInt32 => Ok("uint32_t"),
        DType::UInt16 => Ok("uint16_t"),
        DType::UInt8 => Ok("uint8_t"),
        _ => Err(invalid_input(format!(
            "bitwise reduce does not support dtype {dtype:?}"
        ))),
    }
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
        let plan = ReducePlan::new(
            DType::Float32,
            &[2, 30],
            &[2],
            &[1],
            ReduceReducer::Max,
            None,
        )
        .unwrap();
        assert_eq!(plan.shape.reduce_count, 30);
        assert_eq!(plan.shape.valid_last_width, 30);
    }

    #[test]
    fn reduce_plan_keeps_aligned_last_width_tile_unmasked() {
        let plan = ReducePlan::new(
            DType::Float32,
            &[2, 64],
            &[2],
            &[1],
            ReduceReducer::Add,
            None,
        )
        .unwrap();
        assert_eq!(plan.shape.reduce_count, 64);
        assert_eq!(plan.shape.valid_last_width, TILE_C as u32);
    }
}
