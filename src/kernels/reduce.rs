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
    output_tile_rows_per_prefix: u32,
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
        let (output_rank, output_dim0, output_dim1, output_tile_rows_per_prefix) =
            match output_shape {
                [dim] => (OutputRank::One, 1, *dim, 1),
                [] => {
                    return Err(invalid_input(
                        "reduce kernel currently requires rank >= 1 output",
                    ))
                }
                _ => {
                    let output_rank = output_shape.len();
                    (
                        OutputRank::Two,
                        output_shape[output_rank - 2],
                        output_shape[output_rank - 1],
                        output_allocation_shape[output_rank - 2] / TILE_R,
                    )
                }
            };
        if output_tile_rows_per_prefix == 0 && !matches!(output_rank, OutputRank::One) {
            return Err(invalid_input(format!(
                "reduce output tile rows per prefix must be nonzero for shape {output_shape:?}"
            )));
        };

        let op = ReduceOp::from_reducer(reducer)?;
        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            shape: ReduceKernelShape {
                input_width_tiles: u32_arg(input_width_tiles, "input width tile count")?,
                valid_last_width,
                output_tiles: u32_arg(output_tiles, "output tile count")?,
                inner_output_tiles: u32_arg(inner_output_tiles, "inner output tile count")?,
                output_tile_rows_per_prefix: u32_arg(
                    output_tile_rows_per_prefix,
                    "output tile rows per prefix",
                )?,
                output_rank,
                output_dim0: u32_arg(output_dim0, "output dim0")?,
                output_dim1: u32_arg(output_dim1, "output dim1")?,
            },
            op,
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
    let kernel = DramKernel {
        reader_addrs: [u32_addr(input.addr, "input address")?],
        output_addr: u32_addr(output.addr, "output address")?,
        key: ReduceProgramKey {
            cores,
            dtype: plan.dtype,
            op: plan.op,
            shape: plan.shape,
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
                shape.output_tile_rows_per_prefix,
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
    let tile_rows_per_prefix = shape.output_tile_rows_per_prefix;
    if tile_rows_per_prefix == 0 {
        return Err(invalid_input(
            "matrix reduce requires nonzero output tile rows per prefix",
        ));
    }
    let prefix_offset = tile_row_offset / tile_rows_per_prefix;
    let row_tile_offset = tile_row_offset % tile_rows_per_prefix;
    let row_offset = checked_mul_u32(row_tile_offset, TILE_R as u32, "output row offset")?;
    let prefix_group_offset = checked_mul_u32(
        prefix_offset,
        shape.output_dim0,
        "reduce prefix group offset",
    )?;
    let group_row_offset = prefix_group_offset
        .checked_add(row_offset)
        .ok_or_else(|| invalid_input("reduce group row offset overflow"))?;
    let group_offset = checked_mul_u32(
        group_row_offset,
        shape.inner_output_tiles,
        "reduce group offset",
    )?;
    let end_tile_row = tile_row_offset
        .checked_add(tile_rows)
        .ok_or_else(|| invalid_input("reduce tile row range overflow"))?;
    let mut reduce_rows = 0u32;
    for tile_row in tile_row_offset..end_tile_row {
        let row_tile = tile_row % tile_rows_per_prefix;
        let row_offset = checked_mul_u32(row_tile, TILE_R as u32, "output row offset")?;
        let rows = shape
            .output_dim0
            .saturating_sub(row_offset)
            .min(TILE_R as u32);
        reduce_rows = reduce_rows
            .checked_add(rows)
            .ok_or_else(|| invalid_input("reduce row count overflow"))?;
    }
    let reduce_groups =
        checked_mul_u32(reduce_rows, shape.inner_output_tiles, "reduce group count")?;
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
}
