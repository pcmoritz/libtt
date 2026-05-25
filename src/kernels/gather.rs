use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const GATHER_READER: &str = include_str!("../../kernels/gather_reader.cc");
const GATHER_DIM0_READER: &str = include_str!("../../kernels/gather_dim0_reader.cc");
const GATHER_RANK1_READER: &str = include_str!("../../kernels/gather_rank1_reader.cc");
const GATHER_DIM0_WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const GATHER_WRITER: &str = include_str!("../../kernels/gather_writer.cc");
const READER_OPERAND_ADDR_INDEX: usize = 0;
const READER_START_INDICES_ADDR_INDEX: usize = 1;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GatherKernelShape {
    logical_output_rows: u32,
    operand_tiles_per_row: u32,
    output_tiles_per_row: u32,
    output_row_tile_count: u32,
    logical_operand_rows: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct GatherProgramKey {
    cores: Vec<CoreCoord>,
    shape: GatherKernelShape,
}

struct GatherKernel {
    operand_addr: u32,
    start_indices_addr: u32,
    output_addr: u32,
    key: GatherProgramKey,
}

impl Kernel<GatherProgramKey> for GatherKernel {
    fn program_key(&self) -> GatherProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        gather_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_OPERAND_ADDR_INDEX => Some(self.operand_addr),
            READER_START_INDICES_ADDR_INDEX => Some(self.start_indices_addr),
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

pub(crate) fn gather_bf16_rows(
    device: &mut Device,
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    operand_shape: &[usize],
    start_indices_shape: &[usize],
    output_shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_shapes(operand_shape, start_indices_shape, output_shape)?;
    validate_buffer(operand, DType::Float16B, operand_shape, "gather operand")?;
    validate_buffer(
        start_indices,
        DType::Int32,
        start_indices_shape,
        "gather start_indices",
    )?;

    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles = tiled_tile_count(&output_allocation_shape)?;
    let output = device.alloc(
        output_tiles,
        DType::Float16B,
        &output_allocation_shape,
        name,
    )?;

    let output_row_tile_count = output_allocation_shape[0] / TILE_R;
    let cores = select_worker_cores(device.cores_ref(), output_row_tile_count)?;
    let shape = GatherKernelShape {
        logical_output_rows: u32_arg(output_shape[0], "logical output rows")?,
        operand_tiles_per_row: u32_arg(operand.shape[1] / TILE_C, "operand tiles per row")?,
        output_tiles_per_row: u32_arg(output_allocation_shape[1] / TILE_C, "output tiles per row")?,
        output_row_tile_count: u32_arg(output_row_tile_count, "output row tile count")?,
        logical_operand_rows: u32_arg(operand_shape[0], "logical operand rows")?,
    };
    let kernel = GatherKernel {
        operand_addr: u32_addr(operand.addr, "operand address")?,
        start_indices_addr: u32_addr(start_indices.addr, "start_indices address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: GatherProgramKey { cores, shape },
    };
    kernel.run(device)?;
    Ok(output)
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct GatherDim0Shape {
    axis: u32,
    operand_shape: Vec<u32>,
    output_shape: Vec<u32>,
    operand_tile_rows: u32,
    operand_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    output_tiles: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct GatherDim0ProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: GatherDim0Shape,
}

struct GatherDim0Kernel {
    operand_addr: u32,
    start_indices_addr: u32,
    output_addr: u32,
    key: GatherDim0ProgramKey,
}

impl Kernel<GatherDim0ProgramKey> for GatherDim0Kernel {
    fn program_key(&self) -> GatherDim0ProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        gather_dim0_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_OPERAND_ADDR_INDEX => Some(self.operand_addr),
            READER_START_INDICES_ADDR_INDEX => Some(self.start_indices_addr),
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

pub(crate) fn gather_dim0_slices(
    device: &mut Device,
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    operand_shape: &[usize],
    start_indices_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_dim0_buffers(
        operand,
        start_indices,
        operand_shape,
        start_indices_shape,
        output_shape,
        axis,
        dtype,
    )?;

    let shape = gather_dim0_shape(operand_shape, output_shape, axis)?;
    let output_tiles = usize::try_from(shape.output_tiles).map_err(|_| {
        invalid_input(format!(
            "gather_dim0 output tile count does not fit in usize: {}",
            shape.output_tiles
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(
        output_tiles,
        dtype,
        &tiled_allocation_shape(output_shape)?,
        name,
    )?;
    let kernel = GatherDim0Kernel {
        operand_addr: u32_addr(operand.addr, "gather_dim0 operand address")?,
        start_indices_addr: u32_addr(start_indices.addr, "gather_dim0 start_indices address")?,
        output_addr: u32_addr(output.addr, "gather_dim0 output address")?,
        key: GatherDim0ProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GatherRank1Shape {
    output_tiles: u32,
    logical_output_elements: u32,
    logical_operand_elements: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct GatherRank1ProgramKey {
    cores: Vec<CoreCoord>,
    shape: GatherRank1Shape,
}

struct GatherRank1Kernel {
    operand_addr: u32,
    start_indices_addr: u32,
    output_addr: u32,
    key: GatherRank1ProgramKey,
}

impl Kernel<GatherRank1ProgramKey> for GatherRank1Kernel {
    fn program_key(&self) -> GatherRank1ProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        gather_rank1_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_OPERAND_ADDR_INDEX => Some(self.operand_addr),
            READER_START_INDICES_ADDR_INDEX => Some(self.start_indices_addr),
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

pub(crate) fn gather_s32_rank1(
    device: &mut Device,
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    operand_shape: &[usize],
    start_indices_shape: &[usize],
    output_shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_rank1_shapes(operand_shape, start_indices_shape, output_shape)?;
    validate_buffer(operand, DType::Int32, operand_shape, "rank1 gather operand")?;
    validate_buffer(
        start_indices,
        DType::Int32,
        start_indices_shape,
        "rank1 gather start_indices",
    )?;

    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles = tiled_tile_count(&output_allocation_shape)?;
    let output = device.alloc(output_tiles, DType::Int32, &output_allocation_shape, name)?;

    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let shape = GatherRank1Shape {
        output_tiles: u32_arg(output_tiles, "rank1 gather output tile count")?,
        logical_output_elements: u32_arg(output_shape[0], "rank1 gather output elements")?,
        logical_operand_elements: u32_arg(operand_shape[0], "rank1 gather operand elements")?,
    };
    let kernel = GatherRank1Kernel {
        operand_addr: u32_addr(operand.addr, "rank1 gather operand address")?,
        start_indices_addr: u32_addr(start_indices.addr, "rank1 gather start_indices address")?,
        output_addr: u32_addr(output.addr, "rank1 gather output address")?,
        key: GatherRank1ProgramKey { cores, shape },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_rank1_shapes(
    operand_shape: &[usize],
    start_indices_shape: &[usize],
    output_shape: &[usize],
) -> io::Result<()> {
    if operand_shape.len() != 1 {
        return Err(invalid_input(format!(
            "gather_s32_rank1 requires a rank-1 operand, got {operand_shape:?}"
        )));
    }
    if start_indices_shape.len() != 2 || start_indices_shape[1] != 1 {
        return Err(invalid_input(format!(
            "gather_s32_rank1 requires start_indices shaped [N, 1], got {start_indices_shape:?}"
        )));
    }
    let expected_output_shape = [start_indices_shape[0]];
    if output_shape != expected_output_shape {
        return Err(invalid_input(format!(
            "gather_s32_rank1 output shape mismatch: expected {:?}, got {:?}",
            expected_output_shape, output_shape
        )));
    }
    Ok(())
}

fn validate_dim0_buffers(
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    operand_shape: &[usize],
    start_indices_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
    dtype: DType,
) -> io::Result<()> {
    if operand_shape.is_empty() {
        return Err(invalid_input("gather_axis requires rank >= 1"));
    }
    if axis >= operand_shape.len() {
        return Err(invalid_input(format!(
            "gather_axis axis {axis} is out of bounds for operand shape {operand_shape:?}"
        )));
    }
    if operand.dtype != dtype {
        return Err(invalid_input(format!(
            "gather_axis operand requires {:?}, got {:?}",
            dtype, operand.dtype
        )));
    }
    if start_indices.dtype != DType::Int32 {
        return Err(invalid_input(format!(
            "gather_axis start_indices requires Int32, got {:?}",
            start_indices.dtype
        )));
    }
    if start_indices_shape.len() != 2 || start_indices_shape[1] != 1 {
        return Err(invalid_input(format!(
            "gather_axis requires start_indices shaped [N, 1], got {start_indices_shape:?}"
        )));
    }
    let mut expected_output_shape = operand_shape.to_vec();
    expected_output_shape[axis] = start_indices_shape[0];
    if output_shape != expected_output_shape {
        return Err(invalid_input(format!(
            "gather_axis output shape mismatch: expected {:?}, got {:?}",
            expected_output_shape, output_shape
        )));
    }

    validate_allocation(operand, operand_shape, "gather_axis operand")?;
    validate_allocation(
        start_indices,
        start_indices_shape,
        "gather_axis start_indices",
    )?;
    Ok(())
}

fn gather_dim0_shape(
    operand_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
) -> io::Result<GatherDim0Shape> {
    let operand_allocation_shape = tiled_allocation_shape(operand_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let operand_rank = operand_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    Ok(GatherDim0Shape {
        axis: u32_arg(axis, "gather_axis axis")?,
        operand_shape: u32_shape(operand_shape, "gather_axis operand shape")?,
        output_shape: u32_shape(output_shape, "gather_axis output shape")?,
        operand_tile_rows: u32_arg(
            operand_allocation_shape[operand_rank - 2] / TILE_R,
            "gather_axis operand tile rows",
        )?,
        operand_tiles_per_row: u32_arg(
            operand_allocation_shape[operand_rank - 1] / TILE_C,
            "gather_axis operand tiles per row",
        )?,
        output_tile_rows: u32_arg(
            output_allocation_shape[output_rank - 2] / TILE_R,
            "gather_axis output tile rows",
        )?,
        output_tiles_per_row: u32_arg(
            output_allocation_shape[output_rank - 1] / TILE_C,
            "gather_axis output tiles per row",
        )?,
        output_tiles: u32_arg(
            tiled_shape_tile_count(output_shape)?,
            "gather_axis output tile count",
        )?,
    })
}

fn validate_shapes(
    operand_shape: &[usize],
    start_indices_shape: &[usize],
    output_shape: &[usize],
) -> io::Result<()> {
    if operand_shape.len() != 2 {
        return Err(invalid_input(format!(
            "gather_bf16_rows requires a rank-2 operand, got {operand_shape:?}"
        )));
    }
    if start_indices_shape.len() != 2 || start_indices_shape[1] != 1 {
        return Err(invalid_input(format!(
            "gather_bf16_rows requires start_indices shaped [N, 1], got {start_indices_shape:?}"
        )));
    }
    let expected_output_shape = [start_indices_shape[0], operand_shape[1]];
    if output_shape != expected_output_shape {
        return Err(invalid_input(format!(
            "gather_bf16_rows output shape mismatch: expected {:?}, got {:?}",
            expected_output_shape, output_shape
        )));
    }
    Ok(())
}

fn validate_buffer(
    buffer: &DramBuffer,
    dtype: DType,
    logical_shape: &[usize],
    name: &str,
) -> io::Result<()> {
    if buffer.dtype != dtype {
        return Err(invalid_input(format!(
            "{name} requires {dtype:?}, got {:?}",
            buffer.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    if buffer.shape != expected_shape {
        return Err(invalid_input(format!(
            "{name} allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            buffer.shape, expected_shape, logical_shape
        )));
    }
    let expected_tiles = tiled_tile_count(&expected_shape)?;
    if buffer.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "{name} tile count mismatch: got {}, expected {expected_tiles}",
            buffer.num_tiles
        )));
    }
    Ok(())
}

fn validate_allocation(buffer: &DramBuffer, logical_shape: &[usize], name: &str) -> io::Result<()> {
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    if buffer.shape != expected_shape {
        return Err(invalid_input(format!(
            "{name} allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            buffer.shape, expected_shape, logical_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(logical_shape)?;
    if buffer.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "{name} tile count mismatch: got {}, expected {expected_tiles}",
            buffer.num_tiles
        )));
    }
    Ok(())
}

fn gather_program(key: GatherProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_OPERAND_ADDR_INDEX, READER_START_INDICES_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, row_tiles) =
            split_tile_range(key.shape.output_row_tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, row_tiles, key.shape.output_tiles_per_row],
            vec![
                0,
                0,
                offset,
                row_tiles,
                key.shape.logical_output_rows,
                key.shape.operand_tiles_per_row,
                key.shape.output_tiles_per_row,
                key.shape.logical_operand_rows,
            ],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: GATHER_READER.to_owned(),
        writer_kernel: GATHER_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Int32),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(16, DType::Float16B),
            ],
            ..CompileConfig::default()
        },
        name: "gather_bf16_rows".to_owned(),
        ..Program::new(runtime_args)
    })
}

fn gather_rank1_program(key: GatherRank1ProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_OPERAND_ADDR_INDEX, READER_START_INDICES_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, tiles) =
            split_tile_range(key.shape.output_tiles, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, tiles, 1],
            vec![
                0,
                0,
                offset,
                tiles,
                key.shape.logical_output_elements,
                key.shape.logical_operand_elements,
            ],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: GATHER_RANK1_READER.to_owned(),
        writer_kernel: GATHER_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Int32),
                CBConfig::new(1, DType::Int32),
                CBConfig::new(16, DType::Int32),
            ],
            ..CompileConfig::default()
        },
        name: "gather_s32_rank1".to_owned(),
        ..Program::new(runtime_args)
    })
}

fn gather_dim0_program(key: GatherDim0ProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_OPERAND_ADDR_INDEX, READER_START_INDICES_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.output_tiles, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, 0, offset, n_tiles],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: gather_dim0_reader_source(key.dtype, &key.shape)?,
        writer_kernel: GATHER_DIM0_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.dtype),
                CBConfig::new(1, DType::Int32),
                CBConfig::new(16, key.dtype),
            ],
            ..CompileConfig::default()
        },
        name: format!(
            "gather_axis_{:?}_{}_{}",
            key.dtype,
            key.shape.operand_shape.len(),
            key.shape.axis
        ),
        ..Program::new(runtime_args)
    })
}

fn gather_dim0_reader_source(dtype: DType, shape: &GatherDim0Shape) -> io::Result<String> {
    Ok(format!(
        "#define GATHER_DIM0_RANK {}\n\
         #define GATHER_DIM0_AXIS {}\n\
         #define GATHER_DIM0_OPERAND_SHAPE {}\n\
         #define GATHER_DIM0_OUTPUT_SHAPE {}\n\
         #define GATHER_DIM0_OPERAND_TILE_ROWS {}\n\
         #define GATHER_DIM0_OPERAND_TILES_PER_ROW {}\n\
         #define GATHER_DIM0_OUTPUT_TILE_ROWS {}\n\
         #define GATHER_DIM0_OUTPUT_TILES_PER_ROW {}\n\
         #define GATHER_DIM0_ELEMENT_TYPE {}\n\
         {GATHER_DIM0_READER}",
        shape.operand_shape.len(),
        shape.axis,
        cpp_u32_array(&shape.operand_shape),
        cpp_u32_array(&shape.output_shape),
        shape.operand_tile_rows,
        shape.operand_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
        element_type(dtype),
    ))
}

fn tiled_tile_count(allocation_shape: &[usize]) -> io::Result<usize> {
    if allocation_shape.len() != 2 {
        return Err(invalid_input(format!(
            "gather_bf16_rows requires rank-2 tiled allocation shapes, got {allocation_shape:?}"
        )));
    }
    if allocation_shape[0] % TILE_R != 0 || allocation_shape[1] % TILE_C != 0 {
        return Err(invalid_input(format!(
            "gather_bf16_rows allocation shape must be tile-aligned, got {allocation_shape:?}"
        )));
    }
    let rows = allocation_shape[0] / TILE_R;
    let cols = allocation_shape[1] / TILE_C;
    rows.checked_mul(cols)
        .ok_or_else(|| invalid_input("gather_bf16_rows tile count overflow"))
}

fn u32_shape(shape: &[usize], name: &str) -> io::Result<Vec<u32>> {
    shape
        .iter()
        .enumerate()
        .map(|(index, &dim)| u32_arg(dim, &format!("{name} dimension {index}")))
        .collect()
}

fn cpp_u32_array(values: &[u32]) -> String {
    let values = values
        .iter()
        .map(|value| format!("{value}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{values}}}")
}

fn element_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}

fn u32_addr(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arg_u32(blob: &[u8], index: usize) -> u32 {
        let start = index * std::mem::size_of::<u32>();
        u32::from_le_bytes(
            blob[start..start + std::mem::size_of::<u32>()]
                .try_into()
                .unwrap(),
        )
    }

    #[test]
    fn gather_program_splits_output_row_tiles_across_cores() {
        let program = gather_program(GatherProgramKey {
            cores: vec![CoreCoord { x: 1, y: 2 }, CoreCoord { x: 1, y: 3 }],
            shape: GatherKernelShape {
                logical_output_rows: 96,
                operand_tiles_per_row: 4,
                output_tiles_per_row: 4,
                output_row_tile_count: 3,
                logical_operand_rows: 288,
            },
        })
        .expect("gather program");

        let blobs = program.runtime_args.blobs();
        assert_eq!(blobs.len(), 2);
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 2));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (2, 1));
        assert_eq!(arg_u32(&blobs[0], 3), 4);
        assert_eq!((arg_u32(&blobs[0], 6), arg_u32(&blobs[0], 7)), (0, 2));
        assert_eq!((arg_u32(&blobs[1], 6), arg_u32(&blobs[1], 7)), (2, 1));
        assert_eq!(arg_u32(&blobs[0], 8), 96);
        assert_eq!(arg_u32(&blobs[0], 9), 4);
        assert_eq!(arg_u32(&blobs[0], 10), 4);
        assert_eq!(arg_u32(&blobs[0], 11), 288);
    }
}
