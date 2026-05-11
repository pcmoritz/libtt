use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, DType, DramBuffer, TILE_C, TILE_R};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const GATHER_READER: &str = include_str!("../../kernels/gather_reader.cc");
const READER_OPERAND_ADDR_INDEX: usize = 0;
const READER_START_INDICES_ADDR_INDEX: usize = 1;
const READER_OUTPUT_ADDR_INDEX: usize = 2;

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
            READER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
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

fn gather_program(key: GatherProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        Vec::new(),
        vec![
            READER_OPERAND_ADDR_INDEX,
            READER_START_INDICES_ADDR_INDEX,
            READER_OUTPUT_ADDR_INDEX,
        ],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, row_tiles) =
            split_tile_range(key.shape.output_row_tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            Vec::new(),
            vec![
                0,
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
        assert_eq!((arg_u32(&blobs[0], 3), arg_u32(&blobs[0], 4)), (0, 2));
        assert_eq!((arg_u32(&blobs[1], 3), arg_u32(&blobs[1], 4)), (2, 1));
        assert_eq!(arg_u32(&blobs[0], 5), 96);
        assert_eq!(arg_u32(&blobs[0], 6), 4);
        assert_eq!(arg_u32(&blobs[0], 7), 4);
        assert_eq!(arg_u32(&blobs[0], 8), 288);
    }
}
