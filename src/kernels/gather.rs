use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use crate::kernels::reshape_view::ReshapeSourceView;
use std::io;

const GATHER_READER: &str = include_str!("../../kernels/gather_reader.cc");
const WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_OPERAND_ADDR_INDEX: usize = 0;
const READER_START_INDICES_ADDR_INDEX: usize = 1;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum GatherMode {
    Bf16Rows,
    Axis,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct GatherShape {
    mode: GatherMode,
    axis: u32,
    operand_shape: Vec<u32>,
    output_shape: Vec<u32>,
    operand_source_view: Option<ReshapeSourceView>,
    operand_tile_rows: u32,
    operand_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    output_tiles: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct GatherProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: GatherShape,
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

pub(crate) fn gather(
    device: &mut Device,
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    operand_shape: &[usize],
    operand_source_view: Option<ReshapeSourceView>,
    start_indices_shape: &[usize],
    output_shape: &[usize],
    axis: usize,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_gather_buffers(
        operand,
        start_indices,
        operand_shape,
        operand_source_view.as_ref(),
        start_indices_shape,
        axis,
        dtype,
    )?;

    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let shape = gather_shape(
        operand_shape,
        operand_source_view,
        output_shape,
        axis,
        dtype,
    )?;
    let output_tiles = usize::try_from(shape.output_tiles).map_err(|_| {
        invalid_input(format!(
            "gather output tile count does not fit in usize: {}",
            shape.output_tiles
        ))
    })?;
    let work_units = match shape.mode {
        GatherMode::Bf16Rows => usize::try_from(shape.output_tile_rows).map_err(|_| {
            invalid_input(format!(
                "gather output row tile count does not fit in usize: {}",
                shape.output_tile_rows
            ))
        })?,
        GatherMode::Axis => output_tiles,
    };
    let cores = select_worker_cores(device.cores_ref(), work_units)?;
    let output = device.alloc(output_tiles, dtype, &output_allocation_shape, name)?;
    let kernel = GatherKernel {
        operand_addr: u32_addr(operand.addr, "gather operand address")?,
        start_indices_addr: u32_addr(start_indices.addr, "gather start_indices address")?,
        output_addr: u32_addr(output.addr, "gather output address")?,
        key: GatherProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_gather_buffers(
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    operand_shape: &[usize],
    operand_source_view: Option<&ReshapeSourceView>,
    start_indices_shape: &[usize],
    axis: usize,
    dtype: DType,
) -> io::Result<()> {
    if operand_shape.is_empty() {
        return Err(invalid_input("gather requires rank >= 1"));
    }
    if axis >= operand_shape.len() {
        return Err(invalid_input(format!(
            "gather axis {axis} is out of bounds for operand shape {operand_shape:?}"
        )));
    }
    if operand.dtype != dtype {
        return Err(invalid_input(format!(
            "gather operand requires {:?}, got {:?}",
            dtype, operand.dtype
        )));
    }
    if start_indices.dtype != DType::Int32 {
        return Err(invalid_input(format!(
            "gather start_indices requires Int32, got {:?}",
            start_indices.dtype
        )));
    }
    if start_indices_shape.len() != 2 || start_indices_shape[1] != 1 {
        return Err(invalid_input(format!(
            "gather requires start_indices shaped [N, 1], got {start_indices_shape:?}"
        )));
    }

    validate_allocation(
        operand,
        operand_source_view.map_or(operand_shape, ReshapeSourceView::source_shape),
        "gather operand",
    )?;
    validate_allocation(start_indices, start_indices_shape, "gather start_indices")?;
    Ok(())
}

fn gather_shape(
    operand_shape: &[usize],
    operand_source_view: Option<ReshapeSourceView>,
    output_shape: &[usize],
    axis: usize,
    dtype: DType,
) -> io::Result<GatherShape> {
    let operand_physical_shape = operand_source_view
        .as_ref()
        .map_or(operand_shape, ReshapeSourceView::source_shape);
    let operand_allocation_shape = tiled_allocation_shape(operand_physical_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let operand_rank = operand_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    let mode = if operand_source_view.is_none()
        && axis == 0
        && operand_shape.len() == 2
        && dtype == DType::Float16B
    {
        GatherMode::Bf16Rows
    } else {
        GatherMode::Axis
    };
    let (kernel_operand_shape, kernel_output_shape, kernel_axis) = match mode {
        GatherMode::Bf16Rows => (operand_shape.to_vec(), output_shape.to_vec(), axis),
        GatherMode::Axis => {
            let padded_rank = operand_shape.len().max(3);
            (
                pad_rank(operand_shape, padded_rank),
                pad_rank(output_shape, padded_rank),
                axis + padded_rank - operand_shape.len(),
            )
        }
    };

    Ok(GatherShape {
        mode,
        axis: u32_arg(kernel_axis, "gather axis")?,
        operand_shape: u32_shape(&kernel_operand_shape, "gather operand shape")?,
        output_shape: u32_shape(&kernel_output_shape, "gather output shape")?,
        operand_source_view,
        operand_tile_rows: u32_arg(
            operand_allocation_shape[operand_rank - 2] / TILE_R,
            "gather operand tile rows",
        )?,
        operand_tiles_per_row: u32_arg(
            operand_allocation_shape[operand_rank - 1] / TILE_C,
            "gather operand tiles per row",
        )?,
        output_tile_rows: u32_arg(
            output_allocation_shape[output_rank - 2] / TILE_R,
            "gather output tile rows",
        )?,
        output_tiles_per_row: u32_arg(
            output_allocation_shape[output_rank - 1] / TILE_C,
            "gather output tiles per row",
        )?,
        output_tiles: u32_arg(
            tiled_shape_tile_count(output_shape)?,
            "gather output tile count",
        )?,
    })
}

fn pad_rank(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut padded = vec![1; rank - shape.len()];
    padded.extend_from_slice(shape);
    padded
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
    match key.shape.mode {
        GatherMode::Bf16Rows => {
            for (core_index, &core) in key.cores.iter().enumerate() {
                let (offset, row_tiles) =
                    split_tile_range(key.shape.output_tile_rows, core_index, key.cores.len())?;
                runtime_args.add_core(
                    core,
                    vec![
                        0,
                        offset * key.shape.output_tiles_per_row,
                        row_tiles * key.shape.output_tiles_per_row,
                    ],
                    vec![
                        0,
                        0,
                        offset,
                        row_tiles,
                        key.shape.output_shape[0],
                        key.shape.operand_tiles_per_row,
                        key.shape.output_tiles_per_row,
                        key.shape.operand_shape[0],
                    ],
                    Vec::new(),
                )?;
            }
        }
        GatherMode::Axis => {
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
        }
    }
    let runtime_args = runtime_args.build()?;
    let name = match key.shape.mode {
        GatherMode::Bf16Rows => "gather_bf16_rows".to_owned(),
        GatherMode::Axis => format!(
            "gather_axis_{:?}_{}_{}",
            key.dtype,
            key.shape.operand_shape.len(),
            key.shape.axis
        ),
    };
    Ok(Program {
        reader_kernel: gather_reader_source(key.dtype, &key.shape)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.dtype),
                CBConfig::new(1, DType::Int32),
                CBConfig::new(16, key.dtype),
            ],
            ..CompileConfig::default()
        },
        name,
        ..Program::new(runtime_args)
    })
}

fn gather_reader_source(dtype: DType, shape: &GatherShape) -> io::Result<String> {
    let source_view = shape.operand_source_view.as_ref();
    Ok(format!(
        "#define GATHER_BF16_ROWS {}\n\
         #define GATHER_RANK {}\n\
         #define GATHER_AXIS {}\n\
         #define GATHER_OPERAND_SHAPE {}\n\
         #define GATHER_OUTPUT_SHAPE {}\n\
         #define GATHER_OPERAND_RESHAPE_VIEW {}\n\
         #define GATHER_SOURCE_ROWS {}\n\
         #define GATHER_SOURCE_COLS {}\n\
         #define GATHER_SOURCE_TILE_ROWS {}\n\
         #define GATHER_SOURCE_TILES_PER_ROW {}\n\
         #define GATHER_OPERAND_TILE_ROWS {}\n\
         #define GATHER_OPERAND_TILES_PER_ROW {}\n\
         #define GATHER_OUTPUT_TILE_ROWS {}\n\
         #define GATHER_OUTPUT_TILES_PER_ROW {}\n\
         #define GATHER_ELEMENT_TYPE {}\n\
         {GATHER_READER}",
        matches!(shape.mode, GatherMode::Bf16Rows) as u32,
        shape.operand_shape.len(),
        shape.axis,
        cpp_u32_array(&shape.operand_shape),
        cpp_u32_array(&shape.output_shape),
        source_view.is_some() as u32,
        source_view.map_or(1, |view| view.rows),
        source_view.map_or(1, |view| view.cols),
        source_view.map_or(1, |view| view.tile_rows),
        source_view.map_or(1, |view| view.tiles_per_row),
        shape.operand_tile_rows,
        shape.operand_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
        element_type(dtype),
    ))
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
    fn gather_program_splits_bf16_row_tiles_across_cores() {
        let shape =
            gather_shape(&[288, 128], None, &[96, 128], 0, DType::Float16B).expect("gather shape");
        let program = gather_program(GatherProgramKey {
            cores: vec![CoreCoord { x: 1, y: 2 }, CoreCoord { x: 1, y: 3 }],
            dtype: DType::Float16B,
            shape,
        })
        .expect("gather program");

        let blobs = program.runtime_args.blobs();
        assert_eq!(program.runtime_args.section_sizes(), (12, 32, 0));
        assert_eq!(blobs.len(), 2);
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 8));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (8, 4));
        assert_eq!((arg_u32(&blobs[0], 5), arg_u32(&blobs[0], 6)), (0, 2));
        assert_eq!((arg_u32(&blobs[1], 5), arg_u32(&blobs[1], 6)), (2, 1));
        assert!(program.reader_kernel.contains("#define GATHER_BF16_ROWS 1"));
    }

    #[test]
    fn gather_program_uses_axis_mode_for_non_bf16_rows() {
        let shape = gather_shape(&[4, 8], None, &[2, 8], 0, DType::Float32).expect("gather shape");
        let program = gather_program(GatherProgramKey {
            cores: vec![CoreCoord { x: 1, y: 2 }],
            dtype: DType::Float32,
            shape,
        })
        .expect("gather program");

        assert_eq!(program.runtime_args.section_sizes(), (12, 16, 0));
        assert!(program.reader_kernel.contains("#define GATHER_BF16_ROWS 0"));
        assert!(program.reader_kernel.contains("#define GATHER_RANK 3"));
        assert!(program.reader_kernel.contains("#define GATHER_AXIS 1"));
    }

    #[test]
    fn gather_program_pads_rank1_axis_shape() {
        let shape = gather_shape(&[8], None, &[2], 0, DType::Float32).expect("gather shape");
        let program = gather_program(GatherProgramKey {
            cores: vec![CoreCoord { x: 1, y: 2 }],
            dtype: DType::Float32,
            shape,
        })
        .expect("gather program");

        assert!(program.reader_kernel.contains("#define GATHER_RANK 3"));
        assert!(program.reader_kernel.contains("#define GATHER_AXIS 2"));
        assert!(program
            .reader_kernel
            .contains("#define GATHER_OPERAND_SHAPE {1u, 1u, 8u}"));
    }
}
