use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/reshape_reader.cc");
const WRITER: &str = include_str!("../../kernels/tile_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReshapeView {
    rows: u32,
    cols: u32,
    tile_rows: u32,
    tiles_per_row: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReshapeKernelShape {
    logical_volume: u32,
    input: ReshapeView,
    output: ReshapeView,
    output_tile_count: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReshapeProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: ReshapeKernelShape,
}

struct ReshapeKernel {
    input_addr: u32,
    output_addr: u32,
    key: ReshapeProgramKey,
}

impl Kernel<ReshapeProgramKey> for ReshapeKernel {
    fn program_key(&self) -> ReshapeProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        reshape_program(self.key.clone())
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

pub(crate) fn reshape(
    device: &mut Device,
    input: &DramBuffer,
    input_shape: &[usize],
    output_shape: &[usize],
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_input(input, dtype, input_shape)?;
    let shape = reshape_shape(input_shape, output_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles = tiled_shape_tile_count(output_shape)?;
    let output = device.alloc(output_tiles, dtype, &output_allocation_shape, name)?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;

    let kernel = ReshapeKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: ReshapeProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_input(input: &DramBuffer, dtype: DType, logical_shape: &[usize]) -> io::Result<()> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "reshape input requires {dtype:?}, got {:?}",
            input.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "reshape input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, logical_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(logical_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "reshape input tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn reshape_shape(input_shape: &[usize], output_shape: &[usize]) -> io::Result<ReshapeKernelShape> {
    let input_volume = checked_volume(input_shape, "input shape")?;
    let output_volume = checked_volume(output_shape, "output shape")?;
    if input_volume != output_volume {
        return Err(invalid_input(format!(
            "reshape input shape {input_shape:?} has volume {input_volume}, output shape {output_shape:?} has volume {output_volume}"
        )));
    }
    let output_tile_count = tiled_shape_tile_count(output_shape)?;
    Ok(ReshapeKernelShape {
        logical_volume: u32_arg(input_volume, "logical volume")?,
        input: reshape_view(input_shape)?,
        output: reshape_view(output_shape)?,
        output_tile_count: u32_arg(output_tile_count, "output tile count")?,
    })
}

fn reshape_view(shape: &[usize]) -> io::Result<ReshapeView> {
    let allocation_shape = tiled_allocation_shape(shape)?;
    let rank = shape.len();
    let (rows, cols) = if rank >= 2 {
        (shape[rank - 2], shape[rank - 1])
    } else {
        (
            allocation_shape[allocation_shape.len() - 2],
            allocation_shape[allocation_shape.len() - 1],
        )
    };
    Ok(ReshapeView {
        rows: u32_arg(rows, "reshape rows")?,
        cols: u32_arg(cols, "reshape cols")?,
        tile_rows: u32_arg(
            allocation_shape[allocation_shape.len() - 2] / TILE_R,
            "reshape tile rows",
        )?,
        tiles_per_row: u32_arg(
            allocation_shape[allocation_shape.len() - 1] / TILE_C,
            "reshape tiles per row",
        )?,
    })
}

fn reshape_program(key: ReshapeProgramKey) -> io::Result<Program> {
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
            vec![
                0,
                offset,
                n_tiles,
                key.shape.logical_volume,
                key.shape.input.rows,
                key.shape.input.cols,
                key.shape.input.tile_rows,
                key.shape.input.tiles_per_row,
                key.shape.output.rows,
                key.shape.output.cols,
                key.shape.output.tile_rows,
                key.shape.output.tiles_per_row,
            ],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: reshape_reader_source(key.dtype)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("reshape_{:?}", key.dtype),
        ..Program::new(runtime_args)
    })
}

fn reshape_reader_source(dtype: DType) -> io::Result<String> {
    let element_type = match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    };
    Ok(format!(
        "#define RESHAPE_ELEMENT_TYPE {element_type}\n{READER}"
    ))
}

fn checked_volume(shape: &[usize], label: &str) -> io::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!("reshape {label} volume overflow"),
            )
        })
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

    #[test]
    fn reshape_shape_preserves_volume_and_describes_rank3_output() {
        let shape = reshape_shape(&[18, 128], &[18, 4, 32]).expect("reshape shape");

        assert_eq!(shape.logical_volume, 18 * 128);
        assert_eq!(shape.input.rows, 18);
        assert_eq!(shape.input.cols, 128);
        assert_eq!(shape.input.tile_rows, 1);
        assert_eq!(shape.input.tiles_per_row, 4);
        assert_eq!(shape.output.rows, 4);
        assert_eq!(shape.output.cols, 32);
        assert_eq!(shape.output.tile_rows, 1);
        assert_eq!(shape.output.tiles_per_row, 1);
        assert_eq!(shape.output_tile_count, 18);
    }

    #[test]
    fn reshape_shape_rejects_volume_mismatch() {
        let err = reshape_shape(&[2, 3], &[2, 4]).expect_err("volume mismatch");
        assert!(err.to_string().contains("volume"));
    }

    #[test]
    fn reshape_shape_reports_volume_overflow_as_out_of_memory() {
        let err = reshape_shape(&[usize::MAX, 2], &[usize::MAX, 2]).expect_err("volume overflow");

        assert_eq!(err.kind(), io::ErrorKind::OutOfMemory);
        assert!(err.to_string().contains("input shape volume overflow"));
    }
}
