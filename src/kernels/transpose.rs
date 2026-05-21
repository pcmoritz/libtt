use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{
    select_worker_cores, split_tile_range, DramKernel, Kernel, RuntimeArgsBuilder,
};
use std::io;

const READER: &str = include_str!("../../kernels/transpose_reader.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct TransposeKernelShape {
    input_rows: u32,
    input_cols: u32,
    input_tiles_per_row: u32,
    output_tiles_per_row: u32,
    output_tile_count: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TransposeProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: TransposeKernelShape,
}

pub(crate) fn transpose_rank2(
    device: &mut Device,
    input: &DramBuffer,
    input_shape: &[usize],
    output_shape: &[usize],
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_input(input, dtype, input_shape)?;
    let shape = transpose_shape(input_shape, output_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles = tiled_shape_tile_count(output_shape)?;
    let output = device.alloc(output_tiles, dtype, &output_allocation_shape, name)?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let kernel = DramKernel {
        reader_addrs: [u32_addr(input.addr, "input address")?],
        output_addr: u32_addr(output.addr, "output address")?,
        key: TransposeProgramKey {
            cores,
            dtype,
            shape,
        },
        build: transpose_program,
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_input(input: &DramBuffer, dtype: DType, logical_shape: &[usize]) -> io::Result<()> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "transpose input requires {dtype:?}, got {:?}",
            input.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "transpose input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, logical_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(logical_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "transpose input tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn transpose_shape(
    input_shape: &[usize],
    output_shape: &[usize],
) -> io::Result<TransposeKernelShape> {
    if input_shape.len() != 2 || output_shape.len() != 2 {
        return Err(invalid_input(format!(
            "rank-2 transpose requires rank-2 input/output, got {input_shape:?} -> {output_shape:?}"
        )));
    }
    if output_shape != [input_shape[1], input_shape[0]] {
        return Err(invalid_input(format!(
            "rank-2 transpose output shape mismatch: expected [{}, {}], got {output_shape:?}",
            input_shape[1], input_shape[0]
        )));
    }
    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    Ok(TransposeKernelShape {
        input_rows: u32_arg(input_shape[0], "input rows")?,
        input_cols: u32_arg(input_shape[1], "input cols")?,
        input_tiles_per_row: u32_arg(input_allocation_shape[1] / TILE_C, "input tiles per row")?,
        output_tiles_per_row: u32_arg(output_allocation_shape[1] / TILE_C, "output tiles per row")?,
        output_tile_count: u32_arg(tiled_shape_tile_count(output_shape)?, "output tile count")?,
    })
}

fn transpose_program(key: TransposeProgramKey) -> io::Result<Program> {
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
                key.shape.input_rows,
                key.shape.input_cols,
                key.shape.input_tiles_per_row,
                key.shape.output_tiles_per_row,
            ],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: transpose_reader_source(key.dtype)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("transpose_rank2_{:?}", key.dtype),
        ..Program::new(runtime_args)
    })
}

fn transpose_reader_source(dtype: DType) -> io::Result<String> {
    let element_type = match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    };
    Ok(format!(
        "#define TRANSPOSE_ELEMENT_TYPE {element_type}\n{READER}"
    ))
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
    fn transpose_shape_describes_rank2_transpose() {
        let shape = transpose_shape(&[64, 96], &[96, 64]).expect("transpose shape");

        assert_eq!(shape.input_rows, 64);
        assert_eq!(shape.input_cols, 96);
        assert_eq!(shape.input_tiles_per_row, 3);
        assert_eq!(shape.output_tiles_per_row, 2);
        assert_eq!(shape.output_tile_count, 6);
    }

    #[test]
    fn transpose_program_splits_output_tiles_across_cores() {
        let shape = transpose_shape(&[64, 96], &[96, 64]).expect("transpose shape");
        let program = transpose_program(TransposeProgramKey {
            cores: vec![CoreCoord { x: 1, y: 2 }, CoreCoord { x: 1, y: 3 }],
            dtype: DType::Float16B,
            shape,
        })
        .expect("transpose program");

        assert_eq!(program.runtime_args.section_sizes(), (12, 28, 0));
        let blobs = program.runtime_args.blobs();
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 3));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (3, 3));
        assert_eq!((arg_u32(&blobs[0], 4), arg_u32(&blobs[0], 5)), (0, 3));
        assert_eq!((arg_u32(&blobs[1], 4), arg_u32(&blobs[1], 5)), (3, 3));
    }

    #[test]
    fn transpose_shape_rejects_non_transpose_output() {
        let err = transpose_shape(&[2, 3], &[2, 3]).expect_err("shape should fail");
        assert!(err.to_string().contains("output shape mismatch"));
    }
}
