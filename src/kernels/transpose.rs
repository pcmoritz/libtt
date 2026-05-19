use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{
    select_worker_cores, split_tile_range, DramKernel, Kernel, RuntimeArgsBuilder,
};
use std::io;

const GENERAL_READER: &str = include_str!("../../kernels/transpose_general_reader.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;
const MAX_RANK: usize = 8;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GeneralTransposeKernelShape {
    rank: u32,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_rows: u32,
    output_cols: u32,
    output_tiles_per_row: u32,
    output_matrix_tiles: u32,
    output_tile_count: u32,
    input_shape: [u32; MAX_RANK],
    output_shape: [u32; MAX_RANK],
    permutation: [u32; MAX_RANK],
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct GeneralTransposeProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: GeneralTransposeKernelShape,
}

pub(crate) fn transpose_general(
    device: &mut Device,
    input: &DramBuffer,
    input_shape: &[usize],
    output_shape: &[usize],
    permutation: &[i64],
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_input(input, dtype, input_shape)?;
    let shape = general_transpose_shape(input_shape, output_shape, permutation)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles = tiled_shape_tile_count(output_shape)?;
    let output = device.alloc(output_tiles, dtype, &output_allocation_shape, name)?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let kernel = DramKernel {
        reader_addrs: [u32_addr(input.addr, "input address")?],
        output_addr: u32_addr(output.addr, "output address")?,
        key: GeneralTransposeProgramKey {
            cores,
            dtype,
            shape,
        },
        build: general_transpose_program,
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

fn general_transpose_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    permutation: &[i64],
) -> io::Result<GeneralTransposeKernelShape> {
    let rank = input_shape.len();
    if !(2..=MAX_RANK).contains(&rank) || output_shape.len() != rank || permutation.len() != rank {
        return Err(invalid_input(format!(
            "general transpose requires matching ranks in 2..={MAX_RANK}, got input={input_shape:?} output={output_shape:?} permutation={permutation:?}"
        )));
    }
    let mut seen = vec![false; rank];
    let mut permutation_usize = Vec::with_capacity(rank);
    for &dim in permutation {
        let dim = usize::try_from(dim)
            .map_err(|_| invalid_input("general transpose permutation dims must be >= 0"))?;
        if dim >= rank {
            return Err(invalid_input(format!(
                "general transpose permutation dim {dim} is out of bounds for rank {rank}"
            )));
        }
        if std::mem::replace(&mut seen[dim], true) {
            return Err(invalid_input(format!(
                "general transpose permutation repeats dim {dim}"
            )));
        }
        permutation_usize.push(dim);
    }
    if rank != 2 && (input_shape.contains(&0) || output_shape.contains(&0)) {
        return Err(invalid_input(
            "general transpose zero-sized dimensions are not currently supported",
        ));
    }
    let expected_output = permutation_usize
        .iter()
        .map(|&dim| input_shape[dim])
        .collect::<Vec<_>>();
    if output_shape != expected_output {
        return Err(invalid_input(format!(
            "general transpose output shape mismatch: expected {:?}, got {output_shape:?}",
            expected_output
        )));
    }

    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tile_rows = output_allocation_shape[rank - 2] / TILE_R;
    let output_tiles_per_row = output_allocation_shape[rank - 1] / TILE_C;
    let output_matrix_tiles = output_tile_rows
        .checked_mul(output_tiles_per_row)
        .ok_or_else(|| invalid_input("general transpose output matrix tile count overflow"))?;
    Ok(GeneralTransposeKernelShape {
        rank: u32_arg(rank, "rank")?,
        input_tile_rows: u32_arg(input_allocation_shape[rank - 2] / TILE_R, "input tile rows")?,
        input_tiles_per_row: u32_arg(
            input_allocation_shape[rank - 1] / TILE_C,
            "input tiles per row",
        )?,
        output_rows: u32_arg(output_shape[rank - 2], "output rows")?,
        output_cols: u32_arg(output_shape[rank - 1], "output cols")?,
        output_tiles_per_row: u32_arg(output_tiles_per_row, "output tiles per row")?,
        output_matrix_tiles: u32_arg(output_matrix_tiles, "output matrix tiles")?,
        output_tile_count: u32_arg(tiled_shape_tile_count(output_shape)?, "output tile count")?,
        input_shape: padded_array(input_shape)?,
        output_shape: padded_array(output_shape)?,
        permutation: padded_array(&permutation_usize)?,
    })
}

fn general_transpose_program(key: GeneralTransposeProgramKey) -> io::Result<Program> {
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
            vec![0, offset, n_tiles],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: general_transpose_reader_source(key.dtype, &key.shape)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("transpose_general_{:?}_rank{}", key.dtype, key.shape.rank),
        ..Program::new(runtime_args)
    })
}

fn general_transpose_reader_source(
    dtype: DType,
    shape: &GeneralTransposeKernelShape,
) -> io::Result<String> {
    let element_type = match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    };
    Ok(format!(
        "#define TRANSPOSE_GENERAL_MAX_RANK {MAX_RANK}\n\
         #define TRANSPOSE_GENERAL_RANK {}\n\
         #define TRANSPOSE_GENERAL_INPUT_TILE_ROWS {}\n\
         #define TRANSPOSE_GENERAL_INPUT_TILES_PER_ROW {}\n\
         #define TRANSPOSE_GENERAL_OUTPUT_ROWS {}\n\
         #define TRANSPOSE_GENERAL_OUTPUT_COLS {}\n\
         #define TRANSPOSE_GENERAL_OUTPUT_TILES_PER_ROW {}\n\
         #define TRANSPOSE_GENERAL_OUTPUT_MATRIX_TILES {}\n\
         #define TRANSPOSE_GENERAL_OUTPUT_SHAPE {}\n\
         #define TRANSPOSE_GENERAL_INPUT_SHAPE {}\n\
         #define TRANSPOSE_GENERAL_PERMUTATION {}\n\
         #define TRANSPOSE_GENERAL_ELEMENT_TYPE {element_type}\n\
         {GENERAL_READER}",
        shape.rank,
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.output_rows,
        shape.output_cols,
        shape.output_tiles_per_row,
        shape.output_matrix_tiles,
        format_u32_array(&shape.output_shape),
        format_u32_array(&shape.input_shape),
        format_u32_array(&shape.permutation),
    ))
}

fn format_u32_array(values: &[u32; MAX_RANK]) -> String {
    format!(
        "{{{}}}",
        values
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn padded_array(values: &[usize]) -> io::Result<[u32; MAX_RANK]> {
    let mut out = [0u32; MAX_RANK];
    for (index, &value) in values.iter().enumerate() {
        out[index] = u32_arg(value, "ranked value")?;
    }
    Ok(out)
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
    fn general_transpose_shape_describes_rank2_transpose() {
        let shape =
            general_transpose_shape(&[64, 96], &[96, 64], &[1, 0]).expect("transpose shape");

        assert_eq!(shape.rank, 2);
        assert_eq!(shape.input_tile_rows, 2);
        assert_eq!(shape.input_tiles_per_row, 3);
        assert_eq!(shape.output_rows, 96);
        assert_eq!(shape.output_cols, 64);
        assert_eq!(shape.output_tiles_per_row, 2);
        assert_eq!(shape.output_matrix_tiles, 6);
        assert_eq!(shape.output_tile_count, 6);
    }

    #[test]
    fn general_transpose_program_splits_rank2_output_tiles_across_cores() {
        let shape =
            general_transpose_shape(&[64, 96], &[96, 64], &[1, 0]).expect("transpose shape");
        let program = general_transpose_program(GeneralTransposeProgramKey {
            cores: vec![CoreCoord { x: 1, y: 2 }, CoreCoord { x: 1, y: 3 }],
            dtype: DType::Float16B,
            shape,
        })
        .expect("transpose program");

        assert_eq!(program.runtime_args.section_sizes(), (12, 12, 0));
        assert!(program
            .reader_kernel
            .contains("#define TRANSPOSE_GENERAL_RANK 2"));
        let blobs = program.runtime_args.blobs();
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 3));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (3, 3));
    }

    #[test]
    fn transpose_shape_rejects_non_transpose_output() {
        let err =
            general_transpose_shape(&[2, 3], &[2, 3], &[1, 0]).expect_err("shape should fail");
        assert!(err.to_string().contains("output shape mismatch"));
    }
}
