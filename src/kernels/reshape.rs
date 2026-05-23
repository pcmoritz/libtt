use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/reshape_reader.cc");
const ROW_MAJOR_READER: &str = include_str!("../../kernels/reshape_row_major_reader.cc");
const ROW_PACK_DIRECT_READER: &str =
    include_str!("../../kernels/reshape_row_pack_direct_reader.cc");
const ROW_COPY_DIRECT_READER: &str =
    include_str!("../../kernels/reshape_row_copy_direct_reader.cc");
const ROW_UNPACK_DIRECT_READER: &str =
    include_str!("../../kernels/reshape_row_unpack_direct_reader.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const READER_OUTPUT_ADDR_INDEX: usize = 1;
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
    mode: ReshapeMode,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ReshapeMode {
    Generic,
    RowCopyDirect,
    RowMajorPackDirect,
    RowMajorUnpackDirect,
    RowMajorPack,
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
            READER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
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
    let partition_count = reshape_partition_count(&shape)?;
    let cores = select_worker_cores(device.cores_ref(), partition_count)?;

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
        mode: reshape_mode(input_shape, output_shape)?,
    })
}

fn reshape_mode(input_shape: &[usize], output_shape: &[usize]) -> io::Result<ReshapeMode> {
    let input = reshape_view(input_shape)?;
    let output = reshape_view(output_shape)?;
    if input.cols == output.cols && input.cols % TILE_C as u32 == 0 {
        return Ok(ReshapeMode::RowCopyDirect);
    }
    if input.cols
        == output.rows.checked_mul(output.cols).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "reshape output row-major pack shape overflow",
            )
        })?
        && output.cols % TILE_C as u32 == 0
    {
        // A single row flattened over [rows, cols] can be packed by copying one
        // source tile row into each destination tile row. When the destination
        // fits in one tile row, split that work by row fragments so small Q/K/V
        // decode reshapes can use many cores instead of one core per output tile.
        if output.rows <= TILE_R as u32 {
            return Ok(ReshapeMode::RowMajorPackDirect);
        }
        return Ok(ReshapeMode::RowMajorPack);
    }
    if output.cols
        == input.rows.checked_mul(input.cols).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "reshape output row-major unpack shape overflow",
            )
        })?
        && input.cols % TILE_C as u32 == 0
    {
        return Ok(ReshapeMode::RowMajorUnpackDirect);
    }
    Ok(ReshapeMode::Generic)
}

fn reshape_partition_count(shape: &ReshapeKernelShape) -> io::Result<usize> {
    match shape.mode {
        ReshapeMode::RowCopyDirect | ReshapeMode::RowMajorPackDirect => {
            let fragments = match shape.mode {
                ReshapeMode::RowCopyDirect => row_copy_fragments(shape)?,
                ReshapeMode::RowMajorPackDirect => row_major_pack_fragments(shape)?,
                _ => unreachable!(),
            };
            usize::try_from(fragments).map_err(|_| {
                invalid_input(format!(
                    "reshape fragment count does not fit in usize: {fragments}"
                ))
            })
        }
        ReshapeMode::RowMajorUnpackDirect => {
            let fragments = row_major_unpack_fragments(shape)?;
            usize::try_from(fragments).map_err(|_| {
                invalid_input(format!(
                    "reshape fragment count does not fit in usize: {fragments}"
                ))
            })
        }
        ReshapeMode::Generic | ReshapeMode::RowMajorPack => {
            usize::try_from(shape.output_tile_count).map_err(|_| {
                invalid_input(format!(
                    "reshape output tile count does not fit in usize: {}",
                    shape.output_tile_count
                ))
            })
        }
    }
}

fn row_copy_fragments(shape: &ReshapeKernelShape) -> io::Result<u32> {
    let rows = shape
        .logical_volume
        .checked_div(shape.output.cols)
        .ok_or_else(|| invalid_input("reshape row copy has zero output columns"))?;
    if rows
        .checked_mul(shape.output.cols)
        .ok_or_else(|| invalid_input("reshape row copy volume overflow"))?
        != shape.logical_volume
    {
        return Err(invalid_input(
            "reshape row copy logical volume is not divisible by output columns",
        ));
    }
    rows.checked_mul(shape.output.tiles_per_row)
        .ok_or_else(|| invalid_input("reshape row copy fragment count overflow"))
}

fn row_major_pack_fragments(shape: &ReshapeKernelShape) -> io::Result<u32> {
    let elements_per_batch = shape
        .output
        .rows
        .checked_mul(shape.output.cols)
        .ok_or_else(|| invalid_input("reshape row-major pack elements per batch overflow"))?;
    let batches = shape
        .logical_volume
        .checked_div(elements_per_batch)
        .ok_or_else(|| invalid_input("reshape row-major pack has zero elements per batch"))?;
    if batches
        .checked_mul(elements_per_batch)
        .ok_or_else(|| invalid_input("reshape row-major pack volume overflow"))?
        != shape.logical_volume
    {
        return Err(invalid_input(
            "reshape row-major pack logical volume is not divisible by output matrix size",
        ));
    }
    batches
        .checked_mul(shape.output.rows)
        .and_then(|value| value.checked_mul(shape.output.tiles_per_row))
        .ok_or_else(|| invalid_input("reshape row-major pack fragment count overflow"))
}

fn row_major_unpack_fragments(shape: &ReshapeKernelShape) -> io::Result<u32> {
    let rows = shape
        .logical_volume
        .checked_div(shape.output.cols)
        .ok_or_else(|| invalid_input("reshape row-major unpack has zero output columns"))?;
    if rows
        .checked_mul(shape.output.cols)
        .ok_or_else(|| invalid_input("reshape row-major unpack volume overflow"))?
        != shape.logical_volume
    {
        return Err(invalid_input(
            "reshape row-major unpack logical volume is not divisible by output columns",
        ));
    }
    rows.checked_mul(shape.output.tiles_per_row)
        .ok_or_else(|| invalid_input("reshape row-major unpack fragment count overflow"))
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
    let (writer_dynamic_indices, reader_dynamic_indices) = match key.shape.mode {
        ReshapeMode::RowCopyDirect
        | ReshapeMode::RowMajorPackDirect
        | ReshapeMode::RowMajorUnpackDirect => (
            Vec::new(),
            vec![READER_INPUT_ADDR_INDEX, READER_OUTPUT_ADDR_INDEX],
        ),
        ReshapeMode::Generic | ReshapeMode::RowMajorPack => (
            vec![WRITER_OUTPUT_ADDR_INDEX],
            vec![READER_INPUT_ADDR_INDEX],
        ),
    };
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        writer_dynamic_indices,
        reader_dynamic_indices,
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let partition_count = u32::try_from(reshape_partition_count(&key.shape)?)
            .map_err(|_| invalid_input("reshape partition count does not fit in u32"))?;
        let (offset, n_tiles) = split_tile_range(partition_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            reshape_writer_args(key.shape.mode, offset, n_tiles),
            reshape_reader_args(&key.shape, offset, n_tiles),
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: reshape_reader_source(key.dtype, key.shape.mode)?,
        writer_kernel: match key.shape.mode {
            ReshapeMode::RowCopyDirect
            | ReshapeMode::RowMajorPackDirect
            | ReshapeMode::RowMajorUnpackDirect => String::new(),
            ReshapeMode::Generic | ReshapeMode::RowMajorPack => WRITER.to_owned(),
        },
        compile: CompileConfig {
            cbs: match key.shape.mode {
                ReshapeMode::RowCopyDirect
                | ReshapeMode::RowMajorPackDirect
                | ReshapeMode::RowMajorUnpackDirect => vec![CBConfig::new(0, key.dtype)],
                ReshapeMode::Generic | ReshapeMode::RowMajorPack => {
                    vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)]
                }
            },
            ..CompileConfig::default()
        },
        name: format!("reshape_{:?}_{:?}", key.dtype, key.shape.mode),
        ..Program::new(runtime_args)
    })
}

fn reshape_writer_args(mode: ReshapeMode, offset: u32, n_tiles: u32) -> Vec<u32> {
    match mode {
        ReshapeMode::RowCopyDirect
        | ReshapeMode::RowMajorPackDirect
        | ReshapeMode::RowMajorUnpackDirect => Vec::new(),
        ReshapeMode::Generic | ReshapeMode::RowMajorPack => vec![0, offset, n_tiles],
    }
}

fn reshape_reader_args(shape: &ReshapeKernelShape, offset: u32, n_tiles: u32) -> Vec<u32> {
    match shape.mode {
        ReshapeMode::Generic => vec![
            0,
            offset,
            n_tiles,
            shape.logical_volume,
            shape.input.rows,
            shape.input.cols,
            shape.input.tile_rows,
            shape.input.tiles_per_row,
            shape.output.rows,
            shape.output.cols,
            shape.output.tile_rows,
            shape.output.tiles_per_row,
        ],
        ReshapeMode::RowCopyDirect => vec![
            0,
            0,
            offset,
            n_tiles,
            shape.input.rows,
            shape.input.tile_rows,
            shape.input.tiles_per_row,
            shape.output.rows,
            shape.output.tile_rows,
            shape.output.tiles_per_row,
        ],
        ReshapeMode::RowMajorPackDirect => vec![
            0,
            0,
            offset,
            n_tiles,
            shape.input.tiles_per_row,
            shape.output.tiles_per_row,
            shape.output.rows,
            shape.output.tile_rows,
        ],
        ReshapeMode::RowMajorUnpackDirect => vec![
            0,
            0,
            offset,
            n_tiles,
            shape.input.cols,
            shape.input.tile_rows,
            shape.input.tiles_per_row,
            shape.output.rows,
            shape.output.tile_rows,
            shape.output.tiles_per_row,
        ],
        ReshapeMode::RowMajorPack => vec![
            0,
            offset,
            n_tiles,
            shape.input.rows,
            shape.input.cols,
            shape.input.tile_rows,
            shape.input.tiles_per_row,
            shape.output.rows,
            shape.output.cols,
            shape.output.tile_rows,
            shape.output.tiles_per_row,
        ],
    }
}

fn reshape_reader_source(dtype: DType, mode: ReshapeMode) -> io::Result<String> {
    let element_type = match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    };
    let mode_define = match mode {
        ReshapeMode::Generic => {
            return Ok(format!(
                "#define RESHAPE_ELEMENT_TYPE {element_type}\n{READER}"
            ));
        }
        ReshapeMode::RowCopyDirect => {
            return Ok(format!(
                "#define RESHAPE_ELEMENT_TYPE {element_type}\n{ROW_COPY_DIRECT_READER}"
            ));
        }
        ReshapeMode::RowMajorPackDirect => {
            return Ok(format!(
                "#define RESHAPE_ELEMENT_TYPE {element_type}\n{ROW_PACK_DIRECT_READER}"
            ));
        }
        ReshapeMode::RowMajorUnpackDirect => {
            return Ok(format!(
                "#define RESHAPE_ELEMENT_TYPE {element_type}\n{ROW_UNPACK_DIRECT_READER}"
            ));
        }
        ReshapeMode::RowMajorPack => "#define RESHAPE_ROW_MAJOR_MODE_PACK 1\n",
    };
    Ok(format!(
        "#define RESHAPE_ELEMENT_TYPE {element_type}\n{mode_define}{ROW_MAJOR_READER}"
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
    fn reshape_shape_preserves_volume_and_describes_rank3_row_pack_output() {
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
        assert_eq!(shape.mode, ReshapeMode::RowMajorPackDirect);
        assert_eq!(reshape_partition_count(&shape).expect("partitions"), 18 * 4);
    }

    #[test]
    fn reshape_shape_detects_decode_row_pack_and_unpack_modes() {
        let packed = reshape_shape(&[1, 4096], &[1, 32, 128]).expect("pack shape");
        assert_eq!(packed.mode, ReshapeMode::RowMajorPackDirect);
        assert_eq!(reshape_partition_count(&packed).expect("partitions"), 128);

        let unpacked = reshape_shape(&[1, 32, 128], &[1, 4096]).expect("unpack shape");
        assert_eq!(unpacked.mode, ReshapeMode::RowMajorUnpackDirect);
        assert_eq!(reshape_partition_count(&unpacked).expect("partitions"), 128);
    }

    #[test]
    fn reshape_shape_detects_multi_row_pack_and_unpack_modes() {
        let packed = reshape_shape(&[27, 1024], &[27, 8, 128]).expect("pack shape");
        assert_eq!(packed.mode, ReshapeMode::RowMajorPackDirect);
        assert_eq!(
            reshape_partition_count(&packed).expect("partitions"),
            27 * 8 * 4
        );

        let unpacked = reshape_shape(&[27, 32, 128], &[27, 4096]).expect("unpack shape");
        assert_eq!(unpacked.mode, ReshapeMode::RowMajorUnpackDirect);
        assert_eq!(
            reshape_partition_count(&unpacked).expect("partitions"),
            27 * 128
        );
    }

    #[test]
    fn reshape_shape_detects_row_copy_direct_mode() {
        let split = reshape_shape(&[32, 128], &[8, 4, 128]).expect("split rows shape");
        assert_eq!(split.mode, ReshapeMode::RowCopyDirect);
        assert_eq!(reshape_partition_count(&split).expect("partitions"), 128);

        let merged = reshape_shape(&[8, 4, 128], &[1, 32, 128]).expect("merge rows shape");
        assert_eq!(merged.mode, ReshapeMode::RowCopyDirect);
        assert_eq!(reshape_partition_count(&merged).expect("partitions"), 128);
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
