use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const WRITER: &str = include_str!("../../kernels/iota_writer.cc");
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;
const WRITER_TILE_OFFSET_INDEX: usize = 1;
const WRITER_TILE_COUNT_INDEX: usize = 2;
const WRITER_RANK_INDEX: usize = 3;
const WRITER_DIM0_INDEX: usize = 4;
const WRITER_DIM1_INDEX: usize = 5;
const WRITER_IOTA_DIMENSION_INDEX: usize = 6;
const WRITER_OUTPUT_TILES_PER_ROW_INDEX: usize = 7;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct IotaProgramKey {
    cores: Vec<CoreCoord>,
    tile_count: u32,
    dtype: DType,
}

struct IotaKernel {
    output_addr: u32,
    rank: u32,
    dim0: u32,
    dim1: u32,
    iota_dimension: u32,
    output_tiles_per_row: u32,
    key: IotaProgramKey,
}

impl Kernel<IotaProgramKey> for IotaKernel {
    fn program_key(&self) -> IotaProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        iota_program(self.key.clone())
    }

    #[inline]
    fn writer_runtime_arg(&self, core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            WRITER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            WRITER_TILE_OFFSET_INDEX => tile_range(self.key.tile_count, core, &self.key.cores)
                .ok()
                .map(|(offset, _)| offset),
            WRITER_TILE_COUNT_INDEX => tile_range(self.key.tile_count, core, &self.key.cores)
                .ok()
                .map(|(_, count)| count),
            WRITER_RANK_INDEX => Some(self.rank),
            WRITER_DIM0_INDEX => Some(self.dim0),
            WRITER_DIM1_INDEX => Some(self.dim1),
            WRITER_IOTA_DIMENSION_INDEX => Some(self.iota_dimension),
            WRITER_OUTPUT_TILES_PER_ROW_INDEX => Some(self.output_tiles_per_row),
            _ => None,
        }
    }
}

pub(crate) fn iota(
    device: &mut Device,
    dtype: DType,
    shape: &[usize],
    iota_dimension: usize,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_iota(dtype, shape, iota_dimension)?;

    let allocation_shape = tiled_allocation_shape(shape)?;
    let output_tiles = tiled_shape_tile_count(shape)?;
    let output_tiles_per_row = allocation_shape[allocation_shape.len() - 1] / TILE_C;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(output_tiles, dtype, &allocation_shape, name)?;

    let dim0 = shape[0];
    let dim1 = shape.get(1).copied().unwrap_or(1);
    let tile_count = u32_arg(output_tiles, "tile count")?;
    let kernel = IotaKernel {
        output_addr: u32_addr(output.addr, "output address")?,
        rank: u32_arg(shape.len(), "rank")?,
        dim0: u32_arg(dim0, "dim0")?,
        dim1: u32_arg(dim1, "dim1")?,
        iota_dimension: u32_arg(iota_dimension, "iota dimension")?,
        output_tiles_per_row: u32_arg(output_tiles_per_row, "output tiles per row")?,
        key: IotaProgramKey {
            cores,
            tile_count,
            dtype,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_iota(dtype: DType, shape: &[usize], iota_dimension: usize) -> io::Result<()> {
    validate_dtype(dtype)?;
    if shape.is_empty() || shape.len() > 2 {
        return Err(invalid_input(format!(
            "iota currently supports rank-1 and rank-2 outputs, got {shape:?}"
        )));
    }
    if iota_dimension >= shape.len() {
        return Err(invalid_input(format!(
            "iota dimension {iota_dimension} is out of bounds for shape {shape:?}"
        )));
    }
    Ok(())
}

fn validate_dtype(dtype: DType) -> io::Result<()> {
    if matches!(
        dtype,
        DType::Float16B | DType::Float32 | DType::Int32 | DType::UInt32
    ) {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "iota currently supports Float16B, Float32, Int32, and UInt32 outputs, got {dtype:?}"
        )))
    }
}

fn iota_program(key: IotaProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![
            WRITER_OUTPUT_ADDR_INDEX,
            WRITER_TILE_OFFSET_INDEX,
            WRITER_TILE_COUNT_INDEX,
            WRITER_RANK_INDEX,
            WRITER_DIM0_INDEX,
            WRITER_DIM1_INDEX,
            WRITER_IOTA_DIMENSION_INDEX,
            WRITER_OUTPUT_TILES_PER_ROW_INDEX,
        ],
        Vec::new(),
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles, 0, 0, 0, 0, 0],
            Vec::new(),
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        writer_kernel: iota_writer_source(key.dtype)?,
        compile: CompileConfig {
            cbs: vec![CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("iota_{:?}", key.dtype),
        ..Program::new(runtime_args)
    })
}

fn iota_writer_source(dtype: DType) -> io::Result<String> {
    let defines = match dtype {
        DType::Int32 => "#define IOTA_DTYPE_INT32 1\n#define IOTA_DTYPE_UINT32 0\n#define IOTA_DTYPE_FLOAT32 0\n#define IOTA_DTYPE_BFLOAT16 0\n",
        DType::UInt32 => "#define IOTA_DTYPE_INT32 0\n#define IOTA_DTYPE_UINT32 1\n#define IOTA_DTYPE_FLOAT32 0\n#define IOTA_DTYPE_BFLOAT16 0\n",
        DType::Float32 => "#define IOTA_DTYPE_INT32 0\n#define IOTA_DTYPE_UINT32 0\n#define IOTA_DTYPE_FLOAT32 1\n#define IOTA_DTYPE_BFLOAT16 0\n",
        DType::Float16B => "#define IOTA_DTYPE_INT32 0\n#define IOTA_DTYPE_UINT32 0\n#define IOTA_DTYPE_FLOAT32 0\n#define IOTA_DTYPE_BFLOAT16 1\n",
        other => {
            return Err(invalid_input(format!(
                "iota currently does not support {other:?} outputs"
            )));
        }
    };
    Ok(format!("{defines}{WRITER}"))
}

fn tile_range(tile_count: u32, core: CoreCoord, cores: &[CoreCoord]) -> io::Result<(u32, u32)> {
    let core_index = cores
        .iter()
        .position(|candidate| *candidate == core)
        .ok_or_else(|| invalid_input(format!("core {core} is not part of this iota launch")))?;
    split_tile_range(tile_count, core_index, cores.len())
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
    fn iota_program_splits_tiles_across_cores() {
        let program = iota_program(IotaProgramKey {
            cores: vec![CoreCoord { x: 1, y: 2 }, CoreCoord { x: 1, y: 3 }],
            tile_count: 5,
            dtype: DType::Float32,
        })
        .expect("iota program");

        let blobs = program.runtime_args.blobs();
        assert_eq!(blobs.len(), 2);
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 3));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (3, 2));
    }

    #[test]
    fn validate_iota_rejects_unsupported_rank_and_dtype() {
        assert!(validate_iota(DType::Float32, &[2, 3], 1).is_ok());
        assert!(validate_iota(DType::Float32, &[2, 3, 4], 1).is_err());
        assert!(validate_iota(DType::UInt16, &[2, 3], 1).is_err());
    }
}
