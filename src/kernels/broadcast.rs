use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::io;

const VECTOR_TO_COLUMN: &str = include_str!("../../kernels/broadcast_vector_to_column.cc");
const INPUT_ADDR_INDEX: usize = 0;
const OUTPUT_ADDR_INDEX: usize = 1;
const TILE_R: usize = 32;
const TILE_C: usize = 32;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct VectorToColumnProgramKey {
    core: CoreCoord,
    tile_count: u32,
    dtype: DType,
}

struct VectorToColumnKernel {
    input_addr: u32,
    output_addr: u32,
    key: VectorToColumnProgramKey,
}

impl Kernel<VectorToColumnProgramKey> for VectorToColumnKernel {
    fn program_key(&self) -> VectorToColumnProgramKey {
        self.key
    }

    fn build_program(&self) -> io::Result<Program> {
        vector_to_column_program(self.key)
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            INPUT_ADDR_INDEX => Some(self.input_addr),
            OUTPUT_ADDR_INDEX => Some(self.output_addr),
            _ => None,
        }
    }
}

pub(crate) fn vector_to_column(
    device: &mut Device,
    input: &DramBuffer,
    rows: usize,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "broadcast input requires {:?}, got {:?}",
            dtype, input.dtype
        )));
    }

    let tile_count = rows.max(1).div_ceil(TILE_R);
    if input.num_tiles < tile_count {
        return Err(invalid_input(format!(
            "broadcast input tile count mismatch: got {}, expected at least {tile_count}",
            input.num_tiles
        )));
    }

    let core = device
        .cores_ref()
        .first()
        .copied()
        .ok_or_else(|| invalid_input("no worker cores are available"))?;
    let output_shape = vec![round_up_to_tile(rows)?, TILE_C];
    let output = device.alloc(tile_count, dtype, &output_shape, name)?;
    let tile_count = u32::try_from(tile_count)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {tile_count}")))?;
    let kernel = VectorToColumnKernel {
        input_addr: u32_arg(input.addr, "input address")?,
        output_addr: u32_arg(output.addr, "output address")?,
        key: VectorToColumnProgramKey {
            core,
            tile_count,
            dtype,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn round_up_to_tile(value: usize) -> io::Result<usize> {
    value
        .max(1)
        .checked_next_multiple_of(TILE_R)
        .ok_or_else(|| invalid_input("shape dimension overflow"))
}

fn vector_to_column_program(key: VectorToColumnProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        Vec::new(),
        vec![INPUT_ADDR_INDEX, OUTPUT_ADDR_INDEX],
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        Vec::new(),
        vec![0, 0, 0, key.tile_count],
        Vec::new(),
    )?;
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: VECTOR_TO_COLUMN.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("broadcast_vector_to_column_{:?}", key.dtype),
        ..Program::new(runtime_args)
    })
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}
