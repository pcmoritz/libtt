use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_shape_tile_count, DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const WRITER: &str = include_str!("../../kernels/constant_writer.cc");
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;
const WRITER_PACKED_WORD_INDEX: usize = 1;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ConstantProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    tile_count: u32,
}

struct ConstantKernel {
    output_addr: u32,
    packed_word: u32,
    key: ConstantProgramKey,
}

impl Kernel<ConstantProgramKey> for ConstantKernel {
    fn program_key(&self) -> ConstantProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        constant_program(self.key.clone())
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            WRITER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            WRITER_PACKED_WORD_INDEX => Some(self.packed_word),
            _ => None,
        }
    }
}

pub(crate) fn splat_constant(
    device: &mut Device,
    dtype: DType,
    packed_value: u32,
    allocation_shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let tile_count = tiled_shape_tile_count(allocation_shape)?;
    let output = device.alloc(tile_count, dtype, allocation_shape, name)?;
    let cores = select_worker_cores(device.cores_ref(), tile_count)?;
    let kernel = ConstantKernel {
        output_addr: u32_addr(output.addr, "constant output address")?,
        packed_word: repeated_word(dtype, packed_value),
        key: ConstantProgramKey {
            cores,
            dtype,
            tile_count: u32_arg(tile_count, "constant tile count")?,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn constant_program(key: ConstantProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX, WRITER_PACKED_WORD_INDEX],
        Vec::new(),
        Vec::new(),
    );
    for (core_index, core) in key.cores.iter().enumerate() {
        let (tile_offset, tile_count) =
            split_tile_range(key.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            *core,
            vec![0, 0, tile_offset, tile_count],
            Vec::new(),
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("constant_{:?}_{}", key.dtype, key.cores.len()),
        ..Program::new(runtime_args)
    })
}

fn repeated_word(dtype: DType, packed_value: u32) -> u32 {
    match dtype.bytes_per_element() {
        1 => {
            let byte = packed_value & 0xff;
            byte | (byte << 8) | (byte << 16) | (byte << 24)
        }
        2 => {
            let half = packed_value & 0xffff;
            half | (half << 16)
        }
        _ => packed_value,
    }
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
