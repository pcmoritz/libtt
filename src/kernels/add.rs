use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, CoreSelection, Program};
use crate::dram::{DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::io;

const BF16_READER: &str = include_str!("../../kernels/add_reader.cc");
const BF16_WRITER: &str = include_str!("../../kernels/add_writer.cc");
const BF16_COMPUTE: &str = include_str!("../../kernels/add_compute.cc");
const READER_LHS_ADDR_INDEX: usize = 0;
const READER_RHS_ADDR_INDEX: usize = 1;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct AddProgramKey {
    core: CoreCoord,
    tile_count: u32,
}

struct AddBf16Kernel {
    lhs_addr: u32,
    rhs_addr: u32,
    output_addr: u32,
    key: AddProgramKey,
}

impl Kernel<AddProgramKey> for AddBf16Kernel {
    fn program_key(&self) -> AddProgramKey {
        self.key
    }

    fn build_program(&self) -> io::Result<Program> {
        bf16_program(self.key)
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_LHS_ADDR_INDEX => Some(self.lhs_addr),
            READER_RHS_ADDR_INDEX => Some(self.rhs_addr),
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

pub(crate) fn eltwise_add_bf16(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if lhs.dtype != DType::Float16B || rhs.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "eltwise_add_bf16 requires bf16 inputs, got {:?} and {:?}",
            lhs.dtype, rhs.dtype
        )));
    }
    if lhs.num_tiles != rhs.num_tiles {
        return Err(invalid_input(format!(
            "input tile counts must match, got {} and {}",
            lhs.num_tiles, rhs.num_tiles
        )));
    }
    if lhs.shape != rhs.shape {
        return Err(invalid_input(format!(
            "input shapes must match, got {:?} and {:?}",
            lhs.shape, rhs.shape
        )));
    }

    let lhs_addr = u32_arg(lhs.addr, "lhs address")?;
    let rhs_addr = u32_arg(rhs.addr, "rhs address")?;
    let tile_count = u32::try_from(lhs.num_tiles)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {}", lhs.num_tiles)))?;
    let core = device
        .cores_ref()
        .first()
        .copied()
        .ok_or_else(|| invalid_input("no worker cores are available"))?;
    let output = device.alloc(lhs.num_tiles, DType::Float16B, &lhs.shape, name)?;
    let output_addr = u32_arg(output.addr, "output address")?;

    let kernel = AddBf16Kernel {
        lhs_addr,
        rhs_addr,
        output_addr,
        key: AddProgramKey { core, tile_count },
    };
    kernel.run(device)?;
    Ok(output)
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}

fn bf16_program(key: AddProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_LHS_ADDR_INDEX, READER_RHS_ADDR_INDEX],
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        vec![0, 0, key.tile_count],
        vec![0, 0, 0, key.tile_count],
        vec![key.tile_count],
    )?;
    let (runtime_args, writer_args, reader_args, compute_args, semaphores) =
        runtime_args.into_program_parts()?;
    Ok(Program {
        cores: CoreSelection::Count(1),
        reader_kernel: BF16_READER.to_owned(),
        compute_kernel: BF16_COMPUTE.to_owned(),
        writer_kernel: BF16_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(16, DType::Float16B),
            ],
            ..CompileConfig::default()
        },
        name: "eltwise_add_bf16".to_owned(),
        reader_args,
        writer_args,
        compute_args,
        semaphores,
        ..Program::new(runtime_args)
    })
}
