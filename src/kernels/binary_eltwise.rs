use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::io;

const BF16_READER: &str = include_str!("../../kernels/binary_eltwise_reader.cc");
const BF16_WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const ADD_BF16_COMPUTE: &str = include_str!("../../kernels/add_compute.cc");
const MAX_BF16_COMPUTE: &str = include_str!("../../kernels/max_compute.cc");
const READER_LHS_ADDR_INDEX: usize = 0;
const READER_RHS_ADDR_INDEX: usize = 1;
const READER_LHS_SINGLE_TILE_INDEX: usize = 4;
const READER_RHS_SINGLE_TILE_INDEX: usize = 5;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum BinaryEltwiseOp {
    Add,
    Max,
}

impl BinaryEltwiseOp {
    fn compute_source(self) -> &'static str {
        match self {
            Self::Add => ADD_BF16_COMPUTE,
            Self::Max => MAX_BF16_COMPUTE,
        }
    }

    fn kernel_name(self) -> &'static str {
        match self {
            Self::Add => "eltwise_add_bf16",
            Self::Max => "eltwise_max_bf16",
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct BinaryEltwiseProgramKey {
    op: BinaryEltwiseOp,
    core: CoreCoord,
    tile_count: u32,
}

struct BinaryEltwiseBf16Kernel {
    lhs_addr: u32,
    rhs_addr: u32,
    lhs_single_tile: bool,
    rhs_single_tile: bool,
    output_addr: u32,
    key: BinaryEltwiseProgramKey,
}

impl Kernel<BinaryEltwiseProgramKey> for BinaryEltwiseBf16Kernel {
    fn program_key(&self) -> BinaryEltwiseProgramKey {
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
            READER_LHS_SINGLE_TILE_INDEX => Some(u32::from(self.lhs_single_tile)),
            READER_RHS_SINGLE_TILE_INDEX => Some(u32::from(self.rhs_single_tile)),
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

pub(crate) fn eltwise_bf16(
    device: &mut Device,
    op: BinaryEltwiseOp,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if lhs.dtype != DType::Float16B || rhs.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "{} requires bf16 inputs, got {:?} and {:?}",
            op.kernel_name(),
            lhs.dtype,
            rhs.dtype
        )));
    }
    if lhs.shape != rhs.shape {
        return Err(invalid_input(format!(
            "input shapes must match, got {:?} and {:?}",
            lhs.shape, rhs.shape
        )));
    }
    let output_tiles = lhs.validate_single_or_logical_tile_count("lhs")?;
    rhs.validate_single_or_logical_tile_count("rhs")?;

    let lhs_addr = u32_arg(lhs.addr, "lhs address")?;
    let rhs_addr = u32_arg(rhs.addr, "rhs address")?;
    let tile_count = u32::try_from(output_tiles)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {output_tiles}")))?;
    let core = device
        .cores_ref()
        .first()
        .copied()
        .ok_or_else(|| invalid_input("no worker cores are available"))?;
    let output = device.alloc(output_tiles, DType::Float16B, &lhs.shape, name)?;
    let output_addr = u32_arg(output.addr, "output address")?;

    let kernel = BinaryEltwiseBf16Kernel {
        lhs_addr,
        rhs_addr,
        lhs_single_tile: lhs.num_tiles == 1,
        rhs_single_tile: rhs.num_tiles == 1,
        output_addr,
        key: BinaryEltwiseProgramKey {
            op,
            core,
            tile_count,
        },
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

fn bf16_program(key: BinaryEltwiseProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![
            READER_LHS_ADDR_INDEX,
            READER_RHS_ADDR_INDEX,
            READER_LHS_SINGLE_TILE_INDEX,
            READER_RHS_SINGLE_TILE_INDEX,
        ],
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        vec![0, 0, key.tile_count],
        vec![0, 0, 0, key.tile_count, 0, 0],
        vec![key.tile_count],
    )?;
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: BF16_READER.to_owned(),
        compute_kernel: key.op.compute_source().to_owned(),
        writer_kernel: BF16_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(16, DType::Float16B),
            ],
            ..CompileConfig::default()
        },
        name: key.op.kernel_name().to_owned(),
        ..Program::new(runtime_args)
    })
}
