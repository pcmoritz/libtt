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
const READER_LHS_CONSTANT_INDEX: usize = 4;
const READER_RHS_CONSTANT_INDEX: usize = 5;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;
const TILE_R: usize = 32;
const TILE_C: usize = 32;

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
    lhs_constant: Option<u32>,
    rhs_constant: Option<u32>,
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
            READER_LHS_CONSTANT_INDEX => Some(self.lhs_constant.unwrap_or(0)),
            READER_RHS_CONSTANT_INDEX => Some(self.rhs_constant.unwrap_or(0)),
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

#[derive(Clone, Copy)]
pub(crate) enum Bf16EltwiseInput<'a> {
    Dram(&'a DramBuffer),
    Constant(u32),
}

pub(crate) fn eltwise_bf16(
    device: &mut Device,
    op: BinaryEltwiseOp,
    lhs: Bf16EltwiseInput<'_>,
    rhs: Bf16EltwiseInput<'_>,
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let output_tiles = shape_tile_count(shape)?;
    validate_input(lhs, shape, output_tiles, "lhs")?;
    validate_input(rhs, shape, output_tiles, "rhs")?;

    let lhs_addr = input_addr(lhs, "lhs address")?;
    let rhs_addr = input_addr(rhs, "rhs address")?;
    let tile_count = u32::try_from(output_tiles)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {output_tiles}")))?;
    let core = device
        .cores_ref()
        .first()
        .copied()
        .ok_or_else(|| invalid_input("no worker cores are available"))?;
    let output = device.alloc(output_tiles, DType::Float16B, shape, name)?;
    let output_addr = u32_arg(output.addr, "output address")?;

    let kernel = BinaryEltwiseBf16Kernel {
        lhs_addr,
        rhs_addr,
        lhs_constant: input_constant(lhs),
        rhs_constant: input_constant(rhs),
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

fn validate_input(
    input: Bf16EltwiseInput<'_>,
    shape: &[usize],
    expected_tiles: usize,
    name: &str,
) -> io::Result<()> {
    let Bf16EltwiseInput::Dram(buffer) = input else {
        return Ok(());
    };
    if buffer.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "{name} requires bf16 input, got {:?}",
            buffer.dtype
        )));
    }
    if buffer.shape != shape {
        return Err(invalid_input(format!(
            "{name} shape mismatch: got {:?}, expected {:?}",
            buffer.shape, shape
        )));
    }
    if buffer.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "{name} tile count mismatch: got {}, expected {expected_tiles}",
            buffer.num_tiles
        )));
    }
    Ok(())
}

fn input_addr(input: Bf16EltwiseInput<'_>, name: &str) -> io::Result<u32> {
    match input {
        Bf16EltwiseInput::Dram(buffer) => u32_arg(buffer.addr, name),
        Bf16EltwiseInput::Constant(_) => Ok(0),
    }
}

fn input_constant(input: Bf16EltwiseInput<'_>) -> Option<u32> {
    match input {
        Bf16EltwiseInput::Dram(_) => None,
        Bf16EltwiseInput::Constant(value) => Some(value),
    }
}

#[allow(clippy::manual_is_multiple_of)]
fn shape_tile_count(shape: &[usize]) -> io::Result<usize> {
    if shape.len() < 2 {
        return Err(invalid_input("shape must have at least two dimensions"));
    }
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    if rows % TILE_R != 0 || cols % TILE_C != 0 {
        return Err(invalid_input(format!(
            "shape rows/cols must be multiples of {TILE_R}x{TILE_C}"
        )));
    }
    let tiles_per_batch = (rows / TILE_R)
        .checked_mul(cols / TILE_C)
        .ok_or_else(|| invalid_input("shape tile count is too large"))?;
    shape[..shape.len() - 2]
        .iter()
        .try_fold(tiles_per_batch, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| invalid_input("shape tile count is too large"))
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
            READER_LHS_CONSTANT_INDEX,
            READER_RHS_CONSTANT_INDEX,
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
