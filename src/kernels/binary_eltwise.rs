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
const COMPARE_COMPUTE: &str = include_str!("../../kernels/compare_compute.cc");
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
    CompareEq,
    CompareNe,
    CompareGe,
    CompareGt,
    CompareLe,
    CompareLt,
}

impl BinaryEltwiseOp {
    fn compute_source(self, input_dtype: DType) -> io::Result<String> {
        match self {
            Self::Add => Ok(ADD_BF16_COMPUTE.to_owned()),
            Self::Max => Ok(MAX_BF16_COMPUTE.to_owned()),
            Self::CompareEq
            | Self::CompareNe
            | Self::CompareGe
            | Self::CompareGt
            | Self::CompareLe
            | Self::CompareLt => compare_compute_source(input_dtype, self),
        }
    }

    fn kernel_name(self, input_dtype: DType, output_dtype: DType) -> String {
        match self {
            Self::Add => "eltwise_add_bf16".to_owned(),
            Self::Max => "eltwise_max_bf16".to_owned(),
            Self::CompareEq => format!("eltwise_compare_eq_{input_dtype:?}_{output_dtype:?}"),
            Self::CompareNe => format!("eltwise_compare_ne_{input_dtype:?}_{output_dtype:?}"),
            Self::CompareGe => format!("eltwise_compare_ge_{input_dtype:?}_{output_dtype:?}"),
            Self::CompareGt => format!("eltwise_compare_gt_{input_dtype:?}_{output_dtype:?}"),
            Self::CompareLe => format!("eltwise_compare_le_{input_dtype:?}_{output_dtype:?}"),
            Self::CompareLt => format!("eltwise_compare_lt_{input_dtype:?}_{output_dtype:?}"),
        }
    }

    fn output_dtype(self) -> DType {
        match self {
            Self::Add | Self::Max => DType::Float16B,
            Self::CompareEq
            | Self::CompareNe
            | Self::CompareGe
            | Self::CompareGt
            | Self::CompareLe
            | Self::CompareLt => DType::UInt8,
        }
    }

    fn compare_zero_op(self) -> io::Result<CompareZeroOp> {
        match self {
            Self::CompareEq => Ok(CompareZeroOp {
                init: "eqz_tile_init",
                float_tile: "eqz_tile",
                int32_tile: "eqz_tile_int32",
            }),
            Self::CompareNe => Ok(CompareZeroOp {
                init: "nez_tile_init",
                float_tile: "nez_tile",
                int32_tile: "nez_tile_int32",
            }),
            Self::CompareGe => Ok(CompareZeroOp {
                init: "gez_tile_init",
                float_tile: "gez_tile",
                int32_tile: "gez_tile_int32",
            }),
            Self::CompareGt => Ok(CompareZeroOp {
                init: "gtz_tile_init",
                float_tile: "gtz_tile",
                int32_tile: "gtz_tile_int32",
            }),
            Self::CompareLe => Ok(CompareZeroOp {
                init: "lez_tile_init",
                float_tile: "lez_tile",
                int32_tile: "lez_tile_int32",
            }),
            Self::CompareLt => Ok(CompareZeroOp {
                init: "ltz_tile_init",
                float_tile: "ltz_tile",
                int32_tile: "ltz_tile_int32",
            }),
            Self::Add | Self::Max => Err(invalid_input("op is not a compare")),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct BinaryEltwiseProgramKey {
    op: BinaryEltwiseOp,
    core: CoreCoord,
    tile_count: u32,
    input_dtype: DType,
    output_dtype: DType,
}

struct BinaryEltwiseKernel {
    lhs_addr: u32,
    rhs_addr: u32,
    lhs_constant: Option<u32>,
    rhs_constant: Option<u32>,
    output_addr: u32,
    key: BinaryEltwiseProgramKey,
}

impl Kernel<BinaryEltwiseProgramKey> for BinaryEltwiseKernel {
    fn program_key(&self) -> BinaryEltwiseProgramKey {
        self.key
    }

    fn build_program(&self) -> io::Result<Program> {
        eltwise_program(self.key)
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
pub(crate) enum EltwiseInput<'a> {
    Dram(&'a DramBuffer),
    Constant(u32),
}

pub(crate) fn eltwise(
    device: &mut Device,
    op: BinaryEltwiseOp,
    lhs: EltwiseInput<'_>,
    rhs: EltwiseInput<'_>,
    input_dtype: DType,
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let output_tiles = shape_tile_count(shape)?;
    validate_input(lhs, input_dtype, shape, output_tiles, "lhs")?;
    validate_input(rhs, input_dtype, shape, output_tiles, "rhs")?;

    let lhs_addr = input_addr(lhs, "lhs address")?;
    let rhs_addr = input_addr(rhs, "rhs address")?;
    let tile_count = u32::try_from(output_tiles)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {output_tiles}")))?;
    let core = device
        .cores_ref()
        .first()
        .copied()
        .ok_or_else(|| invalid_input("no worker cores are available"))?;
    let output_dtype = op.output_dtype();
    let output_shape = allocation_shape(shape)?;
    let output = device.alloc(output_tiles, output_dtype, &output_shape, name)?;
    let output_addr = u32_arg(output.addr, "output address")?;

    let kernel = BinaryEltwiseKernel {
        lhs_addr,
        rhs_addr,
        lhs_constant: input_constant(lhs),
        rhs_constant: input_constant(rhs),
        output_addr,
        key: BinaryEltwiseProgramKey {
            op,
            core,
            tile_count,
            input_dtype,
            output_dtype,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_input(
    input: EltwiseInput<'_>,
    dtype: DType,
    shape: &[usize],
    expected_tiles: usize,
    name: &str,
) -> io::Result<()> {
    let EltwiseInput::Dram(buffer) = input else {
        return Ok(());
    };
    if buffer.dtype != dtype {
        return Err(invalid_input(format!(
            "{name} requires {:?} input, got {:?}",
            dtype, buffer.dtype
        )));
    }
    if !buffer_shape_matches(&buffer.shape, shape)? {
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

fn buffer_shape_matches(buffer_shape: &[usize], logical_shape: &[usize]) -> io::Result<bool> {
    if buffer_shape == logical_shape {
        return Ok(true);
    }
    Ok(buffer_shape == allocation_shape(logical_shape)?.as_slice())
}

fn allocation_shape(shape: &[usize]) -> io::Result<Vec<usize>> {
    match shape.len() {
        0 => Ok(vec![TILE_R, TILE_C]),
        1 => Ok(vec![TILE_R, round_up_to_tile_dim(shape[0])?]),
        _ => Ok(shape.to_vec()),
    }
}

fn round_up_to_tile_dim(dim: usize) -> io::Result<usize> {
    dim.max(1)
        .checked_add(TILE_C - 1)
        .map(|value| value / TILE_C * TILE_C)
        .ok_or_else(|| invalid_input("shape dimension overflow"))
}

fn input_addr(input: EltwiseInput<'_>, name: &str) -> io::Result<u32> {
    match input {
        EltwiseInput::Dram(buffer) => u32_arg(buffer.addr, name),
        EltwiseInput::Constant(_) => Ok(0),
    }
}

fn input_constant(input: EltwiseInput<'_>) -> Option<u32> {
    match input {
        EltwiseInput::Dram(_) => None,
        EltwiseInput::Constant(value) => Some(value),
    }
}

#[allow(clippy::manual_is_multiple_of)]
fn shape_tile_count(shape: &[usize]) -> io::Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    if shape.len() == 1 {
        return Ok(shape[0].div_ceil(TILE_C));
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

fn eltwise_program(key: BinaryEltwiseProgramKey) -> io::Result<Program> {
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
        reader_kernel: reader_source(key.input_dtype)?,
        compute_kernel: key.op.compute_source(key.input_dtype)?,
        writer_kernel: writer_source(key.output_dtype)?,
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.input_dtype),
                CBConfig::new(1, key.input_dtype),
                CBConfig::new(16, key.output_dtype),
            ],
            dst_accum_mode: matches!(
                key.input_dtype,
                DType::Int32 | DType::UInt32 | DType::Float32
            ),
            ..CompileConfig::default()
        },
        name: key.op.kernel_name(key.input_dtype, key.output_dtype),
        ..Program::new(runtime_args)
    })
}

fn reader_source(dtype: DType) -> io::Result<String> {
    Ok(BF16_READER
        .replace("DataFormat::Float16_b", data_format(dtype)?)
        .replace("packed_bf16", "packed_value"))
}

fn writer_source(dtype: DType) -> io::Result<String> {
    Ok(BF16_WRITER.replace("DataFormat::Float16_b", data_format(dtype)?))
}

fn compare_compute_source(dtype: DType, op: BinaryEltwiseOp) -> io::Result<String> {
    let zero_op = op.compare_zero_op()?;
    let substitutions = match dtype {
        DType::Float16B | DType::Float32 => CompareComputeFns {
            sub_init: "sub_binary_tile_init",
            sub_tile: "sub_binary_tile",
            zero_tile: zero_op.float_tile,
        },
        DType::Int32 => CompareComputeFns {
            sub_init: "sub_int_tile_init",
            sub_tile: "sub_int32_tile",
            zero_tile: zero_op.int32_tile,
        },
        _ => {
            return Err(invalid_input(format!(
                "compare currently supports Float16B, Float32, and Int32 inputs, got {dtype:?}"
            )))
        }
    };
    Ok(COMPARE_COMPUTE
        .replace("COMPARE_SUB_INIT", substitutions.sub_init)
        .replace("COMPARE_SUB_TILE", substitutions.sub_tile)
        .replace("COMPARE_ZERO_INIT", zero_op.init)
        .replace("COMPARE_ZERO_TILE", substitutions.zero_tile))
}

struct CompareComputeFns {
    sub_init: &'static str,
    sub_tile: &'static str,
    zero_tile: &'static str,
}

struct CompareZeroOp {
    init: &'static str,
    float_tile: &'static str,
    int32_tile: &'static str,
}

fn data_format(dtype: DType) -> io::Result<&'static str> {
    match dtype {
        DType::Float32 => Ok("DataFormat::Float32"),
        DType::Float16 => Ok("DataFormat::Float16"),
        DType::Float16B => Ok("DataFormat::Float16_b"),
        DType::Int32 => Ok("DataFormat::Int32"),
        DType::UInt16 => Ok("DataFormat::UInt16"),
        DType::Int8 => Ok("DataFormat::Int8"),
        DType::UInt32 => Ok("DataFormat::UInt32"),
        DType::UInt8 => Ok("DataFormat::UInt8"),
    }
}
