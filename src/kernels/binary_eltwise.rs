use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{DType, DramBuffer};
use crate::executable::CompareDirection;
use crate::hw::CoreCoord;
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/binary_eltwise_reader.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
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
    Compare(CompareDirection),
}

impl BinaryEltwiseOp {
    fn compute_source(self, input_dtype: DType) -> io::Result<String> {
        match self {
            Self::Add => Ok(ADD_BF16_COMPUTE.to_owned()),
            Self::Max => Ok(MAX_BF16_COMPUTE.to_owned()),
            Self::Compare(direction) => compare_compute_source(input_dtype, direction),
        }
    }

    fn kernel_name(self, input_dtype: DType, output_dtype: DType) -> String {
        match self {
            Self::Add => "eltwise_add_bf16".to_owned(),
            Self::Max => "eltwise_max_bf16".to_owned(),
            Self::Compare(direction) => {
                format!("eltwise_compare_{direction:?}_{input_dtype:?}_{output_dtype:?}")
            }
        }
    }

    fn output_dtype(self) -> DType {
        match self {
            Self::Add | Self::Max => DType::Float16B,
            Self::Compare(_) => DType::UInt8,
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
        1 => Ok(vec![
            TILE_R,
            shape[0]
                .max(1)
                .checked_next_multiple_of(TILE_C)
                .ok_or_else(|| invalid_input("shape dimension overflow"))?,
        ]),
        _ => Ok(shape.to_vec()),
    }
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
        reader_kernel: READER.to_owned(),
        compute_kernel: key.op.compute_source(key.input_dtype)?,
        writer_kernel: WRITER.to_owned(),
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

fn compare_compute_source(dtype: DType, direction: CompareDirection) -> io::Result<String> {
    let int32_input = match dtype {
        DType::Int32 => "true",
        DType::Float16B | DType::Float32 => "false",
        _ => {
            return Err(invalid_input(format!(
                "compare currently supports Float16B, Float32, and Int32 inputs, got {dtype:?}"
            )))
        }
    };
    Ok(COMPARE_COMPUTE
        .replace("COMPARE_INT32_INPUT", int32_input)
        .replace("COMPARE_DIRECTION", compare_direction_variant(direction)))
}

fn compare_direction_variant(direction: CompareDirection) -> &'static str {
    match direction {
        CompareDirection::Eq => "Eq",
        CompareDirection::Ne => "Ne",
        CompareDirection::Ge => "Ge",
        CompareDirection::Gt => "Gt",
        CompareDirection::Le => "Le",
        CompareDirection::Lt => "Lt",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_compute_source_dispatches_dtype_and_direction_in_cpp() {
        let source = compare_compute_source(DType::Int32, CompareDirection::Lt)
            .expect("compare source should support int32");
        let float_source = compare_compute_source(DType::Float32, CompareDirection::Lt)
            .expect("compare source should support float32");

        assert_ne!(source, float_source);
        assert!(source.contains("constexpr bool int32_input = true"));
        assert!(float_source.contains("constexpr bool int32_input = false"));
        assert!(source.contains("compare_sub_init<int32_input>()"));
        assert!(source.contains("compare_sub_tile<int32_input>(0, 1, 0)"));
        assert!(source.contains("compare_zero_init(direction)"));
        assert!(source.contains("compare_zero_tile<int32_input>(direction, 0)"));
        assert!(source.contains("CompareDirection::Lt"));
        assert!(!source.contains("COMPARE_DIRECTION"));
        assert!(!source.contains("COMPARE_INT32_INPUT"));
    }
}
