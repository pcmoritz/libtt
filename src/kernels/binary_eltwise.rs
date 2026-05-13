use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::executable::CompareDirection;
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/binary_eltwise_reader.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const ADD_BF16_COMPUTE: &str = include_str!("../../kernels/add_compute.cc");
const SUBTRACT_COMPUTE: &str = include_str!("../../kernels/subtract_compute.cc");
const MULTIPLY_COMPUTE: &str = include_str!("../../kernels/multiply_compute.cc");
const DIVIDE_COMPUTE: &str = include_str!("../../kernels/divide_compute.cc");
const POWER_COMPUTE: &str = include_str!("../../kernels/power_compute.cc");
const MAX_COMPUTE: &str = include_str!("../../kernels/max_compute.cc");
const COMPARE_COMPUTE: &str = include_str!("../../kernels/compare_compute.cc");
const READER_LHS_ADDR_INDEX: usize = 0;
const READER_RHS_ADDR_INDEX: usize = 1;
const READER_LHS_CONSTANT_INDEX: usize = 4;
const READER_RHS_CONSTANT_INDEX: usize = 5;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum BinaryEltwiseOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Max,
    Compare(CompareDirection),
}

impl BinaryEltwiseOp {
    fn compute_source(self, input_dtype: DType) -> io::Result<String> {
        match self {
            Self::Add => Ok(ADD_BF16_COMPUTE.to_owned()),
            Self::Subtract => subtract_compute_source(input_dtype),
            Self::Multiply => Ok(MULTIPLY_COMPUTE.to_owned()),
            Self::Divide => divide_compute_source(input_dtype),
            Self::Power => power_compute_source(input_dtype),
            Self::Max => max_compute_source(input_dtype),
            Self::Compare(direction) => compare_compute_source(input_dtype, direction),
        }
    }

    fn kernel_name(self, input_dtype: DType, output_dtype: DType) -> String {
        match self {
            Self::Add => format!("eltwise_add_{input_dtype:?}_{output_dtype:?}"),
            Self::Subtract => format!("eltwise_subtract_{input_dtype:?}_{output_dtype:?}"),
            Self::Multiply => format!("eltwise_multiply_{input_dtype:?}_{output_dtype:?}"),
            Self::Divide => format!("eltwise_divide_{input_dtype:?}_{output_dtype:?}"),
            Self::Power => format!("eltwise_power_{input_dtype:?}_{output_dtype:?}"),
            Self::Max => format!("eltwise_max_{input_dtype:?}_{output_dtype:?}"),
            Self::Compare(direction) => {
                format!("eltwise_compare_{direction:?}_{input_dtype:?}_{output_dtype:?}")
            }
        }
    }

    fn output_dtype(self, input_dtype: DType) -> DType {
        match self {
            Self::Add => input_dtype,
            Self::Subtract => input_dtype,
            Self::Multiply => input_dtype,
            Self::Divide => input_dtype,
            Self::Power => input_dtype,
            Self::Max => input_dtype,
            Self::Compare(_) => DType::UInt8,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct BinaryEltwiseProgramKey {
    op: BinaryEltwiseOp,
    cores: Vec<CoreCoord>,
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
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        eltwise_program(self.key.clone())
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
    let output_tiles = tiled_shape_tile_count(shape)?;
    validate_input(lhs, input_dtype, shape, output_tiles, "lhs")?;
    validate_input(rhs, input_dtype, shape, output_tiles, "rhs")?;

    let lhs_addr = input_addr(lhs, "lhs address")?;
    let rhs_addr = input_addr(rhs, "rhs address")?;
    let tile_count = u32::try_from(output_tiles)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {output_tiles}")))?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output_dtype = op.output_dtype(input_dtype);
    let output_shape = tiled_allocation_shape(shape)?;
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
            cores,
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
    let expected_shape = tiled_allocation_shape(shape)?;
    if buffer.shape != expected_shape {
        return Err(invalid_input(format!(
            "{name} allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            buffer.shape, expected_shape, shape
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
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, 0, offset, n_tiles, 0, 0],
            vec![n_tiles],
        )?;
    }
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

fn divide_compute_source(dtype: DType) -> io::Result<String> {
    if !matches!(dtype, DType::Float16 | DType::Float16B | DType::Float32) {
        return Err(invalid_input(format!(
            "divide currently supports Float16, Float16B, and Float32 inputs, got {dtype:?}"
        )));
    }
    Ok(DIVIDE_COMPUTE.to_owned())
}

fn subtract_compute_source(dtype: DType) -> io::Result<String> {
    if !matches!(
        dtype,
        DType::Float16 | DType::Float16B | DType::Float32 | DType::Int32
    ) {
        return Err(invalid_input(format!(
            "subtract currently supports Float16, Float16B, Float32, and Int32 inputs, got {dtype:?}"
        )));
    }
    Ok(SUBTRACT_COMPUTE.to_owned())
}

fn power_compute_source(dtype: DType) -> io::Result<String> {
    if !matches!(dtype, DType::Float16 | DType::Float16B | DType::Float32) {
        return Err(invalid_input(format!(
            "power currently supports Float16, Float16B, and Float32 inputs, got {dtype:?}"
        )));
    }
    Ok(POWER_COMPUTE.to_owned())
}

fn max_compute_source(dtype: DType) -> io::Result<String> {
    if !matches!(dtype, DType::Float16 | DType::Float16B | DType::Float32) {
        return Err(invalid_input(format!(
            "max currently supports Float16, Float16B, and Float32 inputs, got {dtype:?}"
        )));
    }
    Ok(MAX_COMPUTE.to_owned())
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

    fn arg_u32(blob: &[u8], index: usize) -> u32 {
        let start = index * std::mem::size_of::<u32>();
        u32::from_le_bytes(
            blob[start..start + std::mem::size_of::<u32>()]
                .try_into()
                .unwrap(),
        )
    }

    #[test]
    fn binary_eltwise_program_splits_tiles_across_cores() {
        let program = eltwise_program(BinaryEltwiseProgramKey {
            op: BinaryEltwiseOp::Add,
            cores: vec![
                CoreCoord { x: 1, y: 2 },
                CoreCoord { x: 1, y: 3 },
                CoreCoord { x: 1, y: 4 },
            ],
            tile_count: 10,
            input_dtype: DType::Float16B,
            output_dtype: DType::Float16B,
        })
        .expect("binary eltwise program");

        assert_eq!(program.runtime_args.cores().len(), 3);
        assert_eq!(program.runtime_args.section_sizes(), (12, 24, 4));

        let blobs = program.runtime_args.blobs();
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 4));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (4, 3));
        assert_eq!((arg_u32(&blobs[2], 1), arg_u32(&blobs[2], 2)), (7, 3));

        assert_eq!((arg_u32(&blobs[0], 5), arg_u32(&blobs[0], 6)), (0, 4));
        assert_eq!((arg_u32(&blobs[1], 5), arg_u32(&blobs[1], 6)), (4, 3));
        assert_eq!((arg_u32(&blobs[2], 5), arg_u32(&blobs[2], 6)), (7, 3));

        assert_eq!(arg_u32(&blobs[0], 9), 4);
        assert_eq!(arg_u32(&blobs[1], 9), 3);
        assert_eq!(arg_u32(&blobs[2], 9), 3);
    }
}
