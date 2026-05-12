use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::binary_eltwise::EltwiseInput;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/unary_eltwise_reader.cc");
const COMPUTE: &str = include_str!("../../kernels/unary_eltwise_compute.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const READER_INPUT_CONSTANT_INDEX: usize = 3;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum UnaryEltwiseOp {
    Cosine,
    Sine,
    Rsqrt,
    Negate,
    Exponential,
    Convert,
}

impl UnaryEltwiseOp {
    fn compute_source(self, input_dtype: DType, output_dtype: DType) -> io::Result<String> {
        let (header, init, tile) = match self {
            Self::Cosine | Self::Sine | Self::Rsqrt | Self::Negate | Self::Exponential => {
                if input_dtype != output_dtype {
                    return Err(invalid_input(format!(
                        "{self:?} output dtype must match input dtype, got {input_dtype:?} -> {output_dtype:?}"
                    )));
                }
                if !matches!(
                    input_dtype,
                    DType::Float16 | DType::Float16B | DType::Float32
                ) {
                    return Err(invalid_input(format!(
                        "{self:?} currently supports Float16, Float16B, and Float32 inputs, got {input_dtype:?}"
                    )));
                }
                self.unary_op_source()
            }
            Self::Convert => convert_source(input_dtype, output_dtype)?,
        };
        Ok(COMPUTE
            .replace("UNARY_OP_HEADER", header)
            .replace("UNARY_OP_INIT", &init)
            .replace("UNARY_OP_TILE", &tile))
    }

    fn unary_op_source(self) -> (&'static str, String, String) {
        match self {
            Self::Cosine => (
                "compute_kernel_api/eltwise_unary/trigonometry.h",
                "cos_tile_init()".to_owned(),
                "cos_tile(0)".to_owned(),
            ),
            Self::Sine => (
                "compute_kernel_api/eltwise_unary/trigonometry.h",
                "sin_tile_init()".to_owned(),
                "sin_tile(0)".to_owned(),
            ),
            Self::Rsqrt => (
                "compute_kernel_api/eltwise_unary/rsqrt.h",
                "rsqrt_tile_init()".to_owned(),
                "rsqrt_tile(0)".to_owned(),
            ),
            Self::Negate => (
                "compute_kernel_api/eltwise_unary/negative.h",
                "negative_tile_init()".to_owned(),
                "negative_tile(0)".to_owned(),
            ),
            Self::Exponential => (
                "compute_kernel_api/eltwise_unary/exp.h",
                "exp_tile_init()".to_owned(),
                "exp_tile(0)".to_owned(),
            ),
            Self::Convert => unreachable!("convert source is generated separately"),
        }
    }

    fn kernel_name(self, input_dtype: DType, output_dtype: DType) -> String {
        match self {
            Self::Cosine => format!("eltwise_cosine_{input_dtype:?}"),
            Self::Sine => format!("eltwise_sine_{input_dtype:?}"),
            Self::Rsqrt => format!("eltwise_rsqrt_{input_dtype:?}"),
            Self::Negate => format!("eltwise_negate_{input_dtype:?}"),
            Self::Exponential => format!("eltwise_exponential_{input_dtype:?}"),
            Self::Convert => format!("eltwise_convert_{input_dtype:?}_{output_dtype:?}"),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct UnaryEltwiseProgramKey {
    op: UnaryEltwiseOp,
    cores: Vec<CoreCoord>,
    tile_count: u32,
    input_dtype: DType,
    output_dtype: DType,
}

struct UnaryEltwiseKernel {
    input_addr: u32,
    input_constant: Option<u32>,
    output_addr: u32,
    key: UnaryEltwiseProgramKey,
}

impl Kernel<UnaryEltwiseProgramKey> for UnaryEltwiseKernel {
    fn program_key(&self) -> UnaryEltwiseProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        eltwise_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_INPUT_ADDR_INDEX => Some(self.input_addr),
            READER_INPUT_CONSTANT_INDEX => Some(self.input_constant.unwrap_or(0)),
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

pub(crate) fn eltwise(
    device: &mut Device,
    op: UnaryEltwiseOp,
    input: EltwiseInput<'_>,
    input_dtype: DType,
    output_dtype: DType,
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let output_tiles = tiled_shape_tile_count(shape)?;
    validate_input(input, input_dtype, shape, output_tiles, "input")?;

    let input_addr = input_addr(input, "input address")?;
    let tile_count = u32::try_from(output_tiles)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {output_tiles}")))?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output_shape = tiled_allocation_shape(shape)?;
    let output = device.alloc(output_tiles, output_dtype, &output_shape, name)?;
    let output_addr = u32_arg(output.addr, "output address")?;

    let kernel = UnaryEltwiseKernel {
        input_addr,
        input_constant: input_constant(input),
        output_addr,
        key: UnaryEltwiseProgramKey {
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

fn eltwise_program(key: UnaryEltwiseProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX, READER_INPUT_CONSTANT_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, offset, n_tiles, 0],
            vec![n_tiles],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: READER.to_owned(),
        compute_kernel: key.op.compute_source(key.input_dtype, key.output_dtype)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.input_dtype),
                CBConfig::new(16, key.output_dtype),
            ],
            dst_accum_mode: matches!(
                key.input_dtype,
                DType::Int32 | DType::UInt32 | DType::Float32
            ) || matches!(
                key.output_dtype,
                DType::Int32 | DType::UInt32 | DType::Float32
            ),
            ..CompileConfig::default()
        },
        name: key.op.kernel_name(key.input_dtype, key.output_dtype),
        ..Program::new(runtime_args)
    })
}

fn convert_source(
    input_dtype: DType,
    output_dtype: DType,
) -> io::Result<(&'static str, String, String)> {
    let input = typecast_data_format(input_dtype)?;
    let output = typecast_data_format(output_dtype)?;
    Ok((
        "compute_kernel_api/eltwise_unary/typecast.h",
        format!("typecast_tile_init<{input}, {output}>()"),
        format!("typecast_tile<{input}, {output}>(0)"),
    ))
}

fn typecast_data_format(dtype: DType) -> io::Result<u32> {
    match dtype {
        DType::Float16B | DType::Float32 | DType::Int32 | DType::UInt16 | DType::UInt32 => {
            Ok(dtype as u32)
        }
        _ => Err(invalid_input(format!(
            "convert currently supports Float16B, Float32, Int32, UInt16, and UInt32, got {dtype:?}"
        ))),
    }
}
