use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, Kernel, RuntimeArgsBuilder};
use std::io;

const WRITER: &str = include_str!("../../kernels/argmax_writer.cc");
const WRITER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 1;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ArgMaxProgramKey {
    core: CoreCoord,
    input_dtype: DType,
    input_tiles: u32,
    logical_len: u32,
}

struct ArgMaxKernel {
    input_addr: u32,
    output_addr: u32,
    key: ArgMaxProgramKey,
}

impl Kernel<ArgMaxProgramKey> for ArgMaxKernel {
    fn program_key(&self) -> ArgMaxProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        argmax_program(self.key.clone())
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            WRITER_INPUT_ADDR_INDEX => Some(self.input_addr),
            WRITER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            _ => None,
        }
    }
}

pub(crate) fn argmax(
    device: &mut Device,
    input: &DramBuffer,
    input_shape: &[usize],
    dimensions: &[i64],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_argmax(input, input_shape, dimensions)?;

    let cores = select_worker_cores(device.cores_ref(), 1)?;
    let [core] = cores.as_slice() else {
        return Err(invalid_input("argmax requires one worker core"));
    };
    let output_shape = tiled_allocation_shape(&[])?;
    let output = device.alloc(1, DType::Int32, &output_shape, name)?;
    let kernel = ArgMaxKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: ArgMaxProgramKey {
            core: *core,
            input_dtype: input.dtype,
            input_tiles: u32_arg(input.num_tiles, "input tile count")?,
            logical_len: u32_arg(input_shape[0], "argmax length")?,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_argmax(
    input: &DramBuffer,
    input_shape: &[usize],
    dimensions: &[i64],
) -> io::Result<()> {
    if !matches!(input.dtype, DType::Float16B | DType::Float32) {
        return Err(invalid_input(format!(
            "argmax currently supports bf16 and f32 inputs, got {:?}",
            input.dtype
        )));
    }
    if input_shape.len() != 1 {
        return Err(invalid_input(format!(
            "argmax currently supports rank-1 inputs, got {input_shape:?}"
        )));
    }
    if dimensions != [0] {
        return Err(invalid_input(format!(
            "argmax currently supports only dimension [0], got {dimensions:?}"
        )));
    }
    let expected_shape = tiled_allocation_shape(input_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "argmax input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, input_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(input_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "argmax input tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn argmax_program(key: ArgMaxProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_INPUT_ADDR_INDEX, WRITER_OUTPUT_ADDR_INDEX],
        Vec::new(),
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        vec![0, 0, key.logical_len, key.input_tiles],
        Vec::new(),
        Vec::new(),
    )?;
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        writer_kernel: argmax_writer_source(key.input_dtype)?,
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.input_dtype),
                CBConfig::new(16, DType::Int32),
            ],
            ..CompileConfig::default()
        },
        name: format!("argmax_{:?}", key.input_dtype),
        ..Program::new(runtime_args)
    })
}

fn argmax_writer_source(dtype: DType) -> io::Result<String> {
    let define = match dtype {
        DType::Float16B => "ARGMAX_DTYPE_BFLOAT16",
        DType::Float32 => "ARGMAX_DTYPE_FLOAT32",
        other => {
            return Err(invalid_input(format!(
                "argmax currently does not support {other:?} inputs"
            )));
        }
    };
    Ok(format!("#define {define}\n{WRITER}"))
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
