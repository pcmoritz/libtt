use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, Kernel, RuntimeArgsBuilder};
use std::io;

const WRITER: &str = include_str!("../../kernels/topk_writer.cc");
const WRITER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_VALUES_ADDR_INDEX: usize = 1;
const WRITER_INDICES_ADDR_INDEX: usize = 2;
const MAX_TOP_K: usize = 32;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TopKProgramKey {
    core: CoreCoord,
    input_dtype: DType,
    input_tiles: u32,
    logical_len: u32,
    k: u32,
}

struct TopKKernel {
    input_addr: u32,
    values_addr: u32,
    indices_addr: u32,
    key: TopKProgramKey,
}

impl Kernel<TopKProgramKey> for TopKKernel {
    fn program_key(&self) -> TopKProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        top_k_program(self.key.clone())
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            WRITER_INPUT_ADDR_INDEX => Some(self.input_addr),
            WRITER_VALUES_ADDR_INDEX => Some(self.values_addr),
            WRITER_INDICES_ADDR_INDEX => Some(self.indices_addr),
            _ => None,
        }
    }
}

pub(crate) fn top_k(
    device: &mut Device,
    input: &DramBuffer,
    input_shape: &[usize],
    k: usize,
    name: impl Into<String>,
) -> io::Result<(DramBuffer, DramBuffer)> {
    validate_top_k(input, input_shape, k)?;

    let cores = select_worker_cores(device.cores_ref(), 1)?;
    let [core] = cores.as_slice() else {
        return Err(invalid_input("top_k requires one worker core"));
    };
    let output_shape = tiled_allocation_shape(&[k])?;
    let name = name.into();
    let values = device.alloc(1, input.dtype, &output_shape, format!("{name}_values"))?;
    let indices = device.alloc(1, DType::Int32, &output_shape, format!("{name}_indices"))?;
    let kernel = TopKKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        values_addr: u32_addr(values.addr, "values address")?,
        indices_addr: u32_addr(indices.addr, "indices address")?,
        key: TopKProgramKey {
            core: *core,
            input_dtype: input.dtype,
            input_tiles: u32_arg(input.num_tiles, "input tile count")?,
            logical_len: u32_arg(input_shape[0], "top_k length")?,
            k: u32_arg(k, "top_k k")?,
        },
    };
    kernel.run(device)?;
    Ok((values, indices))
}

fn validate_top_k(input: &DramBuffer, input_shape: &[usize], k: usize) -> io::Result<()> {
    if !matches!(input.dtype, DType::Float16B | DType::Float32) {
        return Err(invalid_input(format!(
            "top_k currently supports bf16 and f32 inputs, got {:?}",
            input.dtype
        )));
    }
    if input_shape.len() != 1 {
        return Err(invalid_input(format!(
            "top_k currently supports rank-1 inputs, got {input_shape:?}"
        )));
    }
    if k == 0 || k > MAX_TOP_K {
        return Err(invalid_input(format!(
            "top_k currently requires 1 <= k <= {MAX_TOP_K}, got {k}"
        )));
    }
    if k > input_shape[0] {
        return Err(invalid_input(format!(
            "top_k k must be <= input length, got k={k} length={}",
            input_shape[0]
        )));
    }
    let expected_shape = tiled_allocation_shape(input_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "top_k input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, input_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(input_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "top_k input tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn top_k_program(key: TopKProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![
            WRITER_INPUT_ADDR_INDEX,
            WRITER_VALUES_ADDR_INDEX,
            WRITER_INDICES_ADDR_INDEX,
        ],
        Vec::new(),
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        vec![0, 0, 0, key.logical_len, key.input_tiles, key.k],
        Vec::new(),
        Vec::new(),
    )?;
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        writer_kernel: top_k_writer_source(key.input_dtype)?,
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.input_dtype),
                CBConfig::new(16, key.input_dtype),
                CBConfig::new(17, DType::Int32),
            ],
            ..CompileConfig::default()
        },
        name: format!("top_k_{:?}_{}", key.input_dtype, key.k),
        ..Program::new(runtime_args)
    })
}

fn top_k_writer_source(dtype: DType) -> io::Result<String> {
    let define = match dtype {
        DType::Float16B => "TOPK_DTYPE_BFLOAT16",
        DType::Float32 => "TOPK_DTYPE_FLOAT32",
        other => {
            return Err(invalid_input(format!(
                "top_k currently does not support {other:?} inputs"
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
