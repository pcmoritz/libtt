use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, Kernel, RuntimeArgsBuilder};
use std::io;

const WRITER: &str = include_str!("../../kernels/topk_writer.cc");
const TOP1_FINAL_WRITER: &str = include_str!("../../kernels/top1_final_writer.cc");
const WRITER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_VALUES_ADDR_INDEX: usize = 1;
const WRITER_INDICES_ADDR_INDEX: usize = 2;
const TOP1_FINAL_PARTIAL_VALUES_ADDR_INDEX: usize = 0;
const TOP1_FINAL_PARTIAL_INDICES_ADDR_INDEX: usize = 1;
const TOP1_FINAL_VALUES_ADDR_INDEX: usize = 2;
const TOP1_FINAL_INDICES_ADDR_INDEX: usize = 3;
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Top1FinalProgramKey {
    core: CoreCoord,
    input_dtype: DType,
    partial_count: u32,
}

struct Top1FinalKernel {
    partial_values_addr: u32,
    partial_indices_addr: u32,
    values_addr: u32,
    indices_addr: u32,
    key: Top1FinalProgramKey,
}

impl Kernel<Top1FinalProgramKey> for Top1FinalKernel {
    fn program_key(&self) -> Top1FinalProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        top1_final_program(self.key.clone())
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            TOP1_FINAL_PARTIAL_VALUES_ADDR_INDEX => Some(self.partial_values_addr),
            TOP1_FINAL_PARTIAL_INDICES_ADDR_INDEX => Some(self.partial_indices_addr),
            TOP1_FINAL_VALUES_ADDR_INDEX => Some(self.values_addr),
            TOP1_FINAL_INDICES_ADDR_INDEX => Some(self.indices_addr),
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

pub(crate) fn top1_finalize_partials(
    device: &mut Device,
    partial_values: &DramBuffer,
    partial_indices: &DramBuffer,
    partial_count: usize,
    name: impl Into<String>,
) -> io::Result<(DramBuffer, DramBuffer)> {
    if partial_values.dtype != DType::Float16B && partial_values.dtype != DType::Float32 {
        return Err(invalid_input(format!(
            "top1 final currently supports bf16 and f32 values, got {:?}",
            partial_values.dtype
        )));
    }
    if partial_indices.dtype != DType::Int32 {
        return Err(invalid_input(format!(
            "top1 final indices must be Int32, got {:?}",
            partial_indices.dtype
        )));
    }
    if partial_values.num_tiles != partial_count || partial_indices.num_tiles != partial_count {
        return Err(invalid_input(format!(
            "top1 final partial tile count mismatch: expected {partial_count}, got {} and {}",
            partial_values.num_tiles, partial_indices.num_tiles
        )));
    }

    let cores = select_worker_cores(device.cores_ref(), 1)?;
    let [core] = cores.as_slice() else {
        return Err(invalid_input("top1 final requires one worker core"));
    };
    let output_shape = tiled_allocation_shape(&[1])?;
    let name = name.into();
    let values = device.alloc(
        1,
        partial_values.dtype,
        &output_shape,
        format!("{name}_values"),
    )?;
    let indices = device.alloc(1, DType::Int32, &output_shape, format!("{name}_indices"))?;
    let final_kernel = Top1FinalKernel {
        partial_values_addr: u32_addr(partial_values.addr, "top1 partial values address")?,
        partial_indices_addr: u32_addr(partial_indices.addr, "top1 partial indices address")?,
        values_addr: u32_addr(values.addr, "values address")?,
        indices_addr: u32_addr(indices.addr, "indices address")?,
        key: Top1FinalProgramKey {
            core: *core,
            input_dtype: partial_values.dtype,
            partial_count: u32_arg(partial_count, "top1 partial count")?,
        },
    };
    final_kernel.run(device)?;
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

fn top1_final_program(key: Top1FinalProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![
            TOP1_FINAL_PARTIAL_VALUES_ADDR_INDEX,
            TOP1_FINAL_PARTIAL_INDICES_ADDR_INDEX,
            TOP1_FINAL_VALUES_ADDR_INDEX,
            TOP1_FINAL_INDICES_ADDR_INDEX,
        ],
        Vec::new(),
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        vec![0, 0, 0, 0, key.partial_count],
        Vec::new(),
        Vec::new(),
    )?;
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        writer_kernel: top1_writer_source(key.input_dtype, TOP1_FINAL_WRITER)?,
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.input_dtype),
                CBConfig::new(1, DType::Int32),
                CBConfig::new(16, key.input_dtype),
                CBConfig::new(17, DType::Int32),
            ],
            ..CompileConfig::default()
        },
        name: format!("top1_final_{:?}_{}", key.input_dtype, key.partial_count),
        ..Program::new(runtime_args)
    })
}

fn top_k_writer_source(dtype: DType) -> io::Result<String> {
    top1_writer_source(dtype, WRITER)
}

fn top1_writer_source(dtype: DType, writer: &str) -> io::Result<String> {
    let define = match dtype {
        DType::Float16B => "TOPK_DTYPE_BFLOAT16",
        DType::Float32 => "TOPK_DTYPE_FLOAT32",
        other => {
            return Err(invalid_input(format!(
                "top_k currently does not support {other:?} inputs"
            )));
        }
    };
    Ok(format!("#define {define}\n{writer}"))
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
