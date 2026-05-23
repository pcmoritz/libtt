use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/binary_eltwise_reader.cc");
const COMPUTE: &str = include_str!("../../kernels/swiglu_compute.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_GATE_ADDR_INDEX: usize = 0;
const READER_UP_ADDR_INDEX: usize = 1;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SwiGLUProgramKey {
    cores: Vec<CoreCoord>,
    tile_count: u32,
}

struct SwiGLUKernel {
    gate_addr: u32,
    up_addr: u32,
    output_addr: u32,
    key: SwiGLUProgramKey,
}

impl Kernel<SwiGLUProgramKey> for SwiGLUKernel {
    fn program_key(&self) -> SwiGLUProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        swiglu_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_GATE_ADDR_INDEX => Some(self.gate_addr),
            READER_UP_ADDR_INDEX => Some(self.up_addr),
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

pub(crate) fn swiglu(
    device: &mut Device,
    gate: &DramBuffer,
    up: &DramBuffer,
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_input(gate, up, shape)?;
    let output_tiles = tiled_shape_tile_count(shape)?;
    let output_shape = tiled_allocation_shape(shape)?;
    let output = device.alloc(output_tiles, DType::Float16B, &output_shape, name)?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;

    let kernel = SwiGLUKernel {
        gate_addr: u32_addr(gate.addr, "swiglu gate address")?,
        up_addr: u32_addr(up.addr, "swiglu up address")?,
        output_addr: u32_addr(output.addr, "swiglu output address")?,
        key: SwiGLUProgramKey {
            cores,
            tile_count: u32_arg(output_tiles, "swiglu tile count")?,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_input(gate: &DramBuffer, up: &DramBuffer, shape: &[usize]) -> io::Result<()> {
    if gate.dtype != DType::Float16B || up.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "swiglu currently supports bf16 inputs, got {:?} and {:?}",
            gate.dtype, up.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(shape)?;
    if gate.shape != expected_shape || up.shape != expected_shape {
        return Err(invalid_input(format!(
            "swiglu input allocation shape mismatch: expected {:?}, got {:?} and {:?}",
            expected_shape, gate.shape, up.shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(shape)?;
    if gate.num_tiles != expected_tiles || up.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "swiglu tile count mismatch: expected {expected_tiles}, got {} and {}",
            gate.num_tiles, up.num_tiles
        )));
    }
    Ok(())
}

fn swiglu_program(key: SwiGLUProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_GATE_ADDR_INDEX, READER_UP_ADDR_INDEX],
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
        compute_kernel: COMPUTE.to_owned(),
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(2, DType::Float16B),
                CBConfig::new(3, DType::Float16B),
                CBConfig::new(4, DType::Float16B),
                CBConfig::new(5, DType::Float16B),
                CBConfig::new(16, DType::Float16B),
            ],
            ..CompileConfig::default()
        },
        name: "swiglu_bf16".to_owned(),
        ..Program::new(runtime_args)
    })
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
