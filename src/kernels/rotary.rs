use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/rotary_reader.cc");
const COMPUTE: &str = include_str!("../../kernels/rotary_compute.cc");
const WRITER: &str = include_str!("../../kernels/tile_writer.cc");

const READER_INPUT_ADDR_INDEX: usize = 0;
const READER_COS_ADDR_INDEX: usize = 1;
const READER_SIN_ADDR_INDEX: usize = 2;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RotaryProgramKey {
    cores: Vec<CoreCoord>,
    heads: u32,
    half_dim: u32,
    tile_count: u32,
}

struct RotaryKernel {
    input_addr: u32,
    cos_addr: u32,
    sin_addr: u32,
    output_addr: u32,
    key: RotaryProgramKey,
}

impl Kernel<RotaryProgramKey> for RotaryKernel {
    fn program_key(&self) -> RotaryProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        rotary_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_INPUT_ADDR_INDEX => Some(self.input_addr),
            READER_COS_ADDR_INDEX => Some(self.cos_addr),
            READER_SIN_ADDR_INDEX => Some(self.sin_addr),
            _ => None,
        }
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        (index == WRITER_OUTPUT_ADDR_INDEX).then_some(self.output_addr)
    }
}

pub(crate) fn rotary(
    device: &mut Device,
    input: &DramBuffer,
    cos: &DramBuffer,
    sin: &DramBuffer,
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_rotary_input(input, cos, sin, shape)?;
    let output_tiles = tiled_shape_tile_count(shape)?;
    let output_shape = tiled_allocation_shape(shape)?;
    let output = device.alloc(output_tiles, DType::Float16B, &output_shape, name)?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let heads = shape[shape.len() - 2];
    let half_dim = shape[shape.len() - 1] / 2;
    let kernel = RotaryKernel {
        input_addr: u32_addr(input.addr, "rotary input address")?,
        cos_addr: u32_addr(cos.addr, "rotary cos address")?,
        sin_addr: u32_addr(sin.addr, "rotary sin address")?,
        output_addr: u32_addr(output.addr, "rotary output address")?,
        key: RotaryProgramKey {
            cores,
            heads: u32_arg(heads, "rotary head count")?,
            half_dim: u32_arg(half_dim, "rotary half dimension")?,
            tile_count: u32_arg(output_tiles, "rotary tile count")?,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_rotary_input(
    input: &DramBuffer,
    cos: &DramBuffer,
    sin: &DramBuffer,
    shape: &[usize],
) -> io::Result<()> {
    if input.dtype != DType::Float16B
        || cos.dtype != DType::Float16B
        || sin.dtype != DType::Float16B
    {
        return Err(invalid_input(format!(
            "rotary requires BF16 input/cos/sin, got input={:?} cos={:?} sin={:?}",
            input.dtype, cos.dtype, sin.dtype
        )));
    }
    if shape.len() < 2 || shape[shape.len() - 1] % (2 * TILE_C) != 0 {
        return Err(invalid_input(format!(
            "rotary requires rank >= 2 and last dimension divisible by {}, got {shape:?}",
            2 * TILE_C
        )));
    }
    let heads = shape[shape.len() - 2];
    let half_dim = shape[shape.len() - 1] / 2;
    if half_dim == 0 {
        return Err(invalid_input("rotary half dimension must be nonzero"));
    }
    let expected_input_shape = tiled_allocation_shape(shape)?;
    if input.shape != expected_input_shape {
        return Err(invalid_input(format!(
            "rotary input allocation shape mismatch: got {:?}, expected {:?}",
            input.shape, expected_input_shape
        )));
    }
    let mut scale_shape = shape.to_vec();
    let last = scale_shape.len() - 1;
    scale_shape[last] = half_dim;
    let expected_scale_shape = tiled_allocation_shape(&scale_shape)?;
    if cos.shape != expected_scale_shape || sin.shape != expected_scale_shape {
        return Err(invalid_input(format!(
            "rotary cos/sin allocation shape mismatch: got cos={:?} sin={:?}, expected {:?}",
            cos.shape, sin.shape, expected_scale_shape
        )));
    }
    let input_tiles = tiled_shape_tile_count(shape)?;
    let scale_tiles = tiled_shape_tile_count(&scale_shape)?;
    if input.num_tiles != input_tiles
        || cos.num_tiles != scale_tiles
        || sin.num_tiles != scale_tiles
    {
        return Err(invalid_input(format!(
            "rotary tile count mismatch: input {} expected {}, cos {} sin {} expected {}",
            input.num_tiles, input_tiles, cos.num_tiles, sin.num_tiles, scale_tiles
        )));
    }
    if heads == 0 {
        return Err(invalid_input("rotary head count must be nonzero"));
    }
    Ok(())
}

fn rotary_program(key: RotaryProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![
            READER_INPUT_ADDR_INDEX,
            READER_COS_ADDR_INDEX,
            READER_SIN_ADDR_INDEX,
        ],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, 0, 0, offset, n_tiles],
            vec![n_tiles],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: format!(
            "#define ROTARY_HEADS {}u\n#define ROTARY_HALF_DIM {}u\n{}",
            key.heads, key.half_dim, READER
        ),
        compute_kernel: COMPUTE.to_owned(),
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(2, DType::Float16B),
                CBConfig::new(3, DType::Float16B),
                CBConfig::new(4, DType::Float16B).with_tiles(2),
                CBConfig::new(5, DType::Float16B).with_tiles(2),
                CBConfig::new(16, DType::Float16B).with_tiles(2),
            ],
            dst_full_sync: true,
            ..CompileConfig::default()
        },
        name: format!("rotary_bf16_{}_{}", key.heads, key.half_dim * 2),
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
