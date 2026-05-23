use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/rope_reader.cc");
const COMPUTE: &str = include_str!("../../kernels/rope_compute.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const READER_COS_ADDR_INDEX: usize = 1;
const READER_SIN_ADDR_INDEX: usize = 2;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RopeShape {
    logical_rows: u32,
    width_tiles: u32,
    tile_count: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RopeProgramKey {
    cores: Vec<CoreCoord>,
    shape: RopeShape,
}

struct RopeKernel {
    input_addr: u32,
    cos_addr: u32,
    sin_addr: u32,
    output_addr: u32,
    key: RopeProgramKey,
}

impl Kernel<RopeProgramKey> for RopeKernel {
    fn program_key(&self) -> RopeProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        rope_program(self.key.clone())
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
        match index {
            WRITER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            _ => None,
        }
    }
}

pub(crate) fn rope_decode(
    device: &mut Device,
    input: &DramBuffer,
    input_shape: &[usize],
    cos: &DramBuffer,
    cos_shape: &[usize],
    sin: &DramBuffer,
    sin_shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let shape = validate_and_shape(input, input_shape, cos, cos_shape, sin, sin_shape)?;
    let output_shape = tiled_allocation_shape(input_shape)?;
    let output_tiles = usize::try_from(shape.tile_count).map_err(|_| {
        invalid_input(format!(
            "rope output tile count does not fit in usize: {}",
            shape.tile_count
        ))
    })?;
    let output = device.alloc(output_tiles, input.dtype, &output_shape, name)?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let kernel = RopeKernel {
        input_addr: u32_addr(input.addr, "rope input address")?,
        cos_addr: u32_addr(cos.addr, "rope cos address")?,
        sin_addr: u32_addr(sin.addr, "rope sin address")?,
        output_addr: u32_addr(output.addr, "rope output address")?,
        key: RopeProgramKey { cores, shape },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_and_shape(
    input: &DramBuffer,
    input_shape: &[usize],
    cos: &DramBuffer,
    cos_shape: &[usize],
    sin: &DramBuffer,
    sin_shape: &[usize],
) -> io::Result<RopeShape> {
    if input.dtype != DType::Float16B
        || cos.dtype != DType::Float16B
        || sin.dtype != DType::Float16B
    {
        return Err(invalid_input(format!(
            "rope_decode currently supports bf16 inputs, got {:?}, {:?}, {:?}",
            input.dtype, cos.dtype, sin.dtype
        )));
    }
    if input_shape.len() != 3 {
        return Err(invalid_input(format!(
            "rope_decode currently expects [token, head, dim], got {input_shape:?}"
        )));
    }
    let token_count = input_shape[0];
    let logical_rows = input_shape[1];
    let dim = input_shape[2];
    if logical_rows > TILE_R {
        return Err(invalid_input(format!(
            "rope_decode currently supports at most {TILE_R} rows, got {logical_rows}"
        )));
    }
    if dim == 0 || dim % (2 * TILE_C) != 0 {
        return Err(invalid_input(format!(
            "rope_decode dimension must be a nonzero multiple of {}, got {dim}",
            2 * TILE_C
        )));
    }
    if cos_shape != [token_count, dim] || sin_shape != [token_count, dim] {
        return Err(invalid_input(format!(
            "rope_decode cos/sin shapes must both be [{token_count}, {dim}], got {cos_shape:?} and {sin_shape:?}"
        )));
    }
    let expected_input_shape = tiled_allocation_shape(input_shape)?;
    let expected_rope_shape = tiled_allocation_shape(cos_shape)?;
    if input.shape != expected_input_shape {
        return Err(invalid_input(format!(
            "rope_decode input allocation shape mismatch: got {:?}, expected {:?}",
            input.shape, expected_input_shape
        )));
    }
    if cos.shape != expected_rope_shape || sin.shape != expected_rope_shape {
        return Err(invalid_input(format!(
            "rope_decode cos/sin allocation shapes must be {:?}, got {:?} and {:?}",
            expected_rope_shape, cos.shape, sin.shape
        )));
    }
    let input_tiles = tiled_shape_tile_count(input_shape)?;
    let rope_tiles = tiled_shape_tile_count(cos_shape)?;
    if input.num_tiles != input_tiles || cos.num_tiles != rope_tiles || sin.num_tiles != rope_tiles
    {
        return Err(invalid_input(
            "rope_decode buffer tile count does not match logical shape",
        ));
    }
    Ok(RopeShape {
        logical_rows: u32_arg(logical_rows, "rope logical rows")?,
        width_tiles: u32_arg(expected_input_shape[2] / TILE_C, "rope width tiles")?,
        tile_count: u32_arg(input_tiles, "rope tile count")?,
    })
}

fn rope_program(key: RopeProgramKey) -> io::Result<Program> {
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
        let (offset, n_tiles) =
            split_tile_range(key.shape.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, 0, 0, offset, n_tiles],
            vec![offset, n_tiles],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: rope_reader_source(key.shape),
        compute_kernel: rope_compute_source(key.shape),
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(2, DType::Float16B),
                CBConfig::new(3, DType::Float16B),
                CBConfig::new(16, DType::Float16B),
            ],
            ..CompileConfig::default()
        },
        name: format!(
            "rope_decode_bf16_{}_{}",
            key.shape.logical_rows, key.shape.width_tiles
        ),
        ..Program::new(runtime_args)
    })
}

fn rope_reader_source(shape: RopeShape) -> String {
    format!(
        "#define ROPE_WIDTH_TILES {}\n#define ROPE_LOGICAL_ROWS {}\n{READER}",
        shape.width_tiles, shape.logical_rows
    )
}

fn rope_compute_source(shape: RopeShape) -> String {
    format!("#define ROPE_WIDTH_TILES {}\n{COMPUTE}", shape.width_tiles)
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
