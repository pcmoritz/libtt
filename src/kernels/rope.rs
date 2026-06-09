use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{
    select_worker_cores, split_tile_range, DramKernel, Kernel, RuntimeArgsBuilder,
};
use std::io;

const READER: &str = include_str!("../../kernels/rope_reader.cc");
const COMPUTE: &str = include_str!("../../kernels/rope_compute.cc");
const WRITER: &str = include_str!("../../kernels/tile_writer.cc");

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RopeProgramKey {
    cores: Vec<CoreCoord>,
    output_tiles: u32,
    output_tile_rows: u32,
    tiles_per_row: u32,
    half_tiles: u32,
}

pub(crate) fn rope(
    device: &mut Device,
    input: &DramBuffer,
    cos: &DramBuffer,
    sin: &DramBuffer,
    input_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
    output_shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let shape = validate_rope_shapes(
        input,
        cos,
        sin,
        input_shape,
        cos_shape,
        sin_shape,
        output_shape,
    )?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles = tiled_shape_tile_count(output_shape)?;
    let output = device.alloc(
        output_tiles,
        DType::Float16B,
        &output_allocation_shape,
        name,
    )?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;

    let key = RopeProgramKey {
        cores,
        output_tiles: u32_arg(output_tiles, "rope output tiles")?,
        output_tile_rows: u32_arg(shape.output_tile_rows, "rope output tile rows")?,
        tiles_per_row: u32_arg(shape.tiles_per_row, "rope tiles per row")?,
        half_tiles: u32_arg(shape.half_tiles, "rope half tiles")?,
    };
    let kernel = DramKernel {
        reader_addrs: [
            u32_arg(input.addr, "rope input address")?,
            u32_arg(cos.addr, "rope cos address")?,
            u32_arg(sin.addr, "rope sin address")?,
        ],
        output_addr: u32_arg(output.addr, "rope output address")?,
        key,
        build: rope_program,
    };
    kernel.run(device)?;
    Ok(output)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RopeShape {
    output_tile_rows: usize,
    tiles_per_row: usize,
    half_tiles: usize,
}

fn validate_rope_shapes(
    input: &DramBuffer,
    cos: &DramBuffer,
    sin: &DramBuffer,
    input_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
    output_shape: &[usize],
) -> io::Result<RopeShape> {
    if input.dtype != DType::Float16B
        || cos.dtype != DType::Float16B
        || sin.dtype != DType::Float16B
    {
        return Err(invalid_input(format!(
            "rope requires bf16 input/cos/sin, got {:?}/{:?}/{:?}",
            input.dtype, cos.dtype, sin.dtype
        )));
    }
    if input_shape != output_shape || input_shape.len() != 3 {
        return Err(invalid_input(format!(
            "rope requires rank-3 input/output with identical shapes, got input={input_shape:?} output={output_shape:?}"
        )));
    }
    let [tokens, heads, head_dim]: [usize; 3] = input_shape.try_into().expect("rank checked");
    if tokens == 0 || heads == 0 || head_dim == 0 || head_dim % (2 * TILE_C) != 0 {
        return Err(invalid_input(format!(
            "rope requires non-empty shape with head_dim divisible by {}, got {input_shape:?}",
            2 * TILE_C
        )));
    }
    let half_dim = head_dim / 2;
    let expected_trig_shape = [tokens, half_dim];
    if cos_shape != expected_trig_shape || sin_shape != expected_trig_shape {
        return Err(invalid_input(format!(
            "rope requires cos/sin shapes {:?}, got cos={cos_shape:?} sin={sin_shape:?}",
            expected_trig_shape
        )));
    }

    validate_tiled_buffer(input, input_shape, "input")?;
    validate_tiled_buffer(cos, cos_shape, "cos")?;
    validate_tiled_buffer(sin, sin_shape, "sin")?;

    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let rank = output_allocation_shape.len();
    Ok(RopeShape {
        output_tile_rows: output_allocation_shape[rank - 2] / TILE_R,
        tiles_per_row: output_allocation_shape[rank - 1] / TILE_C,
        half_tiles: half_dim / TILE_C,
    })
}

fn validate_tiled_buffer(
    buffer: &DramBuffer,
    logical_shape: &[usize],
    name: &str,
) -> io::Result<()> {
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    let expected_tiles = tiled_shape_tile_count(logical_shape)?;
    if buffer.shape != expected_shape || buffer.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "rope {name} allocation mismatch: got shape {:?} tiles {}, expected shape {:?} tiles {}",
            buffer.shape, buffer.num_tiles, expected_shape, expected_tiles
        )));
    }
    Ok(())
}

fn rope_program(key: RopeProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(0, vec![0], vec![0, 1, 2], Vec::new());
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.output_tiles, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, 0, 0, offset, n_tiles],
            vec![offset, n_tiles],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    let defines = format!(
        "#define ROPE_TILES_PER_ROW {}\n#define ROPE_OUTPUT_TILE_ROWS {}\n#define ROPE_HALF_TILES {}\n",
        key.tiles_per_row, key.output_tile_rows, key.half_tiles
    );

    Ok(Program {
        reader_kernel: format!("{defines}{READER}"),
        compute_kernel: format!("{defines}{COMPUTE}"),
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, DType::Float16B),
                CBConfig::new(1, DType::Float16B),
                CBConfig::new(2, DType::Float16B),
                CBConfig::new(3, DType::Float16B),
                CBConfig::new(16, DType::Float16B),
                CBConfig::new(24, DType::Float16B),
                CBConfig::new(25, DType::Float16B),
            ],
            dst_full_sync: true,
            ..CompileConfig::default()
        },
        name: format!(
            "rope_d{}_half{}_rows{}",
            key.tiles_per_row * TILE_C as u32,
            key.half_tiles,
            key.output_tile_rows
        ),
        ..Program::new(runtime_args)
    })
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg<T>(value: T, name: &str) -> io::Result<u32>
where
    T: TryInto<u32> + Copy,
{
    value
        .try_into()
        .map_err(|_| invalid_input(format!("{name} is out of u32 range")))
}
