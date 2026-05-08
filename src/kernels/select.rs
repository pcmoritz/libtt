use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{DType, DramBuffer};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/select_reader.cc");
const COMPUTE: &str = include_str!("../../kernels/select_compute.cc");
const WRITER: &str = include_str!("../../kernels/binary_eltwise_writer.cc");
const READER_PRED_ADDR_INDEX: usize = 0;
const READER_TRUE_ADDR_INDEX: usize = 1;
const READER_FALSE_ADDR_INDEX: usize = 2;
const READER_TRUE_CONSTANT_INDEX: usize = 5;
const READER_FALSE_CONSTANT_INDEX: usize = 6;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;
const TILE_R: usize = 32;
const TILE_C: usize = 32;

#[derive(Clone, Copy)]
pub(crate) enum SelectInput<'a> {
    Dram(&'a DramBuffer),
    Constant(u32),
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SelectProgramKey {
    core: CoreCoord,
    tile_count: u32,
    value_dtype: DType,
}

struct SelectKernel {
    pred_addr: u32,
    true_addr: u32,
    false_addr: u32,
    true_constant: Option<u32>,
    false_constant: Option<u32>,
    output_addr: u32,
    key: SelectProgramKey,
}

impl Kernel<SelectProgramKey> for SelectKernel {
    fn program_key(&self) -> SelectProgramKey {
        self.key
    }

    fn build_program(&self) -> io::Result<Program> {
        select_program(self.key)
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_PRED_ADDR_INDEX => Some(self.pred_addr),
            READER_TRUE_ADDR_INDEX => Some(self.true_addr),
            READER_FALSE_ADDR_INDEX => Some(self.false_addr),
            READER_TRUE_CONSTANT_INDEX => Some(self.true_constant.unwrap_or(0)),
            READER_FALSE_CONSTANT_INDEX => Some(self.false_constant.unwrap_or(0)),
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

pub(crate) fn select(
    device: &mut Device,
    pred: &DramBuffer,
    on_true: SelectInput<'_>,
    on_false: SelectInput<'_>,
    value_dtype: DType,
    shape: &[usize],
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let output_tiles = shape_tile_count(shape)?;
    validate_pred(pred, shape, output_tiles)?;
    validate_value(on_true, value_dtype, shape, output_tiles, "on_true")?;
    validate_value(on_false, value_dtype, shape, output_tiles, "on_false")?;

    let tile_count = u32::try_from(output_tiles)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {output_tiles}")))?;
    let core = device
        .cores_ref()
        .first()
        .copied()
        .ok_or_else(|| invalid_input("no worker cores are available"))?;
    let output_shape = allocation_shape(shape)?;
    let output = device.alloc(output_tiles, value_dtype, &output_shape, name)?;
    let kernel = SelectKernel {
        pred_addr: u32_arg(pred.addr, "predicate address")?,
        true_addr: input_addr(on_true, "true address")?,
        false_addr: input_addr(on_false, "false address")?,
        true_constant: input_constant(on_true),
        false_constant: input_constant(on_false),
        output_addr: u32_arg(output.addr, "output address")?,
        key: SelectProgramKey {
            core,
            tile_count,
            value_dtype,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_pred(pred: &DramBuffer, shape: &[usize], expected_tiles: usize) -> io::Result<()> {
    if pred.dtype != DType::UInt8 {
        return Err(invalid_input(format!(
            "predicate requires UInt8 input, got {:?}",
            pred.dtype
        )));
    }
    if !buffer_shape_matches(&pred.shape, shape)? {
        return Err(invalid_input(format!(
            "predicate shape mismatch: got {:?}, expected {:?}",
            pred.shape, shape
        )));
    }
    if pred.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "predicate tile count mismatch: got {}, expected {expected_tiles}",
            pred.num_tiles
        )));
    }
    Ok(())
}

fn validate_value(
    input: SelectInput<'_>,
    dtype: DType,
    shape: &[usize],
    expected_tiles: usize,
    name: &str,
) -> io::Result<()> {
    let SelectInput::Dram(buffer) = input else {
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

fn input_addr(input: SelectInput<'_>, name: &str) -> io::Result<u32> {
    match input {
        SelectInput::Dram(buffer) => u32_arg(buffer.addr, name),
        SelectInput::Constant(_) => Ok(0),
    }
}

fn input_constant(input: SelectInput<'_>) -> Option<u32> {
    match input {
        SelectInput::Dram(_) => None,
        SelectInput::Constant(value) => Some(value),
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

fn select_program(key: SelectProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![
            READER_PRED_ADDR_INDEX,
            READER_TRUE_ADDR_INDEX,
            READER_FALSE_ADDR_INDEX,
            READER_TRUE_CONSTANT_INDEX,
            READER_FALSE_CONSTANT_INDEX,
        ],
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        vec![0, 0, key.tile_count],
        vec![0, 0, 0, 0, key.tile_count, 0, 0],
        vec![key.tile_count],
    )?;
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: READER.to_owned(),
        compute_kernel: COMPUTE.to_owned(),
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(1, key.value_dtype),
                CBConfig::new(2, key.value_dtype),
                CBConfig::new(3, key.value_dtype),
                CBConfig::new(4, key.value_dtype),
                CBConfig::new(16, key.value_dtype),
            ],
            dst_accum_mode: matches!(
                key.value_dtype,
                DType::Int32 | DType::UInt32 | DType::Float32
            ),
            ..CompileConfig::default()
        },
        name: format!("select_{:?}", key.value_dtype),
        ..Program::new(runtime_args)
    })
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}
