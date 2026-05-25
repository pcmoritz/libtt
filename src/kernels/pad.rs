use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const PAD_READER: &str = include_str!("../../kernels/pad_reader.cc");
const PAD_WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const READER_PADDING_VALUE_INDEX: usize = 1;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct PadKernelShape {
    input_shape: Vec<u32>,
    output_shape: Vec<u32>,
    edge_padding_low: Vec<u32>,
    interior_padding: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    tile_count: u32,
    direct_copy: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PadPlan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_allocation_shape: Vec<usize>,
    kernel_shape: PadKernelShape,
}

impl PadPlan {
    pub(crate) fn new(
        input_shape: &[usize],
        output_shape: &[usize],
        edge_padding_low: &[i64],
        edge_padding_high: &[i64],
        interior_padding: &[i64],
    ) -> io::Result<Self> {
        validate_pad(
            input_shape,
            output_shape,
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        )?;

        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let kernel_shape = pad_kernel_shape(
            input_shape,
            output_shape,
            edge_padding_low,
            interior_padding,
        )?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            kernel_shape,
        })
    }

    fn kernel_shape(&self) -> &PadKernelShape {
        &self.kernel_shape
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PadProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: PadKernelShape,
}

struct PadKernel {
    input_addr: u32,
    padding_value_addr: u32,
    output_addr: u32,
    key: PadProgramKey,
}

impl Kernel<PadProgramKey> for PadKernel {
    fn program_key(&self) -> PadProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        pad_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_INPUT_ADDR_INDEX => Some(self.input_addr),
            READER_PADDING_VALUE_INDEX => Some(self.padding_value_addr),
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

pub(crate) fn pad(
    device: &mut Device,
    input: &DramBuffer,
    padding_value: &DramBuffer,
    plan: &PadPlan,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "pad input requires {:?}, got {:?}",
            dtype, input.dtype
        )));
    }
    let expected_input_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_input_shape {
        return Err(invalid_input(format!(
            "pad input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_input_shape, plan.input_shape
        )));
    }

    let input_tile_count = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != input_tile_count {
        return Err(invalid_input(format!(
            "pad input tile count mismatch: got {}, expected {input_tile_count}",
            input.num_tiles
        )));
    }
    validate_padding_value(padding_value, dtype)?;

    let shape = plan.kernel_shape().clone();
    let output_tiles = usize::try_from(shape.tile_count).map_err(|_| {
        invalid_input(format!(
            "pad tile count does not fit in usize: {}",
            shape.tile_count
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(output_tiles, dtype, &plan.output_allocation_shape, name)?;
    let kernel = PadKernel {
        input_addr: u32_addr(input.addr, "pad input address")?,
        padding_value_addr: u32_addr(padding_value.addr, "pad padding value address")?,
        output_addr: u32_addr(output.addr, "pad output address")?,
        key: PadProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_padding_value(buffer: &DramBuffer, dtype: DType) -> io::Result<()> {
    if buffer.dtype != dtype {
        return Err(invalid_input(format!(
            "pad padding value requires {:?}, got {:?}",
            dtype, buffer.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(&[])?;
    if buffer.shape != expected_shape {
        return Err(invalid_input(format!(
            "pad padding value must be scalar allocation {:?}, got {:?}",
            expected_shape, buffer.shape
        )));
    }
    if buffer.num_tiles != 1 {
        return Err(invalid_input(format!(
            "pad padding value must contain one tile, got {}",
            buffer.num_tiles
        )));
    }
    Ok(())
}

fn pad_kernel_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    edge_padding_low: &[i64],
    interior_padding: &[i64],
) -> io::Result<PadKernelShape> {
    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let input_rank = input_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    let tile_count = tiled_shape_tile_count(output_shape)?;
    let edge_padding_low_u32 = u32_indices(edge_padding_low, "edge padding low")?;
    let interior_padding_u32 = u32_indices(interior_padding, "interior padding")?;
    let direct_copy = input_shape == output_shape
        && edge_padding_low_u32.iter().all(|&value| value == 0)
        && interior_padding_u32.iter().all(|&value| value == 0);

    Ok(PadKernelShape {
        input_shape: u32_shape(input_shape, "pad input shape")?,
        output_shape: u32_shape(output_shape, "pad output shape")?,
        edge_padding_low: edge_padding_low_u32,
        interior_padding: interior_padding_u32,
        input_tile_rows: u32_arg(
            input_allocation_shape[input_rank - 2] / TILE_R,
            "pad input tile rows",
        )?,
        input_tiles_per_row: u32_arg(
            input_allocation_shape[input_rank - 1] / TILE_C,
            "pad input tiles per row",
        )?,
        output_tile_rows: u32_arg(
            output_allocation_shape[output_rank - 2] / TILE_R,
            "pad output tile rows",
        )?,
        output_tiles_per_row: u32_arg(
            output_allocation_shape[output_rank - 1] / TILE_C,
            "pad output tiles per row",
        )?,
        tile_count: u32_arg(tile_count, "pad tile count")?,
        direct_copy,
    })
}

fn pad_program(key: PadProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX, READER_PADDING_VALUE_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, 0, offset, n_tiles],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: pad_reader_source(key.dtype, &key.shape)?,
        writer_kernel: PAD_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.dtype),
                CBConfig::new(1, key.dtype),
                CBConfig::new(16, key.dtype),
            ],
            ..CompileConfig::default()
        },
        name: format!("pad_{:?}_{}", key.dtype, key.shape.input_shape.len()),
        ..Program::new(runtime_args)
    })
}

fn pad_reader_source(dtype: DType, shape: &PadKernelShape) -> io::Result<String> {
    let element_type = element_type(dtype);
    Ok(format!(
        "#define PAD_RANK {}\n\
         #define PAD_INPUT_SHAPE {}\n\
         #define PAD_OUTPUT_SHAPE {}\n\
         #define PAD_EDGE_PADDING_LOW {}\n\
         #define PAD_INTERIOR_PADDING {}\n\
         #define PAD_INPUT_TILE_ROWS {}\n\
         #define PAD_INPUT_TILES_PER_ROW {}\n\
         #define PAD_OUTPUT_TILE_ROWS {}\n\
         #define PAD_OUTPUT_TILES_PER_ROW {}\n\
         #define PAD_DIRECT_COPY {}\n\
         #define PAD_ELEMENT_TYPE {element_type}\n\
         {PAD_READER}",
        shape.input_shape.len(),
        cpp_u32_array(&shape.input_shape),
        cpp_u32_array(&shape.output_shape),
        cpp_u32_array(&shape.edge_padding_low),
        cpp_u32_array(&shape.interior_padding),
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
        shape.direct_copy as u32,
    ))
}

fn validate_pad(
    input_shape: &[usize],
    output_shape: &[usize],
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_padding: &[i64],
) -> io::Result<()> {
    let rank = input_shape.len();
    if output_shape.len() != rank {
        return Err(invalid_input(format!(
            "pad output rank {} must match input rank {rank}",
            output_shape.len()
        )));
    }
    if edge_padding_low.len() != rank
        || edge_padding_high.len() != rank
        || interior_padding.len() != rank
    {
        return Err(invalid_input(format!(
            "pad attribute lengths must match rank {rank}: low={}, high={}, interior={}",
            edge_padding_low.len(),
            edge_padding_high.len(),
            interior_padding.len()
        )));
    }

    for dim in 0..rank {
        let low = usize::try_from(edge_padding_low[dim]).map_err(|_| {
            invalid_input(format!(
                "pad edge_padding_low {dim} must be non-negative, got {}",
                edge_padding_low[dim]
            ))
        })?;
        let high = usize::try_from(edge_padding_high[dim]).map_err(|_| {
            invalid_input(format!(
                "pad edge_padding_high {dim} must be non-negative, got {}",
                edge_padding_high[dim]
            ))
        })?;
        let interior = usize::try_from(interior_padding[dim]).map_err(|_| {
            invalid_input(format!(
                "pad interior_padding {dim} must be non-negative, got {}",
                interior_padding[dim]
            ))
        })?;
        let interior_total = input_shape[dim].saturating_sub(1).saturating_mul(interior);
        let expected = low
            .checked_add(input_shape[dim])
            .and_then(|value| value.checked_add(interior_total))
            .and_then(|value| value.checked_add(high))
            .ok_or_else(|| invalid_input(format!("pad output dimension {dim} overflow")))?;
        if output_shape[dim] != expected {
            return Err(invalid_input(format!(
                "pad output dimension {dim} mismatch: expected {expected}, got {}",
                output_shape[dim]
            )));
        }
    }

    Ok(())
}

fn u32_shape(shape: &[usize], name: &str) -> io::Result<Vec<u32>> {
    shape
        .iter()
        .enumerate()
        .map(|(index, &dim)| u32_arg(dim, &format!("{name} dimension {index}")))
        .collect()
}

fn u32_indices(indices: &[i64], name: &str) -> io::Result<Vec<u32>> {
    indices
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            u32::try_from(value)
                .map_err(|_| invalid_input(format!("{name} {index} does not fit in u32: {value}")))
        })
        .collect()
}

fn cpp_u32_array(values: &[u32]) -> String {
    if values.is_empty() {
        return "{1u}".to_owned();
    }
    let values = values
        .iter()
        .map(|value| format!("{value}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{values}}}")
}

fn element_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: usize, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}

fn u32_addr(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}
