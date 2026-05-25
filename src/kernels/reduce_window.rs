use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::executable::{ReduceReducer, ReduceWindowAttributes};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/reduce_window_reader.cc");
const WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReduceWindowShape {
    input_shape: Vec<u32>,
    output_shape: Vec<u32>,
    window_dimensions: Vec<u32>,
    window_strides: Vec<u32>,
    base_dilations: Vec<u32>,
    window_dilations: Vec<u32>,
    padding_low: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    output_tiles: u32,
    window_elements: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReduceWindowPlan {
    input_shape: Vec<usize>,
    output_allocation_shape: Vec<usize>,
    shape: ReduceWindowShape,
    dtype: DType,
}

impl ReduceWindowPlan {
    pub(crate) fn new(
        dtype: DType,
        input_shape: &[usize],
        output_shape: &[usize],
        attributes: &ReduceWindowAttributes,
        reducer: ReduceReducer,
    ) -> io::Result<Self> {
        validate_reduce_window(dtype, input_shape, output_shape, attributes, reducer)?;

        let input_allocation_shape = tiled_allocation_shape(input_shape)?;
        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let input_rank = input_allocation_shape.len();
        let output_rank = output_allocation_shape.len();
        let window_elements = attributes
            .window_dimensions
            .iter()
            .try_fold(1usize, |acc, &dim| {
                let dim = usize::try_from(dim).map_err(|_| {
                    invalid_input(format!("reduce_window window dimension must be positive: {dim}"))
                })?;
                acc.checked_mul(dim)
                    .ok_or_else(|| invalid_input("reduce_window window element count overflow"))
            })?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape: output_allocation_shape.clone(),
            shape: ReduceWindowShape {
                input_shape: u32_shape(input_shape, "reduce_window input shape")?,
                output_shape: u32_shape(output_shape, "reduce_window output shape")?,
                window_dimensions: u32_positive_indices(
                    &attributes.window_dimensions,
                    "reduce_window window_dimensions",
                )?,
                window_strides: u32_positive_indices(
                    &attributes.window_strides,
                    "reduce_window window_strides",
                )?,
                base_dilations: u32_positive_indices(
                    &attributes.base_dilations,
                    "reduce_window base_dilations",
                )?,
                window_dilations: u32_positive_indices(
                    &attributes.window_dilations,
                    "reduce_window window_dilations",
                )?,
                padding_low: u32_non_negative_indices(
                    &attributes.padding_low,
                    "reduce_window padding_low",
                )?,
                input_tile_rows: u32_arg(
                    input_allocation_shape[input_rank - 2] / TILE_R,
                    "reduce_window input tile rows",
                )?,
                input_tiles_per_row: u32_arg(
                    input_allocation_shape[input_rank - 1] / TILE_C,
                    "reduce_window input tiles per row",
                )?,
                output_tile_rows: u32_arg(
                    output_allocation_shape[output_rank - 2] / TILE_R,
                    "reduce_window output tile rows",
                )?,
                output_tiles_per_row: u32_arg(
                    output_allocation_shape[output_rank - 1] / TILE_C,
                    "reduce_window output tiles per row",
                )?,
                output_tiles: u32_arg(
                    tiled_shape_tile_count(output_shape)?,
                    "reduce_window output tile count",
                )?,
                window_elements: u32_arg(window_elements, "reduce_window window element count")?,
            },
            dtype,
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReduceWindowProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: ReduceWindowShape,
}

struct ReduceWindowKernel {
    input_addr: u32,
    output_addr: u32,
    key: ReduceWindowProgramKey,
}

impl Kernel<ReduceWindowProgramKey> for ReduceWindowKernel {
    fn program_key(&self) -> ReduceWindowProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        reduce_window_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_INPUT_ADDR_INDEX => Some(self.input_addr),
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

pub(crate) fn reduce_window(
    device: &mut Device,
    input: &DramBuffer,
    plan: &ReduceWindowPlan,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_input(input, plan)?;
    let output_tiles = usize::try_from(plan.shape.output_tiles).map_err(|_| {
        invalid_input(format!(
            "reduce_window output tile count does not fit in usize: {}",
            plan.shape.output_tiles
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(
        output_tiles,
        plan.dtype,
        &plan.output_allocation_shape,
        name,
    )?;
    let kernel = ReduceWindowKernel {
        input_addr: u32_addr(input.addr, "reduce_window input address")?,
        output_addr: u32_addr(output.addr, "reduce_window output address")?,
        key: ReduceWindowProgramKey {
            cores,
            dtype: plan.dtype,
            shape: plan.shape.clone(),
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_input(input: &DramBuffer, plan: &ReduceWindowPlan) -> io::Result<()> {
    if input.dtype != plan.dtype {
        return Err(invalid_input(format!(
            "reduce_window input requires {:?}, got {:?}",
            plan.dtype, input.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "reduce_window input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, plan.input_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "reduce_window input tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn validate_reduce_window(
    dtype: DType,
    input_shape: &[usize],
    output_shape: &[usize],
    attributes: &ReduceWindowAttributes,
    reducer: ReduceReducer,
) -> io::Result<()> {
    if reducer != ReduceReducer::Add {
        return Err(invalid_input(format!(
            "reduce_window currently supports only add reducers, got {reducer:?}"
        )));
    }
    if !matches!(
        dtype,
        DType::Float32 | DType::Int32 | DType::UInt32 | DType::UInt16 | DType::UInt8
    ) {
        return Err(invalid_input(format!(
            "reduce_window add does not support dtype {dtype:?}"
        )));
    }
    let rank = input_shape.len();
    if rank == 0 {
        return Err(invalid_input(
            "reduce_window currently requires rank >= 1 input",
        ));
    }
    if output_shape.len() != rank {
        return Err(invalid_input(format!(
            "reduce_window output rank {} must match input rank {rank}",
            output_shape.len()
        )));
    }
    validate_attr_len(&attributes.window_dimensions, rank, "window_dimensions")?;
    validate_attr_len(&attributes.window_strides, rank, "window_strides")?;
    validate_attr_len(&attributes.base_dilations, rank, "base_dilations")?;
    validate_attr_len(&attributes.window_dilations, rank, "window_dilations")?;
    validate_attr_len(&attributes.padding_low, rank, "padding_low")?;
    validate_attr_len(&attributes.padding_high, rank, "padding_high")?;

    for dim in 0..rank {
        let window_dimension =
            positive_usize(attributes.window_dimensions[dim], "window_dimensions", dim)?;
        let window_stride = positive_usize(attributes.window_strides[dim], "window_strides", dim)?;
        let base_dilation = positive_usize(attributes.base_dilations[dim], "base_dilations", dim)?;
        let window_dilation =
            positive_usize(attributes.window_dilations[dim], "window_dilations", dim)?;
        let padding_low = non_negative_usize(attributes.padding_low[dim], "padding_low", dim)?;
        let padding_high = non_negative_usize(attributes.padding_high[dim], "padding_high", dim)?;

        let dilated_input = input_shape[dim]
            .saturating_sub(1)
            .checked_mul(base_dilation)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_input(format!("reduce_window dimension {dim} overflow")))?;
        let dilated_window = window_dimension
            .saturating_sub(1)
            .checked_mul(window_dilation)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_input(format!("reduce_window dimension {dim} overflow")))?;
        let padded_input = dilated_input
            .checked_add(padding_low)
            .and_then(|value| value.checked_add(padding_high))
            .ok_or_else(|| invalid_input(format!("reduce_window dimension {dim} overflow")))?;
        if padded_input < dilated_window {
            return Err(invalid_input(format!(
                "reduce_window dimension {dim} has no valid output windows"
            )));
        }
        let expected = (padded_input - dilated_window) / window_stride + 1;
        if output_shape[dim] != expected {
            return Err(invalid_input(format!(
                "reduce_window output dimension {dim} mismatch: expected {expected}, got {}",
                output_shape[dim]
            )));
        }
    }

    Ok(())
}

fn reduce_window_program(key: ReduceWindowProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.output_tiles, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, offset, n_tiles],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: reduce_window_reader_source(&key)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!(
            "reduce_window_add_{:?}_{}",
            key.dtype,
            key.shape.input_shape.len()
        ),
        ..Program::new(runtime_args)
    })
}

fn reduce_window_reader_source(key: &ReduceWindowProgramKey) -> io::Result<String> {
    Ok(format!(
        "#define REDUCE_WINDOW_RANK {}\n\
         #define REDUCE_WINDOW_INPUT_SHAPE {}\n\
         #define REDUCE_WINDOW_OUTPUT_SHAPE {}\n\
         #define REDUCE_WINDOW_WINDOW_DIMENSIONS {}\n\
         #define REDUCE_WINDOW_WINDOW_STRIDES {}\n\
         #define REDUCE_WINDOW_BASE_DILATIONS {}\n\
         #define REDUCE_WINDOW_WINDOW_DILATIONS {}\n\
         #define REDUCE_WINDOW_PADDING_LOW {}\n\
         #define REDUCE_WINDOW_INPUT_TILE_ROWS {}\n\
         #define REDUCE_WINDOW_INPUT_TILES_PER_ROW {}\n\
         #define REDUCE_WINDOW_OUTPUT_TILE_ROWS {}\n\
         #define REDUCE_WINDOW_OUTPUT_TILES_PER_ROW {}\n\
         #define REDUCE_WINDOW_WINDOW_ELEMENTS {}\n\
         #define REDUCE_WINDOW_ELEMENT_TYPE {}\n\
         {READER}",
        key.shape.input_shape.len(),
        cpp_u32_array(&key.shape.input_shape),
        cpp_u32_array(&key.shape.output_shape),
        cpp_u32_array(&key.shape.window_dimensions),
        cpp_u32_array(&key.shape.window_strides),
        cpp_u32_array(&key.shape.base_dilations),
        cpp_u32_array(&key.shape.window_dilations),
        cpp_u32_array(&key.shape.padding_low),
        key.shape.input_tile_rows,
        key.shape.input_tiles_per_row,
        key.shape.output_tile_rows,
        key.shape.output_tiles_per_row,
        key.shape.window_elements,
        element_type(key.dtype),
    ))
}

fn validate_attr_len(values: &[i64], rank: usize, name: &str) -> io::Result<()> {
    if values.len() != rank {
        return Err(invalid_input(format!(
            "reduce_window {name} length {} must match rank {rank}",
            values.len()
        )));
    }
    Ok(())
}

fn positive_usize(value: i64, name: &str, index: usize) -> io::Result<usize> {
    if value <= 0 {
        return Err(invalid_input(format!(
            "reduce_window {name}[{index}] must be positive, got {value}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        invalid_input(format!(
            "reduce_window {name}[{index}] does not fit in usize: {value}"
        ))
    })
}

fn non_negative_usize(value: i64, name: &str, index: usize) -> io::Result<usize> {
    if value < 0 {
        return Err(invalid_input(format!(
            "reduce_window {name}[{index}] must be non-negative, got {value}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        invalid_input(format!(
            "reduce_window {name}[{index}] does not fit in usize: {value}"
        ))
    })
}

fn u32_shape(shape: &[usize], name: &str) -> io::Result<Vec<u32>> {
    shape
        .iter()
        .enumerate()
        .map(|(index, &dim)| u32_arg(dim, &format!("{name} dimension {index}")))
        .collect()
}

fn u32_positive_indices(indices: &[i64], name: &str) -> io::Result<Vec<u32>> {
    indices
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value <= 0 {
                return Err(invalid_input(format!(
                    "{name} {index} must be positive, got {value}"
                )));
            }
            u32::try_from(value)
                .map_err(|_| invalid_input(format!("{name} {index} does not fit in u32: {value}")))
        })
        .collect()
}

fn u32_non_negative_indices(indices: &[i64], name: &str) -> io::Result<Vec<u32>> {
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
    let values = values
        .iter()
        .map(|value| format!("{value}u"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{values}}}")
}

fn element_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "float",
        DType::Int32 => "int32_t",
        DType::UInt32 => "uint32_t",
        DType::UInt16 => "uint16_t",
        DType::UInt8 => "uint8_t",
        _ => "uint32_t",
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
