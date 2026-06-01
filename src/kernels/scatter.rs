use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use crate::kernels::reshape_view::{reshape_source_view, ReshapeSourceView};
use std::io;

const SCATTER_READER: &str = include_str!("../../kernels/scatter_reader.cc");
const SCATTER_WRITER: &str = "void kernel_main() {}\n";
const READER_OPERAND_ADDR_INDEX: usize = 0;
const READER_START_INDICES_ADDR_INDEX: usize = 1;
const READER_UPDATES_ADDR_INDEX: usize = 2;
const READER_OUTPUT_ADDR_INDEX: usize = 3;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct ScatterShape {
    operand_shape: Vec<u32>,
    update_shape: Vec<u32>,
    scatter_dim: u32,
    update_count: u32,
    operand_source_view: Option<ReshapeSourceView>,
    update_source_view: Option<ReshapeSourceView>,
    operand_tile_rows: u32,
    operand_tiles_per_row: u32,
    update_tile_rows: u32,
    update_tiles_per_row: u32,
    output_tiles: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ScatterProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: ScatterShape,
    in_place: bool,
}

struct ScatterKernel {
    operand_addr: u32,
    start_indices_addr: u32,
    updates_addr: u32,
    output_addr: u32,
    key: ScatterProgramKey,
}

impl Kernel<ScatterProgramKey> for ScatterKernel {
    fn program_key(&self) -> ScatterProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        scatter_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_OPERAND_ADDR_INDEX => Some(self.operand_addr),
            READER_START_INDICES_ADDR_INDEX => Some(self.start_indices_addr),
            READER_UPDATES_ADDR_INDEX => Some(self.updates_addr),
            READER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            _ => None,
        }
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        let _ = index;
        None
    }
}

pub(crate) fn scatter_set(
    device: &mut Device,
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    updates: &DramBuffer,
    operand_shape: &[usize],
    operand_source_shape: Option<&[usize]>,
    start_indices_shape: &[usize],
    update_shape: &[usize],
    update_source_shape: Option<&[usize]>,
    scatter_dim: usize,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_scatter_buffers(
        operand,
        start_indices,
        updates,
        operand_shape,
        operand_source_shape,
        start_indices_shape,
        update_shape,
        update_source_shape,
        scatter_dim,
        dtype,
    )?;

    let shape = scatter_shape(
        operand_shape,
        operand_source_shape,
        update_shape,
        update_source_shape,
        scatter_dim,
    )?;
    let output_tiles = usize::try_from(shape.output_tiles).map_err(|_| {
        invalid_input(format!(
            "scatter output tile count does not fit in usize: {}",
            shape.output_tiles
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(
        output_tiles,
        dtype,
        &tiled_allocation_shape(operand_shape)?,
        name,
    )?;
    let kernel = ScatterKernel {
        operand_addr: u32_addr(operand.addr, "scatter operand address")?,
        start_indices_addr: u32_addr(start_indices.addr, "scatter start_indices address")?,
        updates_addr: u32_addr(updates.addr, "scatter updates address")?,
        output_addr: u32_addr(output.addr, "scatter output address")?,
        key: ScatterProgramKey {
            cores,
            dtype,
            shape,
            in_place: false,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

pub(crate) fn scatter_set_in_place(
    device: &mut Device,
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    updates: &DramBuffer,
    operand_shape: &[usize],
    operand_source_shape: Option<&[usize]>,
    start_indices_shape: &[usize],
    update_shape: &[usize],
    update_source_shape: Option<&[usize]>,
    scatter_dim: usize,
    dtype: DType,
) -> io::Result<DramBuffer> {
    validate_scatter_buffers(
        operand,
        start_indices,
        updates,
        operand_shape,
        operand_source_shape,
        start_indices_shape,
        update_shape,
        update_source_shape,
        scatter_dim,
        dtype,
    )?;

    let shape = scatter_shape(
        operand_shape,
        operand_source_shape,
        update_shape,
        update_source_shape,
        scatter_dim,
    )?;
    if !can_scatter_in_place(&shape) {
        return Err(invalid_input(format!(
            "scatter in-place path does not support shape {:?} update {:?} dim {}",
            operand_shape, update_shape, scatter_dim
        )));
    }
    let updated_tiles = tiled_shape_tile_count(update_shape)?;
    let cores = select_worker_cores(device.cores_ref(), updated_tiles)?;
    let kernel = ScatterKernel {
        operand_addr: u32_addr(operand.addr, "scatter operand address")?,
        start_indices_addr: u32_addr(start_indices.addr, "scatter start_indices address")?,
        updates_addr: u32_addr(updates.addr, "scatter updates address")?,
        output_addr: u32_addr(operand.addr, "scatter output address")?,
        key: ScatterProgramKey {
            cores,
            dtype,
            shape,
            in_place: true,
        },
    };
    kernel.run(device)?;
    Ok(operand.clone())
}

fn can_scatter_in_place(shape: &ScatterShape) -> bool {
    let rank = shape.operand_shape.len();
    rank >= 3
        && shape.scatter_dim + 2 < rank as u32
        && shape.operand_source_view.is_none()
        && shape.update_source_view.is_none()
}

fn validate_scatter_buffers(
    operand: &DramBuffer,
    start_indices: &DramBuffer,
    updates: &DramBuffer,
    operand_shape: &[usize],
    operand_source_shape: Option<&[usize]>,
    start_indices_shape: &[usize],
    update_shape: &[usize],
    update_source_shape: Option<&[usize]>,
    scatter_dim: usize,
    dtype: DType,
) -> io::Result<()> {
    if operand_shape.is_empty() {
        return Err(invalid_input("scatter set requires rank >= 1"));
    }
    if scatter_dim >= operand_shape.len() {
        return Err(invalid_input(format!(
            "scatter dim {scatter_dim} is out of bounds for rank {}",
            operand_shape.len()
        )));
    }
    if operand.dtype != dtype {
        return Err(invalid_input(format!(
            "scatter operand requires {:?}, got {:?}",
            dtype, operand.dtype
        )));
    }
    if updates.dtype != dtype {
        return Err(invalid_input(format!(
            "scatter updates requires {:?}, got {:?}",
            dtype, updates.dtype
        )));
    }
    if start_indices.dtype != DType::Int32 {
        return Err(invalid_input(format!(
            "scatter start_indices requires Int32, got {:?}",
            start_indices.dtype
        )));
    }

    if start_indices_shape.len() != 2 || start_indices_shape[1] != 1 {
        return Err(invalid_input(format!(
            "scatter set requires start_indices shaped [N, 1], got {start_indices_shape:?}"
        )));
    }
    let update_count = start_indices_shape[0];
    if update_shape.len() != operand_shape.len() {
        return Err(invalid_input(format!(
            "scatter set update rank must match operand rank, got {update_shape:?} for operand {operand_shape:?}"
        )));
    }
    let mut expected_update_shape = operand_shape.to_vec();
    expected_update_shape[scatter_dim] = update_count;
    if update_shape != expected_update_shape.as_slice() {
        return Err(invalid_input(format!(
            "scatter set update shape mismatch: got {update_shape:?}, expected {expected_update_shape:?}"
        )));
    }

    validate_allocation(
        operand,
        operand_source_shape.unwrap_or(operand_shape),
        "scatter operand",
    )?;
    validate_allocation(start_indices, start_indices_shape, "scatter start_indices")?;
    validate_allocation(
        updates,
        update_source_shape.unwrap_or(update_shape),
        "scatter updates",
    )?;
    Ok(())
}

fn validate_allocation(buffer: &DramBuffer, logical_shape: &[usize], name: &str) -> io::Result<()> {
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    if buffer.shape != expected_shape {
        return Err(invalid_input(format!(
            "{name} allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            buffer.shape, expected_shape, logical_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(logical_shape)?;
    if buffer.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "{name} tile count mismatch: got {}, expected {expected_tiles}",
            buffer.num_tiles
        )));
    }
    Ok(())
}

fn scatter_shape(
    operand_shape: &[usize],
    operand_source_shape: Option<&[usize]>,
    update_shape: &[usize],
    update_source_shape: Option<&[usize]>,
    scatter_dim: usize,
) -> io::Result<ScatterShape> {
    let operand_allocation_shape = tiled_allocation_shape(operand_shape)?;
    let update_allocation_shape =
        tiled_allocation_shape(update_source_shape.unwrap_or(update_shape))?;
    let operand_rank = operand_allocation_shape.len();
    let update_rank = update_allocation_shape.len();
    let output_tiles = tiled_shape_tile_count(operand_shape)?;
    let operand_source_view =
        optional_reshape_source_view(operand_source_shape, operand_shape, "scatter reshape view")?;
    let update_source_view = optional_reshape_source_view(
        update_source_shape,
        update_shape,
        "scatter update reshape view",
    )?;
    Ok(ScatterShape {
        operand_shape: u32_shape(operand_shape, "scatter operand shape")?,
        update_shape: u32_shape(update_shape, "scatter update shape")?,
        scatter_dim: u32_arg(scatter_dim, "scatter dim")?,
        update_count: u32_arg(update_shape[scatter_dim], "scatter update count")?,
        operand_source_view,
        update_source_view,
        operand_tile_rows: u32_arg(
            operand_allocation_shape[operand_rank - 2] / TILE_R,
            "scatter operand tile rows",
        )?,
        operand_tiles_per_row: u32_arg(
            operand_allocation_shape[operand_rank - 1] / TILE_C,
            "scatter operand tiles per row",
        )?,
        update_tile_rows: u32_arg(
            update_allocation_shape[update_rank - 2] / TILE_R,
            "scatter update tile rows",
        )?,
        update_tiles_per_row: u32_arg(
            update_allocation_shape[update_rank - 1] / TILE_C,
            "scatter update tiles per row",
        )?,
        output_tiles: u32_arg(output_tiles, "scatter output tile count")?,
    })
}

fn optional_reshape_source_view(
    source_shape: Option<&[usize]>,
    logical_shape: &[usize],
    name: &str,
) -> io::Result<Option<ReshapeSourceView>> {
    source_shape
        .map(|source_shape| reshape_source_view(source_shape, logical_shape, name))
        .transpose()
}

fn scatter_program(key: ScatterProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        Vec::new(),
        vec![
            READER_OPERAND_ADDR_INDEX,
            READER_START_INDICES_ADDR_INDEX,
            READER_UPDATES_ADDR_INDEX,
            READER_OUTPUT_ADDR_INDEX,
        ],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let work_tiles = if key.in_place {
            let update_shape = key
                .shape
                .update_shape
                .iter()
                .map(|&dim| dim as usize)
                .collect::<Vec<_>>();
            u32_arg(
                tiled_shape_tile_count(&update_shape)?,
                "scatter updated tile count",
            )?
        } else {
            key.shape.output_tiles
        };
        let (offset, n_tiles) = split_tile_range(work_tiles, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            Vec::new(),
            vec![0, 0, 0, 0, offset, n_tiles],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: scatter_reader_source(key.dtype, &key.shape, key.in_place)?,
        writer_kernel: SCATTER_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.dtype),
                CBConfig::new(1, DType::Int32),
                CBConfig::new(2, key.dtype),
                CBConfig::new(16, key.dtype),
            ],
            ..CompileConfig::default()
        },
        name: format!(
            "scatter_set_{:?}_{}",
            key.dtype,
            key.shape.operand_shape.len()
        ),
        ..Program::new(runtime_args)
    })
}

fn scatter_reader_source(dtype: DType, shape: &ScatterShape, in_place: bool) -> io::Result<String> {
    let element_type = element_type(dtype);
    let operand_source_view = shape.operand_source_view.as_ref();
    let update_source_view = shape.update_source_view.as_ref();
    Ok(format!(
        "#define SCATTER_RANK {}\n\
         #define SCATTER_OPERAND_SHAPE {}\n\
         #define SCATTER_UPDATE_SHAPE {}\n\
         #define SCATTER_DIM_ARG {}\n\
         #define SCATTER_UPDATE_COUNT {}\n\
         #define SCATTER_OPERAND_RESHAPE_VIEW {}\n\
         #define SCATTER_SOURCE_ROWS {}\n\
         #define SCATTER_SOURCE_COLS {}\n\
         #define SCATTER_SOURCE_TILE_ROWS {}\n\
         #define SCATTER_SOURCE_TILES_PER_ROW {}\n\
         #define SCATTER_UPDATE_RESHAPE_VIEW {}\n\
         #define SCATTER_UPDATE_SOURCE_ROWS {}\n\
         #define SCATTER_UPDATE_SOURCE_COLS {}\n\
         #define SCATTER_UPDATE_SOURCE_TILE_ROWS {}\n\
         #define SCATTER_UPDATE_SOURCE_TILES_PER_ROW {}\n\
         #define SCATTER_OPERAND_TILE_ROWS {}\n\
         #define SCATTER_OPERAND_TILES_PER_ROW {}\n\
         #define SCATTER_UPDATE_TILE_ROWS {}\n\
         #define SCATTER_UPDATE_TILES_PER_ROW {}\n\
         #define SCATTER_IN_PLACE {}\n\
         #define SCATTER_ELEMENT_TYPE {element_type}\n\
         {SCATTER_READER}",
        shape.operand_shape.len(),
        cpp_u32_array(&shape.operand_shape),
        cpp_u32_array(&shape.update_shape),
        shape.scatter_dim,
        shape.update_count,
        operand_source_view.is_some() as u32,
        operand_source_view.map_or(1, |view| view.rows),
        operand_source_view.map_or(1, |view| view.cols),
        operand_source_view.map_or(1, |view| view.tile_rows),
        operand_source_view.map_or(1, |view| view.tiles_per_row),
        update_source_view.is_some() as u32,
        update_source_view.map_or(1, |view| view.rows),
        update_source_view.map_or(1, |view| view.cols),
        update_source_view.map_or(1, |view| view.tile_rows),
        update_source_view.map_or(1, |view| view.tiles_per_row),
        shape.operand_tile_rows,
        shape.operand_tiles_per_row,
        shape.update_tile_rows,
        shape.update_tiles_per_row,
        in_place as u32,
    ))
}

pub(crate) fn validate_set_dimension_numbers(
    rank: usize,
    update_window_dims: &[i64],
    inserted_window_dims: &[i64],
    input_batching_dims: &[i64],
    scatter_indices_batching_dims: &[i64],
    scatter_dims_to_operand_dims: &[i64],
    index_vector_dim: i64,
) -> io::Result<usize> {
    if scatter_dims_to_operand_dims.len() != 1 {
        return Err(invalid_input(format!(
            "scatter set requires one scatter_dims_to_operand_dims entry, got {scatter_dims_to_operand_dims:?}"
        )));
    }
    let scatter_dim = scatter_dims_to_operand_dims[0];
    if scatter_dim < 0 || scatter_dim >= rank as i64 {
        return Err(invalid_input(format!(
            "scatter dim {scatter_dim} is out of bounds for rank {rank}"
        )));
    }
    let expected_update_window_dims = (0..rank as i64)
        .filter(|dim| *dim != scatter_dim)
        .collect::<Vec<_>>();
    if update_window_dims != expected_update_window_dims.as_slice() {
        return Err(invalid_input(format!(
            "scatter set requires update_window_dims {:?}, got {:?}",
            expected_update_window_dims, update_window_dims
        )));
    }
    if inserted_window_dims != [scatter_dim] {
        return Err(invalid_input(format!(
            "scatter set requires inserted_window_dims [{scatter_dim}], got {inserted_window_dims:?}"
        )));
    }
    if !input_batching_dims.is_empty() || !scatter_indices_batching_dims.is_empty() {
        return Err(invalid_input(
            "scatter set does not support scatter batching dimensions",
        ));
    }
    if index_vector_dim != 1 {
        return Err(invalid_input(format!(
            "scatter set requires index_vector_dim 1, got {index_vector_dim}"
        )));
    }
    Ok(scatter_dim as usize)
}

pub(crate) fn is_full_window_set_dimension_numbers(
    rank: usize,
    update_window_dims: &[i64],
    inserted_window_dims: &[i64],
    input_batching_dims: &[i64],
    scatter_indices_batching_dims: &[i64],
    scatter_dims_to_operand_dims: &[i64],
    index_vector_dim: i64,
) -> bool {
    let expected_update_window_dims = (0..rank as i64).collect::<Vec<_>>();
    update_window_dims == expected_update_window_dims.as_slice()
        && inserted_window_dims.is_empty()
        && input_batching_dims.is_empty()
        && scatter_indices_batching_dims.is_empty()
        && scatter_dims_to_operand_dims.is_empty()
        && index_vector_dim == 0
}

fn u32_shape(shape: &[usize], name: &str) -> io::Result<Vec<u32>> {
    shape
        .iter()
        .enumerate()
        .map(|(index, &dim)| u32_arg(dim, &format!("{name} dimension {index}")))
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
