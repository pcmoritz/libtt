use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const BROADCAST_READER: &str = include_str!("../../kernels/broadcast_reader.cc");
const BROADCAST_GENERIC_READER: &str = include_str!("../../kernels/broadcast_generic_reader.cc");
const BROADCAST_COMPUTE: &str = include_str!("../../kernels/broadcast_compute.cc");
const BROADCAST_WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum BroadcastKernelShape {
    Fast(FastBroadcastKernelShape),
    Generic(GenericBroadcastKernelShape),
}

impl BroadcastKernelShape {
    fn tile_count(&self) -> u32 {
        match self {
            Self::Fast(shape) => shape.tile_count,
            Self::Generic(shape) => shape.tile_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct FastBroadcastKernelShape {
    output_tiles_per_row: u32,
    tile_count: u32,
    mode: BroadcastMode,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct GenericBroadcastKernelShape {
    input_shape: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_shape: Vec<u32>,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    tile_count: u32,
    broadcast_dimensions: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum BroadcastMode {
    Copy,
    Scalar,
    Row,
    Col,
    Transpose,
}

impl BroadcastMode {
    fn cpp_variant(self) -> &'static str {
        match self {
            Self::Copy => "Copy",
            Self::Scalar => "Scalar",
            Self::Row => "Row",
            Self::Col => "Col",
            Self::Transpose => "Transpose",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BroadcastInDimPlan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_allocation_shape: Vec<usize>,
    kernel_shape: BroadcastKernelShape,
}

impl BroadcastInDimPlan {
    pub(crate) fn new(
        input_shape: &[usize],
        output_shape: &[usize],
        broadcast_dimensions: &[i64],
    ) -> io::Result<Self> {
        validate_broadcast_dimensions(input_shape, output_shape, broadcast_dimensions)?;

        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let kernel_shape = broadcast_shape(input_shape, output_shape, broadcast_dimensions)?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            kernel_shape,
        })
    }

    fn kernel_shape(&self) -> BroadcastKernelShape {
        self.kernel_shape.clone()
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct BroadcastProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: BroadcastKernelShape,
}

struct BroadcastKernel {
    input_addr: u32,
    output_addr: u32,
    key: BroadcastProgramKey,
}

impl Kernel<BroadcastProgramKey> for BroadcastKernel {
    fn program_key(&self) -> BroadcastProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        broadcast_program(self.key.clone())
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

pub(crate) fn broadcast_in_dim(
    device: &mut Device,
    input: &DramBuffer,
    plan: &BroadcastInDimPlan,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "broadcast input requires {:?}, got {:?}",
            dtype, input.dtype
        )));
    }
    let expected_input_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_input_shape {
        return Err(invalid_input(format!(
            "broadcast input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_input_shape, plan.input_shape
        )));
    }

    let input_tile_count = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != input_tile_count {
        return Err(invalid_input(format!(
            "broadcast input tile count mismatch: got {}, expected {input_tile_count}",
            input.num_tiles
        )));
    }

    let shape = plan.kernel_shape();
    let output_tiles = usize::try_from(shape.tile_count()).map_err(|_| {
        invalid_input(format!(
            "tile count does not fit in usize: {}",
            shape.tile_count()
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(output_tiles, dtype, &plan.output_allocation_shape, name)?;
    let kernel = BroadcastKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: BroadcastProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn broadcast_program(key: BroadcastProgramKey) -> io::Result<Program> {
    let BroadcastProgramKey {
        cores,
        dtype,
        shape,
    } = key;
    match shape {
        BroadcastKernelShape::Fast(shape) => fast_broadcast_program(cores, dtype, shape),
        BroadcastKernelShape::Generic(shape) => generic_broadcast_program(cores, dtype, shape),
    }
}

fn fast_broadcast_program(
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: FastBroadcastKernelShape,
) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(shape.tile_count, core_index, cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, offset, n_tiles, shape.output_tiles_per_row],
            vec![n_tiles],
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: broadcast_reader_source(shape.mode),
        compute_kernel: broadcast_compute_source(shape.mode),
        writer_kernel: BROADCAST_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, dtype), CBConfig::new(16, dtype)],
            ..CompileConfig::default()
        },
        name: format!("broadcast_in_dim_{:?}_{:?}", dtype, shape.mode),
        ..Program::new(runtime_args)
    })
}

fn generic_broadcast_program(
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: GenericBroadcastKernelShape,
) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(shape.tile_count, core_index, cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, offset, n_tiles],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: broadcast_generic_reader_source(dtype, &shape)?,
        writer_kernel: BROADCAST_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, dtype), CBConfig::new(16, dtype)],
            ..CompileConfig::default()
        },
        name: format!(
            "broadcast_in_dim_{:?}_generic_{}_{}",
            dtype,
            shape.input_shape.len(),
            shape.output_shape.len()
        ),
        ..Program::new(runtime_args)
    })
}

fn broadcast_reader_source(mode: BroadcastMode) -> String {
    BROADCAST_READER.replace("BROADCAST_MODE", mode.cpp_variant())
}

fn broadcast_compute_source(mode: BroadcastMode) -> String {
    BROADCAST_COMPUTE.replace("BROADCAST_MODE", mode.cpp_variant())
}

fn broadcast_generic_reader_source(
    dtype: DType,
    shape: &GenericBroadcastKernelShape,
) -> io::Result<String> {
    let element_type = element_type(dtype);
    Ok(format!(
        concat!(
            "#define BROADCAST_ELEMENT_TYPE {}\n",
            "#define BROADCAST_INPUT_RANK {}\n",
            "#define BROADCAST_OUTPUT_RANK {}\n",
            "#define BROADCAST_INPUT_SHAPE {}\n",
            "#define BROADCAST_OUTPUT_SHAPE {}\n",
            "#define BROADCAST_DIMENSIONS {}\n",
            "#define BROADCAST_INPUT_TILE_ROWS {}\n",
            "#define BROADCAST_INPUT_TILES_PER_ROW {}\n",
            "#define BROADCAST_OUTPUT_TILE_ROWS {}\n",
            "#define BROADCAST_OUTPUT_TILES_PER_ROW {}\n",
            "{}"
        ),
        element_type,
        shape.input_shape.len(),
        shape.output_shape.len(),
        cpp_u32_array(&shape.input_shape),
        cpp_u32_array(&shape.output_shape),
        cpp_u32_array(&shape.broadcast_dimensions),
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
        BROADCAST_GENERIC_READER,
    ))
}

fn element_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    }
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

fn broadcast_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> io::Result<BroadcastKernelShape> {
    if let Some(shape) = fast_broadcast_shape(input_shape, output_shape, broadcast_dimensions)? {
        Ok(BroadcastKernelShape::Fast(shape))
    } else {
        Ok(BroadcastKernelShape::Generic(generic_broadcast_shape(
            input_shape,
            output_shape,
            broadcast_dimensions,
        )?))
    }
}

fn fast_broadcast_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> io::Result<Option<FastBroadcastKernelShape>> {
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles_per_row = output_allocation_shape[output_allocation_shape.len() - 1] / TILE_C;
    let tile_count = tiled_shape_tile_count(output_shape)?;

    let mode = if input_shape.len() <= 2 && output_shape.len() <= 2 {
        rank2_or_smaller_broadcast_mode(input_shape, output_shape, broadcast_dimensions)?
    } else {
        None
    };

    let Some(mode) = mode else {
        return Ok(None);
    };
    Ok(Some(FastBroadcastKernelShape {
        output_tiles_per_row: u32_arg(output_tiles_per_row, "output tiles per row")?,
        tile_count: u32_arg(tile_count, "tile count")?,
        mode,
    }))
}

fn generic_broadcast_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> io::Result<GenericBroadcastKernelShape> {
    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let input_rank = input_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    let tile_count = tiled_shape_tile_count(output_shape)?;

    Ok(GenericBroadcastKernelShape {
        input_shape: u32_shape(input_shape, "input shape")?,
        input_tile_rows: u32_arg(
            input_allocation_shape[input_rank - 2] / TILE_R,
            "input tile rows",
        )?,
        input_tiles_per_row: u32_arg(
            input_allocation_shape[input_rank - 1] / TILE_C,
            "input tiles per row",
        )?,
        output_shape: u32_shape(output_shape, "output shape")?,
        output_tile_rows: u32_arg(
            output_allocation_shape[output_rank - 2] / TILE_R,
            "output tile rows",
        )?,
        output_tiles_per_row: u32_arg(
            output_allocation_shape[output_rank - 1] / TILE_C,
            "output tiles per row",
        )?,
        tile_count: u32_arg(tile_count, "tile count")?,
        broadcast_dimensions: broadcast_dimensions
            .iter()
            .map(|&dim| {
                u32::try_from(dim).map_err(|_| {
                    invalid_input(format!("broadcast dimension does not fit in u32: {dim}"))
                })
            })
            .collect::<io::Result<Vec<_>>>()?,
    })
}

fn validate_broadcast_dimensions(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> io::Result<()> {
    if broadcast_dimensions.len() != input_shape.len() {
        return Err(invalid_input(format!(
            "broadcast dimensions length {} must match input rank {}",
            broadcast_dimensions.len(),
            input_shape.len()
        )));
    }

    let mut previous = None;
    for (input_dim, &output_dim) in broadcast_dimensions.iter().enumerate() {
        let output_dim = usize::try_from(output_dim).map_err(|_| {
            invalid_input(format!(
                "broadcast dimension must be non-negative, got {output_dim}"
            ))
        })?;
        if output_dim >= output_shape.len() {
            return Err(invalid_input(format!(
                "broadcast dimension {output_dim} is out of bounds for output rank {}",
                output_shape.len()
            )));
        }
        if previous.is_some_and(|previous| output_dim <= previous) {
            return Err(invalid_input(
                "broadcast dimensions must be strictly increasing",
            ));
        }
        previous = Some(output_dim);

        let input_size = input_shape[input_dim];
        let output_size = output_shape[output_dim];
        if input_size != output_size && input_size != 1 {
            return Err(invalid_input(format!(
                "broadcast dimension {input_dim} size {input_size} is incompatible with output dimension {output_dim} size {output_size}"
            )));
        }
    }
    Ok(())
}

fn logical_matrix_view(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (1, 1),
        [cols] => (1, *cols),
        [rows, cols] => (*rows, *cols),
        _ => unreachable!("broadcast rank validation should reject rank > 2"),
    }
}

fn rank2_or_smaller_broadcast_mode(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> io::Result<Option<BroadcastMode>> {
    let input_rank = input_shape.len();
    let output_rank = output_shape.len();
    let (input_rows, input_cols) = logical_matrix_view(input_shape);
    let (output_rows, output_cols) = logical_matrix_view(output_shape);

    let mode = match input_rank {
        0 => {
            if output_rank == 0 {
                Some(BroadcastMode::Copy)
            } else {
                Some(BroadcastMode::Scalar)
            }
        }
        1 => rank1_broadcast_mode(
            input_cols,
            output_rank,
            output_rows,
            output_cols,
            broadcast_dimensions,
        )?,
        2 => rank2_broadcast_mode(input_rows, input_cols, output_rows, output_cols),
        _ => unreachable!("broadcast rank validation should reject rank > 2"),
    };
    Ok(mode)
}

fn rank1_broadcast_mode(
    input_cols: usize,
    output_rank: usize,
    output_rows: usize,
    output_cols: usize,
    broadcast_dimensions: &[i64],
) -> io::Result<Option<BroadcastMode>> {
    if input_cols == 1 {
        return Ok(Some(BroadcastMode::Scalar));
    }
    match output_rank {
        1 => Ok(Some(BroadcastMode::Copy)),
        2 => match broadcast_dimensions[0] {
            0 if output_cols == 1 => Ok(Some(BroadcastMode::Transpose)),
            0 => Ok(None),
            1 if output_rows == 1 => Ok(Some(BroadcastMode::Copy)),
            1 => Ok(Some(BroadcastMode::Row)),
            _ => Ok(None),
        },
        _ => unreachable!("broadcast rank validation should reject rank > 2"),
    }
}

fn rank2_broadcast_mode(
    input_rows: usize,
    input_cols: usize,
    output_rows: usize,
    output_cols: usize,
) -> Option<BroadcastMode> {
    if input_rows == output_rows && input_cols == output_cols {
        return Some(BroadcastMode::Copy);
    }
    if input_rows == 1 && input_cols == 1 {
        return Some(BroadcastMode::Scalar);
    }
    if input_rows == 1 && input_cols == output_cols {
        return Some(BroadcastMode::Row);
    }
    if input_cols == 1 && input_rows == output_rows {
        return Some(BroadcastMode::Col);
    }
    None
}

fn u32_shape(shape: &[usize], name: &str) -> io::Result<Vec<u32>> {
    shape
        .iter()
        .map(|&dim| u32_arg(dim, name))
        .collect::<io::Result<Vec<_>>>()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn arg_u32(blob: &[u8], index: usize) -> u32 {
        let start = index * std::mem::size_of::<u32>();
        u32::from_le_bytes(
            blob[start..start + std::mem::size_of::<u32>()]
                .try_into()
                .unwrap(),
        )
    }

    fn fast_shape(plan: &BroadcastInDimPlan) -> FastBroadcastKernelShape {
        match plan.kernel_shape() {
            BroadcastKernelShape::Fast(shape) => shape,
            BroadcastKernelShape::Generic(shape) => {
                panic!("expected fast broadcast shape, got {shape:?}")
            }
        }
    }

    fn generic_shape(plan: &BroadcastInDimPlan) -> GenericBroadcastKernelShape {
        match plan.kernel_shape() {
            BroadcastKernelShape::Generic(shape) => shape,
            BroadcastKernelShape::Fast(shape) => {
                panic!("expected generic broadcast shape, got {shape:?}")
            }
        }
    }

    #[test]
    fn broadcast_plan_normalizes_rank1_column_case() {
        let plan = BroadcastInDimPlan::new(&[32], &[32, 1], &[0]).expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(
            fast_shape(&plan),
            FastBroadcastKernelShape {
                output_tiles_per_row: 1,
                tile_count: 1,
                mode: BroadcastMode::Transpose,
            }
        );
    }

    #[test]
    fn broadcast_plan_allows_degenerate_matrix_dimensions() {
        let plan = BroadcastInDimPlan::new(&[1, 4], &[8, 4], &[0, 1]).expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(fast_shape(&plan).mode, BroadcastMode::Row);
    }

    #[test]
    fn broadcast_plan_rejects_incompatible_mapped_dimensions() {
        let err = BroadcastInDimPlan::new(&[4], &[8, 1], &[0])
            .expect_err("incompatible broadcast should fail");

        assert!(err.to_string().contains("incompatible"));
    }

    #[test]
    fn broadcast_plan_uses_generic_for_rank1_transpose_plus_broadcast() {
        let plan = BroadcastInDimPlan::new(&[4], &[4, 8], &[0])
            .expect("generic broadcast should support transpose plus broadcast");

        let shape = generic_shape(&plan);
        assert_eq!(shape.input_shape, vec![4]);
        assert_eq!(shape.output_shape, vec![4, 8]);
        assert_eq!(shape.output_tile_rows, 1);
        assert_eq!(shape.output_tiles_per_row, 1);
        assert_eq!(shape.tile_count, 1);
        assert_eq!(shape.broadcast_dimensions, vec![0]);
    }

    #[test]
    fn broadcast_plan_uses_generic_for_inserted_middle_dimension() {
        let plan =
            BroadcastInDimPlan::new(&[18, 32], &[18, 1, 32], &[0, 2]).expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![18, 32, 32]);
        let shape = generic_shape(&plan);
        assert_eq!(shape.input_shape, vec![18, 32]);
        assert_eq!(shape.input_tile_rows, 1);
        assert_eq!(shape.input_tiles_per_row, 1);
        assert_eq!(shape.output_shape, vec![18, 1, 32]);
        assert_eq!(shape.output_tile_rows, 1);
        assert_eq!(shape.output_tiles_per_row, 1);
        assert_eq!(shape.tile_count, 18);
        assert_eq!(shape.broadcast_dimensions, vec![0, 2]);
    }

    #[test]
    fn broadcast_plan_uses_generic_for_rank4_attention_shape() {
        let plan = BroadcastInDimPlan::new(&[18, 2, 32], &[18, 2, 2, 32], &[0, 1, 3])
            .expect("valid broadcast");

        let shape = generic_shape(&plan);
        assert_eq!(shape.input_shape, vec![18, 2, 32]);
        assert_eq!(shape.output_shape, vec![18, 2, 2, 32]);
        assert_eq!(shape.tile_count, 36);
        assert_eq!(shape.broadcast_dimensions, vec![0, 1, 3]);
    }

    #[test]
    fn broadcast_plan_uses_generic_for_scalar_to_rank3() {
        let plan = BroadcastInDimPlan::new(&[], &[2, 3, 4], &[]).expect("valid broadcast");

        let shape = generic_shape(&plan);
        assert_eq!(shape.input_shape, Vec::<u32>::new());
        assert_eq!(shape.output_shape, vec![2, 3, 4]);
        assert_eq!(shape.input_tile_rows, 1);
        assert_eq!(shape.input_tiles_per_row, 1);
        assert_eq!(shape.output_tile_rows, 1);
        assert_eq!(shape.output_tiles_per_row, 1);
        assert_eq!(shape.tile_count, 2);
        assert_eq!(shape.broadcast_dimensions, Vec::<u32>::new());
    }

    #[test]
    fn fast_broadcast_program_splits_tiles_across_cores() {
        let program = broadcast_program(BroadcastProgramKey {
            cores: vec![
                CoreCoord { x: 1, y: 2 },
                CoreCoord { x: 1, y: 3 },
                CoreCoord { x: 1, y: 4 },
            ],
            dtype: DType::Float16B,
            shape: BroadcastKernelShape::Fast(FastBroadcastKernelShape {
                output_tiles_per_row: 2,
                tile_count: 5,
                mode: BroadcastMode::Copy,
            }),
        })
        .expect("broadcast program");

        assert_eq!(program.runtime_args.cores().len(), 3);
        let blobs = program.runtime_args.blobs();
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 2));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (2, 2));
        assert_eq!((arg_u32(&blobs[2], 1), arg_u32(&blobs[2], 2)), (4, 1));
        assert_eq!((arg_u32(&blobs[0], 4), arg_u32(&blobs[0], 5)), (0, 2));
        assert_eq!((arg_u32(&blobs[1], 4), arg_u32(&blobs[1], 5)), (2, 2));
        assert_eq!((arg_u32(&blobs[2], 4), arg_u32(&blobs[2], 5)), (4, 1));
        assert_eq!(arg_u32(&blobs[0], 7), 2);
        assert_eq!(arg_u32(&blobs[1], 7), 2);
        assert_eq!(arg_u32(&blobs[2], 7), 1);
    }

    #[test]
    fn generic_broadcast_program_splits_tiles_across_cores() {
        let shape =
            generic_broadcast_shape(&[18, 32], &[18, 1, 32], &[0, 2]).expect("generic shape");
        let program = broadcast_program(BroadcastProgramKey {
            cores: vec![
                CoreCoord { x: 1, y: 2 },
                CoreCoord { x: 1, y: 3 },
                CoreCoord { x: 1, y: 4 },
            ],
            dtype: DType::Float16B,
            shape: BroadcastKernelShape::Generic(shape),
        })
        .expect("broadcast program");

        assert_eq!(program.compute_kernel, "");
        assert!(program.reader_kernel.contains("BROADCAST_INPUT_RANK 2"));
        assert!(program.reader_kernel.contains("BROADCAST_OUTPUT_RANK 3"));
        assert!(program
            .reader_kernel
            .contains("BROADCAST_DIMENSIONS {0u, 2u}"));

        assert_eq!(program.runtime_args.cores().len(), 3);
        let blobs = program.runtime_args.blobs();
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 6));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (6, 6));
        assert_eq!((arg_u32(&blobs[2], 1), arg_u32(&blobs[2], 2)), (12, 6));
        assert_eq!((arg_u32(&blobs[0], 4), arg_u32(&blobs[0], 5)), (0, 6));
        assert_eq!((arg_u32(&blobs[1], 4), arg_u32(&blobs[1], 5)), (6, 6));
        assert_eq!((arg_u32(&blobs[2], 4), arg_u32(&blobs[2], 5)), (12, 6));
    }
}
