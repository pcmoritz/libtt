use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::io;

const BROADCAST_READER: &str = include_str!("../../kernels/broadcast_reader.cc");
const BROADCAST_COMPUTE: &str = include_str!("../../kernels/broadcast_compute.cc");
const BROADCAST_WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct BroadcastKernelShape {
    output_tiles_per_row: u32,
    tile_count: u32,
    mode: BroadcastMode,
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
    fn reader_mode_arg(self) -> u32 {
        match self {
            Self::Copy => 0,
            Self::Scalar => 1,
            Self::Row => 2,
            Self::Col | Self::Transpose => 3,
        }
    }

    fn compute_op_arg(self) -> u32 {
        match self {
            Self::Copy => 0,
            Self::Row => 1,
            Self::Col => 2,
            Self::Scalar => 3,
            Self::Transpose => 4,
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
        validate_rank(input_shape, "input")?;
        validate_rank(output_shape, "output")?;
        validate_broadcast_dimensions(input_shape, output_shape, broadcast_dimensions)?;

        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let output_tiles_per_row =
            output_allocation_shape[output_allocation_shape.len() - 1] / TILE_C;
        let tile_count = tiled_shape_tile_count(output_shape)?;
        let mode = broadcast_mode(input_shape, output_shape, broadcast_dimensions)?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            kernel_shape: BroadcastKernelShape {
                output_tiles_per_row: u32_arg(output_tiles_per_row, "output tiles per row")?,
                tile_count: u32_arg(tile_count, "tile count")?,
                mode,
            },
        })
    }

    fn kernel_shape(&self) -> BroadcastKernelShape {
        self.kernel_shape
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct BroadcastProgramKey {
    core: CoreCoord,
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
        self.key
    }

    fn build_program(&self) -> io::Result<Program> {
        broadcast_program(self.key)
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

    let core = device
        .cores_ref()
        .first()
        .copied()
        .ok_or_else(|| invalid_input("no worker cores are available"))?;
    let shape = plan.kernel_shape();
    let output = device.alloc(
        shape.tile_count as usize,
        dtype,
        &plan.output_allocation_shape,
        name,
    )?;
    let kernel = BroadcastKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: BroadcastProgramKey { core, dtype, shape },
    };
    kernel.run(device)?;
    Ok(output)
}

fn broadcast_program(key: BroadcastProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX],
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        vec![0, 0, key.shape.tile_count],
        vec![
            0,
            0,
            key.shape.tile_count,
            key.shape.mode.reader_mode_arg(),
            key.shape.output_tiles_per_row,
        ],
        vec![key.shape.tile_count],
    )?;
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: BROADCAST_READER.to_owned(),
        compute_kernel: broadcast_compute_source(key.shape.mode),
        writer_kernel: BROADCAST_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("broadcast_in_dim_{:?}", key.dtype),
        ..Program::new(runtime_args)
    })
}

fn broadcast_compute_source(mode: BroadcastMode) -> String {
    BROADCAST_COMPUTE.replace("BROADCAST_COMPUTE_OP", &mode.compute_op_arg().to_string())
}

fn validate_rank(shape: &[usize], name: &str) -> io::Result<()> {
    if shape.len() <= 2 {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "broadcast_in_dim currently supports rank <= 2 {name} shapes, got {shape:?}"
        )))
    }
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

fn broadcast_mode(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> io::Result<BroadcastMode> {
    let input_rank = input_shape.len();
    let output_rank = output_shape.len();
    let (input_rows, input_cols) = logical_matrix_view(input_shape);
    let (output_rows, output_cols) = logical_matrix_view(output_shape);

    match input_rank {
        0 => {
            if output_rank == 0 {
                Ok(BroadcastMode::Copy)
            } else {
                Ok(BroadcastMode::Scalar)
            }
        }
        1 => rank1_broadcast_mode(
            input_cols,
            output_rank,
            output_rows,
            output_cols,
            broadcast_dimensions,
        ),
        2 => rank2_broadcast_mode(input_rows, input_cols, output_rows, output_cols),
        _ => unreachable!("broadcast rank validation should reject rank > 2"),
    }
}

fn rank1_broadcast_mode(
    input_cols: usize,
    output_rank: usize,
    output_rows: usize,
    output_cols: usize,
    broadcast_dimensions: &[i64],
) -> io::Result<BroadcastMode> {
    if input_cols == 1 {
        return Ok(BroadcastMode::Scalar);
    }
    match output_rank {
        1 => Ok(BroadcastMode::Copy),
        2 => match broadcast_dimensions[0] {
            0 if output_cols == 1 => Ok(BroadcastMode::Transpose),
            0 => Err(invalid_input(format!(
                "broadcast_in_dim does not support rank-1 dimension 0 broadcasts to output shape [{output_rows}, {output_cols}] without a native transpose+broadcast kernel"
            ))),
            1 if output_rows == 1 => Ok(BroadcastMode::Copy),
            1 => Ok(BroadcastMode::Row),
            dim => Err(invalid_input(format!(
                "broadcast dimension {dim} is not supported for rank-1 input"
            ))),
        },
        _ => unreachable!("broadcast rank validation should reject rank > 2"),
    }
}

fn rank2_broadcast_mode(
    input_rows: usize,
    input_cols: usize,
    output_rows: usize,
    output_cols: usize,
) -> io::Result<BroadcastMode> {
    if input_rows == output_rows && input_cols == output_cols {
        return Ok(BroadcastMode::Copy);
    }
    if input_rows == 1 && input_cols == 1 {
        return Ok(BroadcastMode::Scalar);
    }
    if input_rows == 1 && input_cols == output_cols {
        return Ok(BroadcastMode::Row);
    }
    if input_cols == 1 && input_rows == output_rows {
        return Ok(BroadcastMode::Col);
    }
    Err(invalid_input(format!(
        "broadcast_in_dim cannot lower [{input_rows}, {input_cols}] to [{output_rows}, {output_cols}] with the native tile broadcast kernel"
    )))
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

    #[test]
    fn broadcast_plan_normalizes_rank1_column_case() {
        let plan = BroadcastInDimPlan::new(&[32], &[32, 1], &[0]).expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(
            plan.kernel_shape(),
            BroadcastKernelShape {
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
        assert_eq!(plan.kernel_shape().mode, BroadcastMode::Row);
    }

    #[test]
    fn broadcast_plan_rejects_incompatible_mapped_dimensions() {
        let err = BroadcastInDimPlan::new(&[4], &[8, 1], &[0])
            .expect_err("incompatible broadcast should fail");

        assert!(err.to_string().contains("incompatible"));
    }

    #[test]
    fn broadcast_plan_rejects_rank1_transpose_plus_broadcast() {
        let err = BroadcastInDimPlan::new(&[4], &[4, 8], &[0])
            .expect_err("unsupported native broadcast should fail");

        assert!(err.to_string().contains("transpose+broadcast"));
    }
}
