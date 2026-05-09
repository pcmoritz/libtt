use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    buffer_shape_matches, tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{Kernel, RuntimeArgsBuilder};
use std::io;

const BROADCAST_IN_DIM: &str = include_str!("../../kernels/broadcast_vector_to_column.cc");
const INPUT_ADDR_INDEX: usize = 0;
const OUTPUT_ADDR_INDEX: usize = 1;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct BroadcastKernelShape {
    input_rank: u32,
    input_rows: u32,
    input_cols: u32,
    output_rows: u32,
    output_cols: u32,
    input_tiles_per_row: u32,
    output_tiles_per_row: u32,
    dim0: u32,
    dim1: u32,
    tile_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BroadcastInDimPlan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) input_allocation_shape: Vec<usize>,
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

        let input_allocation_shape = tiled_allocation_shape(input_shape)?;
        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let (input_rows, input_cols) = logical_matrix_view(input_shape);
        let (output_rows, output_cols) = logical_matrix_view(output_shape);
        let input_tiles_per_row = input_allocation_shape[input_allocation_shape.len() - 1] / TILE_C;
        let output_tiles_per_row =
            output_allocation_shape[output_allocation_shape.len() - 1] / TILE_C;
        let mapped_dims = mapped_physical_dimensions(output_shape.len(), broadcast_dimensions)?;
        let tile_count = tiled_shape_tile_count(output_shape)?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_shape: output_shape.to_vec(),
            input_allocation_shape,
            output_allocation_shape,
            kernel_shape: BroadcastKernelShape {
                input_rank: u32_arg(input_shape.len(), "input rank")?,
                input_rows: u32_arg(input_rows, "input rows")?,
                input_cols: u32_arg(input_cols, "input cols")?,
                output_rows: u32_arg(output_rows, "output rows")?,
                output_cols: u32_arg(output_cols, "output cols")?,
                input_tiles_per_row: u32_arg(input_tiles_per_row, "input tiles per row")?,
                output_tiles_per_row: u32_arg(output_tiles_per_row, "output tiles per row")?,
                dim0: mapped_dims.first().copied().unwrap_or(0),
                dim1: mapped_dims.get(1).copied().unwrap_or(0),
                tile_count: u32_arg(tile_count, "tile count")?,
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
            INPUT_ADDR_INDEX => Some(self.input_addr),
            OUTPUT_ADDR_INDEX => Some(self.output_addr),
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
    if !buffer_shape_matches(&input.shape, &plan.input_shape)? {
        return Err(invalid_input(format!(
            "broadcast input shape mismatch: got {:?}, expected {:?}",
            input.shape, plan.input_shape
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
        Vec::new(),
        vec![INPUT_ADDR_INDEX, OUTPUT_ADDR_INDEX],
        Vec::new(),
    );
    runtime_args.add_core(
        key.core,
        Vec::new(),
        vec![
            0,
            0,
            0,
            key.shape.tile_count,
            key.shape.input_rank,
            key.shape.input_rows,
            key.shape.input_cols,
            key.shape.output_rows,
            key.shape.output_cols,
            key.shape.input_tiles_per_row,
            key.shape.output_tiles_per_row,
            key.shape.dim0,
            key.shape.dim1,
        ],
        Vec::new(),
    )?;
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: BROADCAST_IN_DIM.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("broadcast_in_dim_{:?}", key.dtype),
        ..Program::new(runtime_args)
    })
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

fn mapped_physical_dimensions(
    output_rank: usize,
    broadcast_dimensions: &[i64],
) -> io::Result<Vec<u32>> {
    broadcast_dimensions
        .iter()
        .map(|&dim| {
            let dim = usize::try_from(dim).map_err(|_| {
                invalid_input(format!("broadcast dimension must be non-negative, got {dim}"))
            })?;
            match output_rank {
                0 => Err(invalid_input("scalar outputs cannot have broadcast dimensions")),
                1 => {
                    if dim == 0 {
                        Ok(1)
                    } else {
                        Err(invalid_input(format!(
                            "broadcast dimension {dim} is out of bounds for output rank 1"
                        )))
                    }
                }
                2 => Ok(u32_arg(dim, "broadcast dimension")?),
                _ => Err(invalid_input(format!(
                    "broadcast_in_dim currently supports rank <= 2 output shapes, got rank {output_rank}"
                ))),
            }
        })
        .collect()
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
    fn broadcast_plan_normalizes_vector_to_column() {
        let plan = BroadcastInDimPlan::new(&[32], &[32, 1], &[0]).expect("valid broadcast");

        assert_eq!(plan.input_allocation_shape, vec![32, 32]);
        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(
            plan.kernel_shape(),
            BroadcastKernelShape {
                input_rank: 1,
                input_rows: 1,
                input_cols: 32,
                output_rows: 32,
                output_cols: 1,
                input_tiles_per_row: 1,
                output_tiles_per_row: 1,
                dim0: 0,
                dim1: 0,
                tile_count: 1,
            }
        );
    }

    #[test]
    fn broadcast_plan_allows_degenerate_matrix_dimensions() {
        let plan = BroadcastInDimPlan::new(&[1, 4], &[8, 4], &[0, 1]).expect("valid broadcast");

        assert_eq!(plan.input_allocation_shape, vec![32, 32]);
        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(plan.kernel_shape().output_rows, 8);
        assert_eq!(plan.kernel_shape().output_cols, 4);
    }

    #[test]
    fn broadcast_plan_rejects_incompatible_mapped_dimensions() {
        let err = BroadcastInDimPlan::new(&[4], &[8, 1], &[0])
            .expect_err("incompatible broadcast should fail");

        assert!(err.to_string().contains("incompatible"));
    }
}
