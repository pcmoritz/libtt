use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/take_axis1_reader.cc");
const WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TakeAxis1KernelShape {
    axis1_index: u32,
    output_rows: u32,
    output_cols: u32,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    tile_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TakeAxis1Plan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_allocation_shape: Vec<usize>,
    kernel_shape: TakeAxis1KernelShape,
}

impl TakeAxis1Plan {
    pub(crate) fn new(
        input_shape: &[usize],
        output_shape: &[usize],
        axis1_index: usize,
    ) -> io::Result<Self> {
        let kernel_shape = take_axis1_kernel_shape(input_shape, output_shape, axis1_index)?;
        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape: tiled_allocation_shape(output_shape)?,
            kernel_shape,
        })
    }

    fn kernel_shape(&self) -> &TakeAxis1KernelShape {
        &self.kernel_shape
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TakeAxis1ProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: TakeAxis1KernelShape,
}

struct TakeAxis1Kernel {
    input_addr: u32,
    output_addr: u32,
    key: TakeAxis1ProgramKey,
}

impl Kernel<TakeAxis1ProgramKey> for TakeAxis1Kernel {
    fn program_key(&self) -> TakeAxis1ProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        take_axis1_program(self.key.clone())
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

pub(crate) fn take_axis1(
    device: &mut Device,
    input: &DramBuffer,
    plan: &TakeAxis1Plan,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "take_axis1 input requires {:?}, got {:?}",
            dtype, input.dtype
        )));
    }
    let expected_input_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_input_shape {
        return Err(invalid_input(format!(
            "take_axis1 input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_input_shape, plan.input_shape
        )));
    }

    let input_tile_count = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != input_tile_count {
        return Err(invalid_input(format!(
            "take_axis1 input tile count mismatch: got {}, expected {input_tile_count}",
            input.num_tiles
        )));
    }

    let shape = plan.kernel_shape().clone();
    let output_tiles = usize::try_from(shape.tile_count).map_err(|_| {
        invalid_input(format!(
            "tile count does not fit in usize: {}",
            shape.tile_count
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(output_tiles, dtype, &plan.output_allocation_shape, name)?;
    let kernel = TakeAxis1Kernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: TakeAxis1ProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn take_axis1_kernel_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    axis1_index: usize,
) -> io::Result<TakeAxis1KernelShape> {
    if input_shape.len() != 3 || output_shape.len() != 2 {
        return Err(invalid_input(format!(
            "take_axis1 requires rank-3 input and rank-2 output, got {input_shape:?} -> {output_shape:?}"
        )));
    }
    if axis1_index >= input_shape[1] {
        return Err(invalid_input(format!(
            "take_axis1 index {axis1_index} is out of bounds for axis size {}",
            input_shape[1]
        )));
    }
    if input_shape[0] != output_shape[0] || input_shape[2] != output_shape[1] {
        return Err(invalid_input(format!(
            "take_axis1 output must be input batch/width, got {input_shape:?} -> {output_shape:?}"
        )));
    }

    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let tile_count = tiled_shape_tile_count(output_shape)?;
    Ok(TakeAxis1KernelShape {
        axis1_index: u32_arg(axis1_index, "axis1 index")?,
        output_rows: u32_arg(output_shape[0], "output rows")?,
        output_cols: u32_arg(output_shape[1], "output cols")?,
        input_tile_rows: u32_arg(input_allocation_shape[1] / TILE_R, "input tile rows")?,
        input_tiles_per_row: u32_arg(input_allocation_shape[2] / TILE_C, "input tiles per row")?,
        output_tile_rows: u32_arg(output_allocation_shape[0] / TILE_R, "output tile rows")?,
        output_tiles_per_row: u32_arg(output_allocation_shape[1] / TILE_C, "output tiles per row")?,
        tile_count: u32_arg(tile_count, "tile count")?,
    })
}

fn take_axis1_program(key: TakeAxis1ProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_INPUT_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, offset, n_tiles],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: take_axis1_reader_source(key.dtype, &key.shape)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("take_axis1_{:?}_{}", key.dtype, key.shape.axis1_index),
        ..Program::new(runtime_args)
    })
}

fn take_axis1_reader_source(dtype: DType, shape: &TakeAxis1KernelShape) -> io::Result<String> {
    let element_type = match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    };
    Ok(format!(
        "#define TAKE_AXIS1_INDEX {}\n\
         #define TAKE_AXIS1_OUTPUT_ROWS {}\n\
         #define TAKE_AXIS1_OUTPUT_COLS {}\n\
         #define TAKE_AXIS1_INPUT_TILE_ROWS {}\n\
         #define TAKE_AXIS1_INPUT_TILES_PER_ROW {}\n\
         #define TAKE_AXIS1_OUTPUT_TILE_ROWS {}\n\
         #define TAKE_AXIS1_OUTPUT_TILES_PER_ROW {}\n\
         #define TAKE_AXIS1_ELEMENT_TYPE {element_type}\n\
         {READER}",
        shape.axis1_index,
        shape.output_rows,
        shape.output_cols,
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
    ))
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
    fn take_axis1_plan_describes_head_extract() {
        let plan = TakeAxis1Plan::new(&[18, 4, 32], &[18, 32], 3).expect("valid take_axis1");

        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(
            plan.kernel_shape(),
            &TakeAxis1KernelShape {
                axis1_index: 3,
                output_rows: 18,
                output_cols: 32,
                input_tile_rows: 1,
                input_tiles_per_row: 1,
                output_tile_rows: 1,
                output_tiles_per_row: 1,
                tile_count: 1,
            }
        );
    }

    #[test]
    fn take_axis1_plan_rejects_out_of_bounds_index() {
        let err = TakeAxis1Plan::new(&[18, 4, 32], &[18, 32], 4).expect_err("index should fail");

        assert!(err.to_string().contains("out of bounds"));
    }
}
