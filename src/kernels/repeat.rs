use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/repeat_axis1_reader.cc");
const WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RepeatAxis1KernelShape {
    input_rows: u32,
    output_rows: u32,
    cols: u32,
    repeats: u32,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    tile_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RepeatAxis1Plan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_allocation_shape: Vec<usize>,
    kernel_shape: RepeatAxis1KernelShape,
}

impl RepeatAxis1Plan {
    pub(crate) fn new(input_shape: &[usize], output_shape: &[usize]) -> io::Result<Self> {
        let kernel_shape = repeat_axis1_kernel_shape(input_shape, output_shape)?;
        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape: tiled_allocation_shape(output_shape)?,
            kernel_shape,
        })
    }

    fn kernel_shape(&self) -> &RepeatAxis1KernelShape {
        &self.kernel_shape
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RepeatAxis1ProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: RepeatAxis1KernelShape,
}

struct RepeatAxis1Kernel {
    input_addr: u32,
    output_addr: u32,
    key: RepeatAxis1ProgramKey,
}

impl Kernel<RepeatAxis1ProgramKey> for RepeatAxis1Kernel {
    fn program_key(&self) -> RepeatAxis1ProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        repeat_axis1_program(self.key.clone())
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

pub(crate) fn repeat_axis1(
    device: &mut Device,
    input: &DramBuffer,
    plan: &RepeatAxis1Plan,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "repeat_axis1 input requires {:?}, got {:?}",
            dtype, input.dtype
        )));
    }
    let expected_input_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_input_shape {
        return Err(invalid_input(format!(
            "repeat_axis1 input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_input_shape, plan.input_shape
        )));
    }

    let input_tile_count = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != input_tile_count {
        return Err(invalid_input(format!(
            "repeat_axis1 input tile count mismatch: got {}, expected {input_tile_count}",
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
    let kernel = RepeatAxis1Kernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: RepeatAxis1ProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn repeat_axis1_kernel_shape(
    input_shape: &[usize],
    output_shape: &[usize],
) -> io::Result<RepeatAxis1KernelShape> {
    if input_shape.len() != 3 || output_shape.len() != 3 {
        return Err(invalid_input(format!(
            "repeat_axis1 requires rank-3 input/output, got {input_shape:?} -> {output_shape:?}"
        )));
    }
    if input_shape[0] != output_shape[0] || input_shape[2] != output_shape[2] {
        return Err(invalid_input(format!(
            "repeat_axis1 requires matching batch and width, got {input_shape:?} -> {output_shape:?}"
        )));
    }
    if input_shape[1] == 0
        || output_shape[1] <= input_shape[1]
        || output_shape[1] % input_shape[1] != 0
    {
        return Err(invalid_input(format!(
            "repeat_axis1 output axis must be a nontrivial multiple of input axis, got {input_shape:?} -> {output_shape:?}"
        )));
    }

    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let tile_count = tiled_shape_tile_count(output_shape)?;
    let repeats = output_shape[1] / input_shape[1];
    Ok(RepeatAxis1KernelShape {
        input_rows: u32_arg(input_shape[1], "input rows")?,
        output_rows: u32_arg(output_shape[1], "output rows")?,
        cols: u32_arg(output_shape[2], "cols")?,
        repeats: u32_arg(repeats, "repeats")?,
        input_tile_rows: u32_arg(input_allocation_shape[1] / TILE_R, "input tile rows")?,
        input_tiles_per_row: u32_arg(input_allocation_shape[2] / TILE_C, "input tiles per row")?,
        output_tile_rows: u32_arg(output_allocation_shape[1] / TILE_R, "output tile rows")?,
        output_tiles_per_row: u32_arg(output_allocation_shape[2] / TILE_C, "output tiles per row")?,
        tile_count: u32_arg(tile_count, "tile count")?,
    })
}

fn repeat_axis1_program(key: RepeatAxis1ProgramKey) -> io::Result<Program> {
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
        reader_kernel: repeat_axis1_reader_source(key.dtype, &key.shape)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("repeat_axis1_{:?}_{}", key.dtype, key.shape.repeats),
        ..Program::new(runtime_args)
    })
}

fn repeat_axis1_reader_source(dtype: DType, shape: &RepeatAxis1KernelShape) -> io::Result<String> {
    let element_type = match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    };
    Ok(format!(
        "#define REPEAT_INPUT_ROWS {}\n\
         #define REPEAT_OUTPUT_ROWS {}\n\
         #define REPEAT_COLS {}\n\
         #define REPEAT_FACTOR {}\n\
         #define REPEAT_INPUT_TILE_ROWS {}\n\
         #define REPEAT_INPUT_TILES_PER_ROW {}\n\
         #define REPEAT_OUTPUT_TILE_ROWS {}\n\
         #define REPEAT_OUTPUT_TILES_PER_ROW {}\n\
         #define REPEAT_ELEMENT_TYPE {element_type}\n\
         {READER}",
        shape.input_rows,
        shape.output_rows,
        shape.cols,
        shape.repeats,
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
    fn repeat_axis1_plan_describes_qwen_kv_repeat() {
        let plan = RepeatAxis1Plan::new(&[18, 2, 32], &[18, 4, 32]).expect("valid repeat_axis1");

        assert_eq!(plan.output_allocation_shape, vec![18, 32, 32]);
        assert_eq!(
            plan.kernel_shape(),
            &RepeatAxis1KernelShape {
                input_rows: 2,
                output_rows: 4,
                cols: 32,
                repeats: 2,
                input_tile_rows: 1,
                input_tiles_per_row: 1,
                output_tile_rows: 1,
                output_tiles_per_row: 1,
                tile_count: 18,
            }
        );
    }

    #[test]
    fn repeat_axis1_plan_rejects_non_multiple_axis() {
        let err = RepeatAxis1Plan::new(&[18, 3, 32], &[18, 4, 32])
            .expect_err("non-multiple repeat should fail");

        assert!(err.to_string().contains("nontrivial multiple"));
    }
}
