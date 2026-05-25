use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const CONCATENATE_READER: &str = include_str!("../../kernels/concatenate_reader.cc");
const CONCATENATE_WRITER: &str = include_str!("../../kernels/concatenate_writer.cc");
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ConcatenateAxis {
    Rows,
    Cols,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ConcatenateInputShape {
    rows: u32,
    cols: u32,
    tile_rows: u32,
    tiles_per_row: u32,
    concat_offset: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ConcatenateKernelShape {
    axis: ConcatenateAxis,
    output_rows: u32,
    output_cols: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    tile_count: u32,
    inputs: Vec<ConcatenateInputShape>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ConcatenateProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: ConcatenateKernelShape,
}

struct ConcatenateKernel {
    input_addrs: Vec<u32>,
    output_addr: u32,
    key: ConcatenateProgramKey,
}

impl Kernel<ConcatenateProgramKey> for ConcatenateKernel {
    fn program_key(&self) -> ConcatenateProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        concatenate_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        self.input_addrs.get(index).copied()
    }

    #[inline]
    fn writer_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            WRITER_OUTPUT_ADDR_INDEX => Some(self.output_addr),
            _ => None,
        }
    }
}

pub(crate) fn concatenate(
    device: &mut Device,
    inputs: &[&DramBuffer],
    input_shapes: &[Vec<usize>],
    output_shape: &[usize],
    dimension: usize,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let shape = concatenate_shape(input_shapes, output_shape, dimension)?;
    if inputs.len() != input_shapes.len() {
        return Err(invalid_input(format!(
            "concatenate input buffer count {} does not match input shape count {}",
            inputs.len(),
            input_shapes.len()
        )));
    }

    for (index, (input, logical_shape)) in inputs.iter().zip(input_shapes).enumerate() {
        validate_input(input, dtype, logical_shape, index)?;
    }

    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_tiles = tiled_shape_tile_count(output_shape)?;
    let output = device.alloc(output_tiles, dtype, &output_allocation_shape, name)?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let kernel = ConcatenateKernel {
        input_addrs: inputs
            .iter()
            .enumerate()
            .map(|(index, input)| u32_addr(input.addr, &format!("input {index} address")))
            .collect::<io::Result<Vec<_>>>()?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: ConcatenateProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn concatenate_shape(
    input_shapes: &[Vec<usize>],
    output_shape: &[usize],
    dimension: usize,
) -> io::Result<ConcatenateKernelShape> {
    if input_shapes.len() < 2 {
        return Err(invalid_input(
            "concatenate requires at least two input tensors",
        ));
    }
    let rank = output_shape.len();
    let axis = if rank == 1 {
        if dimension != 0 {
            return Err(invalid_input(format!(
                "concatenate rank-1 tensors require dimension 0, got {dimension}"
            )));
        }
        ConcatenateAxis::Cols
    } else if dimension == rank - 1 {
        ConcatenateAxis::Cols
    } else if dimension == rank - 2 {
        ConcatenateAxis::Rows
    } else {
        return Err(invalid_input(format!(
            "concatenate currently supports dimensions {} and {} for rank-{rank} tensors, got {dimension}",
            rank - 2,
            rank - 1
        )));
    };

    let mut concat_dim_total = 0usize;
    for (index, input_shape) in input_shapes.iter().enumerate() {
        if input_shape.len() != rank {
            return Err(invalid_input(format!(
                "concatenate input {index} rank {} must match output rank {rank}",
                input_shape.len()
            )));
        }
        for (dim, (&input_dim, &output_dim)) in input_shape.iter().zip(output_shape).enumerate() {
            if dim == dimension {
                continue;
            }
            if input_dim != output_dim {
                return Err(invalid_input(format!(
                    "concatenate input {index} shape {input_shape:?} does not match output shape {output_shape:?} outside dimension {dimension}",
                )));
            }
        }
        concat_dim_total = concat_dim_total
            .checked_add(input_shape[dimension])
            .ok_or_else(|| invalid_input("concatenate output dimension overflow"))?;
    }
    if concat_dim_total != output_shape[dimension] {
        return Err(invalid_input(format!(
            "concatenate output dimension {dimension} mismatch: input dimensions sum to {concat_dim_total}, output shape is {output_shape:?}"
        )));
    }

    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let output_allocation_rank = output_allocation_shape.len();
    let output_tile_rows = output_allocation_shape[output_allocation_rank - 2] / TILE_R;
    let output_tiles_per_row = output_allocation_shape[output_allocation_rank - 1] / TILE_C;
    let tile_count = tiled_shape_tile_count(output_shape)?;
    let mut concat_offset = 0usize;
    let mut inputs = Vec::with_capacity(input_shapes.len());
    for input_shape in input_shapes {
        let allocation_shape = tiled_allocation_shape(input_shape)?;
        let allocation_rank = allocation_shape.len();
        let (rows, cols) = if rank == 1 {
            (1usize, input_shape[0])
        } else {
            (input_shape[rank - 2], input_shape[rank - 1])
        };
        inputs.push(ConcatenateInputShape {
            rows: u32_arg(rows, "input rows")?,
            cols: u32_arg(cols, "input cols")?,
            tile_rows: u32_arg(
                allocation_shape[allocation_rank - 2] / TILE_R,
                "input tile rows",
            )?,
            tiles_per_row: u32_arg(
                allocation_shape[allocation_rank - 1] / TILE_C,
                "input tiles per row",
            )?,
            concat_offset: u32_arg(concat_offset, "concat offset")?,
        });
        concat_offset = concat_offset
            .checked_add(input_shape[dimension])
            .ok_or_else(|| invalid_input("concatenate offset overflow"))?;
    }

    Ok(ConcatenateKernelShape {
        axis,
        output_rows: u32_arg(
            if rank == 1 {
                1
            } else {
                output_shape[rank - 2]
            },
            "output rows",
        )?,
        output_cols: u32_arg(
            if rank == 1 {
                output_shape[0]
            } else {
                output_shape[rank - 1]
            },
            "output cols",
        )?,
        output_tile_rows: u32_arg(output_tile_rows, "output tile rows")?,
        output_tiles_per_row: u32_arg(output_tiles_per_row, "output tiles per row")?,
        tile_count: u32_arg(tile_count, "tile count")?,
        inputs,
    })
}

fn validate_input(
    input: &DramBuffer,
    dtype: DType,
    logical_shape: &[usize],
    index: usize,
) -> io::Result<()> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "concatenate input {index} requires {dtype:?}, got {:?}",
            input.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "concatenate input {index} allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, logical_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(logical_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "concatenate input {index} tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn concatenate_program(key: ConcatenateProgramKey) -> io::Result<Program> {
    let input_count = key.shape.inputs.len();
    let reader_dynamic_indices = (0..input_count).collect::<Vec<_>>();
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        reader_dynamic_indices,
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.tile_count, core_index, key.cores.len())?;
        let mut reader = vec![0; input_count];
        reader.extend([
            offset,
            n_tiles,
            key.shape.output_rows,
            key.shape.output_cols,
            key.shape.output_tile_rows,
            key.shape.output_tiles_per_row,
        ]);
        for input in &key.shape.inputs {
            reader.extend([
                input.rows,
                input.cols,
                input.tile_rows,
                input.tiles_per_row,
                input.concat_offset,
            ]);
        }
        runtime_args.add_core(core, vec![0, offset, n_tiles], reader, Vec::new())?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: concatenate_reader_source(key.dtype, key.shape.axis, input_count)?,
        writer_kernel: CONCATENATE_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!(
            "concatenate_{:?}_{:?}_{input_count}",
            key.dtype, key.shape.axis
        ),
        ..Program::new(runtime_args)
    })
}

fn concatenate_reader_source(
    dtype: DType,
    axis: ConcatenateAxis,
    input_count: usize,
) -> io::Result<String> {
    let element_type = match dtype {
        DType::Float32 | DType::Int32 | DType::UInt32 => "uint32_t",
        DType::Float16 | DType::Float16B | DType::UInt16 => "uint16_t",
        DType::Int8 | DType::UInt8 => "uint8_t",
    };
    let axis_cols = matches!(axis, ConcatenateAxis::Cols);
    Ok(format!(
        "#define CONCAT_INPUT_COUNT {input_count}\n#define CONCAT_ELEMENT_TYPE {element_type}\n#define CONCAT_AXIS_COLS {}\n{CONCATENATE_READER}",
        if axis_cols { 1 } else { 0 }
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
    fn concatenate_shape_describes_last_dim_concat() {
        let shape = concatenate_shape(&[vec![18, 16], vec![18, 16]], &[18, 32], 1)
            .expect("shape should be supported");

        assert_eq!(shape.axis, ConcatenateAxis::Cols);
        assert_eq!(shape.output_rows, 18);
        assert_eq!(shape.output_cols, 32);
        assert_eq!(shape.output_tile_rows, 1);
        assert_eq!(shape.output_tiles_per_row, 1);
        assert_eq!(shape.tile_count, 1);
        assert_eq!(shape.inputs[0].concat_offset, 0);
        assert_eq!(shape.inputs[1].concat_offset, 16);
    }

    #[test]
    fn concatenate_shape_describes_row_concat() {
        let shape = concatenate_shape(&[vec![2, 4], vec![3, 4]], &[5, 4], 0)
            .expect("shape should be supported");

        assert_eq!(shape.axis, ConcatenateAxis::Rows);
        assert_eq!(shape.output_rows, 5);
        assert_eq!(shape.output_cols, 4);
        assert_eq!(shape.output_tile_rows, 1);
        assert_eq!(shape.output_tiles_per_row, 1);
        assert_eq!(shape.tile_count, 1);
        assert_eq!(shape.inputs[0].concat_offset, 0);
        assert_eq!(shape.inputs[1].concat_offset, 2);
    }

    #[test]
    fn concatenate_shape_describes_rank1_concat() {
        let shape =
            concatenate_shape(&[vec![0], vec![1]], &[1], 0).expect("shape should be supported");

        assert_eq!(shape.axis, ConcatenateAxis::Cols);
        assert_eq!(shape.output_rows, 1);
        assert_eq!(shape.output_cols, 1);
        assert_eq!(shape.output_tile_rows, 1);
        assert_eq!(shape.output_tiles_per_row, 1);
        assert_eq!(shape.tile_count, 1);
        assert_eq!(shape.inputs[0].rows, 1);
        assert_eq!(shape.inputs[0].cols, 0);
        assert_eq!(shape.inputs[1].concat_offset, 0);
        assert_eq!(shape.inputs[1].cols, 1);
    }

    #[test]
    fn concatenate_shape_rejects_batch_dimension_concat() {
        let err = concatenate_shape(&[vec![2, 3, 4], vec![2, 5, 4]], &[2, 8, 4], 0)
            .expect_err("batch dimension concat should not be accepted");
        assert!(err.to_string().contains("currently supports dimensions"));
    }
}
