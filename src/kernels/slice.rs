use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const SLICE_READER: &str = include_str!("../../kernels/slice_reader.cc");
const SLICE_WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct SliceKernelShape {
    input_shape: Vec<u32>,
    output_shape: Vec<u32>,
    start_indices: Vec<u32>,
    strides: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    tile_count: u32,
    direct_copy: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlicePlan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_allocation_shape: Vec<usize>,
    kernel_shape: SliceKernelShape,
}

impl SlicePlan {
    pub(crate) fn new(
        input_shape: &[usize],
        output_shape: &[usize],
        start_indices: &[i64],
        limit_indices: &[i64],
        strides: &[i64],
    ) -> io::Result<Self> {
        validate_slice(
            input_shape,
            output_shape,
            start_indices,
            limit_indices,
            strides,
        )?;

        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let kernel_shape = slice_kernel_shape(input_shape, output_shape, start_indices, strides)?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            kernel_shape,
        })
    }

    fn kernel_shape(&self) -> &SliceKernelShape {
        &self.kernel_shape
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SliceProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: SliceKernelShape,
}

struct SliceKernel {
    input_addr: u32,
    output_addr: u32,
    key: SliceProgramKey,
}

impl Kernel<SliceProgramKey> for SliceKernel {
    fn program_key(&self) -> SliceProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        slice_program(self.key.clone())
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

pub(crate) fn slice(
    device: &mut Device,
    input: &DramBuffer,
    plan: &SlicePlan,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "slice input requires {:?}, got {:?}",
            dtype, input.dtype
        )));
    }
    let expected_input_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_input_shape {
        return Err(invalid_input(format!(
            "slice input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_input_shape, plan.input_shape
        )));
    }

    let input_tile_count = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != input_tile_count {
        return Err(invalid_input(format!(
            "slice input tile count mismatch: got {}, expected {input_tile_count}",
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
    let kernel = SliceKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: SliceProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn slice_kernel_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    start_indices: &[i64],
    strides: &[i64],
) -> io::Result<SliceKernelShape> {
    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let input_rank = input_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    let tile_count = tiled_shape_tile_count(output_shape)?;
    let start_indices_u32 = u32_indices(start_indices, "start index")?;
    let strides_u32 = u32_indices(strides, "stride")?;
    let direct_copy = input_shape == output_shape
        && start_indices_u32.iter().all(|&value| value == 0)
        && strides_u32.iter().all(|&value| value == 1);

    Ok(SliceKernelShape {
        input_shape: u32_shape(input_shape, "input shape")?,
        output_shape: u32_shape(output_shape, "output shape")?,
        start_indices: start_indices_u32,
        strides: strides_u32,
        input_tile_rows: u32_arg(
            input_allocation_shape[input_rank - 2] / TILE_R,
            "input tile rows",
        )?,
        input_tiles_per_row: u32_arg(
            input_allocation_shape[input_rank - 1] / TILE_C,
            "input tiles per row",
        )?,
        output_tile_rows: u32_arg(
            output_allocation_shape[output_rank - 2] / TILE_R,
            "output tile rows",
        )?,
        output_tiles_per_row: u32_arg(
            output_allocation_shape[output_rank - 1] / TILE_C,
            "output tiles per row",
        )?,
        tile_count: u32_arg(tile_count, "tile count")?,
        direct_copy,
    })
}

fn slice_program(key: SliceProgramKey) -> io::Result<Program> {
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
        reader_kernel: slice_reader_source(key.dtype, &key.shape)?,
        writer_kernel: SLICE_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("slice_{:?}_{}", key.dtype, key.shape.input_shape.len()),
        ..Program::new(runtime_args)
    })
}

fn slice_reader_source(dtype: DType, shape: &SliceKernelShape) -> io::Result<String> {
    let element_type = element_type(dtype);
    Ok(format!(
        "#define SLICE_RANK {}\n\
         #define SLICE_INPUT_SHAPE {}\n\
         #define SLICE_OUTPUT_SHAPE {}\n\
         #define SLICE_START_INDICES {}\n\
         #define SLICE_STRIDES {}\n\
         #define SLICE_INPUT_TILE_ROWS {}\n\
         #define SLICE_INPUT_TILES_PER_ROW {}\n\
         #define SLICE_OUTPUT_TILE_ROWS {}\n\
         #define SLICE_OUTPUT_TILES_PER_ROW {}\n\
         #define SLICE_DIRECT_COPY {}\n\
         #define SLICE_ELEMENT_TYPE {element_type}\n\
         {SLICE_READER}",
        shape.input_shape.len(),
        cpp_u32_array(&shape.input_shape),
        cpp_u32_array(&shape.output_shape),
        cpp_u32_array(&shape.start_indices),
        cpp_u32_array(&shape.strides),
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
        shape.direct_copy as u32,
    ))
}

fn validate_slice(
    input_shape: &[usize],
    output_shape: &[usize],
    start_indices: &[i64],
    limit_indices: &[i64],
    strides: &[i64],
) -> io::Result<()> {
    let rank = input_shape.len();
    if output_shape.len() != rank {
        return Err(invalid_input(format!(
            "slice output rank {} must match input rank {rank}",
            output_shape.len()
        )));
    }
    if start_indices.len() != rank || limit_indices.len() != rank || strides.len() != rank {
        return Err(invalid_input(format!(
            "slice index lengths must match rank {rank}: start={}, limit={}, strides={}",
            start_indices.len(),
            limit_indices.len(),
            strides.len()
        )));
    }

    for dim in 0..rank {
        let start = usize::try_from(start_indices[dim]).map_err(|_| {
            invalid_input(format!(
                "slice start index {dim} must be non-negative, got {}",
                start_indices[dim]
            ))
        })?;
        let limit = usize::try_from(limit_indices[dim]).map_err(|_| {
            invalid_input(format!(
                "slice limit index {dim} must be non-negative, got {}",
                limit_indices[dim]
            ))
        })?;
        let stride = usize::try_from(strides[dim]).map_err(|_| {
            invalid_input(format!(
                "slice stride {dim} must be positive, got {}",
                strides[dim]
            ))
        })?;
        if stride == 0 {
            return Err(invalid_input(format!(
                "slice stride {dim} must be positive"
            )));
        }
        if start > limit || limit > input_shape[dim] {
            return Err(invalid_input(format!(
                "slice dimension {dim} bounds [{start}, {limit}) are invalid for input size {}",
                input_shape[dim]
            )));
        }
        let extent = limit - start;
        let expected = extent.div_ceil(stride);
        if output_shape[dim] != expected {
            return Err(invalid_input(format!(
                "slice output dimension {dim} mismatch: expected {expected}, got {}",
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

    #[test]
    fn slice_plan_describes_strided_matrix_slice() {
        let plan =
            SlicePlan::new(&[4, 4], &[2, 2], &[0, 1], &[4, 3], &[2, 1]).expect("valid slice");

        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(
            plan.kernel_shape(),
            &SliceKernelShape {
                input_shape: vec![4, 4],
                output_shape: vec![2, 2],
                start_indices: vec![0, 1],
                strides: vec![2, 1],
                input_tile_rows: 1,
                input_tiles_per_row: 1,
                output_tile_rows: 1,
                output_tiles_per_row: 1,
                tile_count: 1,
                direct_copy: false,
            }
        );
    }

    #[test]
    fn slice_plan_supports_rank1_tail_slice() {
        let plan = SlicePlan::new(&[288], &[288], &[0], &[288], &[1]).expect("valid slice");

        assert_eq!(plan.output_allocation_shape, vec![32, 288]);
        assert_eq!(plan.kernel_shape().output_tiles_per_row, 9);
        assert_eq!(plan.kernel_shape().tile_count, 9);
        assert!(plan.kernel_shape().direct_copy);
    }

    #[test]
    fn slice_plan_rejects_wrong_output_extent() {
        let err = SlicePlan::new(&[4, 4], &[2, 3], &[0, 1], &[4, 3], &[2, 1])
            .expect_err("wrong output shape should fail");

        assert!(err.to_string().contains("output dimension 1 mismatch"));
    }

    #[test]
    fn slice_reader_source_uses_dummy_arrays_for_scalars() {
        let plan = SlicePlan::new(&[], &[], &[], &[], &[]).expect("valid scalar slice");
        let source = slice_reader_source(DType::Float16B, plan.kernel_shape()).expect("source");

        assert!(source.contains("#define SLICE_RANK 0"));
        assert!(source.contains("#define SLICE_INPUT_SHAPE {1u}"));
        assert!(source.contains("#define SLICE_START_INDICES {1u}"));
    }

    #[test]
    fn slice_program_splits_tiles_across_cores() {
        let plan = SlicePlan::new(
            &[5, 64, 64],
            &[5, 64, 64],
            &[0, 0, 0],
            &[5, 64, 64],
            &[1, 1, 1],
        )
        .expect("valid slice");
        let program = slice_program(SliceProgramKey {
            cores: vec![
                CoreCoord { x: 1, y: 2 },
                CoreCoord { x: 1, y: 3 },
                CoreCoord { x: 1, y: 4 },
            ],
            dtype: DType::Float16B,
            shape: plan.kernel_shape().clone(),
        })
        .expect("slice program");

        assert_eq!(program.runtime_args.cores().len(), 3);
        assert_eq!(program.runtime_args.section_sizes(), (12, 12, 0));
        assert!(program.compute_kernel.is_empty());
        assert!(plan.kernel_shape().direct_copy);
        assert!(program.reader_kernel.contains("#define SLICE_RANK 3"));
        assert!(program
            .reader_kernel
            .contains("#define SLICE_DIRECT_COPY 1"));

        let blobs = program.runtime_args.blobs();
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 7));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (7, 7));
        assert_eq!((arg_u32(&blobs[2], 1), arg_u32(&blobs[2], 2)), (14, 6));
        assert_eq!((arg_u32(&blobs[0], 4), arg_u32(&blobs[0], 5)), (0, 7));
        assert_eq!((arg_u32(&blobs[1], 4), arg_u32(&blobs[1], 5)), (7, 7));
        assert_eq!((arg_u32(&blobs[2], 4), arg_u32(&blobs[2], 5)), (14, 6));
    }
}
