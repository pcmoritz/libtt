use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const BROADCAST_READER: &str = include_str!("../../kernels/broadcast_reader.cc");
const REPEAT_HEADS_READER: &str = include_str!("../../kernels/repeat_heads_reader.cc");
const STACK_HEADS_READER: &str = include_str!("../../kernels/stack_heads_reader.cc");
const BROADCAST_WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct BroadcastKernelShape {
    input_shape: Vec<u32>,
    output_shape: Vec<u32>,
    broadcast_dimensions: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    tile_count: u32,
    direct_copy: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BroadcastInDimPlan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_allocation_shape: Vec<usize>,
    kernel_shape: BroadcastKernelShape,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct RepeatHeadsKernelShape {
    input_shape: Vec<u32>,
    output_shape: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    repeat_factor: u32,
    tile_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RepeatHeadsPlan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_allocation_shape: Vec<usize>,
    kernel_shape: RepeatHeadsKernelShape,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct StackHeadsKernelShape {
    input_shape: Vec<u32>,
    output_shape: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    head_count: u32,
    tile_count: u32,
    flatten: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StackHeadsPlan {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_allocation_shape: Vec<usize>,
    kernel_shape: StackHeadsKernelShape,
}

impl BroadcastInDimPlan {
    pub(crate) fn new(
        input_shape: &[usize],
        output_shape: &[usize],
        broadcast_dimensions: &[i64],
    ) -> io::Result<Self> {
        validate_broadcast_dimensions(input_shape, output_shape, broadcast_dimensions)?;

        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let kernel_shape = broadcast_kernel_shape(input_shape, output_shape, broadcast_dimensions)?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            kernel_shape,
        })
    }

    fn kernel_shape(&self) -> &BroadcastKernelShape {
        &self.kernel_shape
    }
}

impl RepeatHeadsPlan {
    pub(crate) fn new(input_shape: &[usize], output_shape: &[usize]) -> io::Result<Self> {
        let kernel_shape = repeat_heads_kernel_shape(input_shape, output_shape)?;
        let output_allocation_shape = tiled_allocation_shape(output_shape)?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            kernel_shape,
        })
    }

    fn kernel_shape(&self) -> &RepeatHeadsKernelShape {
        &self.kernel_shape
    }
}

impl StackHeadsPlan {
    pub(crate) fn new(
        input_shape: &[usize],
        output_shape: &[usize],
        head_count: usize,
        flatten: bool,
    ) -> io::Result<Self> {
        let kernel_shape =
            stack_heads_kernel_shape(input_shape, output_shape, head_count, flatten)?;
        let output_allocation_shape = tiled_allocation_shape(output_shape)?;

        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            kernel_shape,
        })
    }

    fn kernel_shape(&self) -> &StackHeadsKernelShape {
        &self.kernel_shape
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

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RepeatHeadsProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: RepeatHeadsKernelShape,
}

struct RepeatHeadsKernel {
    input_addr: u32,
    output_addr: u32,
    key: RepeatHeadsProgramKey,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct StackHeadsProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: StackHeadsKernelShape,
}

struct StackHeadsKernel {
    input_addrs: Vec<u32>,
    output_addr: u32,
    key: StackHeadsProgramKey,
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

impl Kernel<RepeatHeadsProgramKey> for RepeatHeadsKernel {
    fn program_key(&self) -> RepeatHeadsProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        repeat_heads_program(self.key.clone())
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

impl Kernel<StackHeadsProgramKey> for StackHeadsKernel {
    fn program_key(&self) -> StackHeadsProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        stack_heads_program(self.key.clone())
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

    let shape = plan.kernel_shape().clone();
    let output_tiles = usize::try_from(shape.tile_count).map_err(|_| {
        invalid_input(format!(
            "tile count does not fit in usize: {}",
            shape.tile_count
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

pub(crate) fn repeat_heads(
    device: &mut Device,
    input: &DramBuffer,
    plan: &RepeatHeadsPlan,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if input.dtype != dtype {
        return Err(invalid_input(format!(
            "repeat_heads input requires {:?}, got {:?}",
            dtype, input.dtype
        )));
    }
    let expected_input_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_input_shape {
        return Err(invalid_input(format!(
            "repeat_heads input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_input_shape, plan.input_shape
        )));
    }

    let input_tile_count = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != input_tile_count {
        return Err(invalid_input(format!(
            "repeat_heads input tile count mismatch: got {}, expected {input_tile_count}",
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
    let kernel = RepeatHeadsKernel {
        input_addr: u32_addr(input.addr, "input address")?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: RepeatHeadsProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

pub(crate) fn stack_heads(
    device: &mut Device,
    inputs: &[&DramBuffer],
    plan: &StackHeadsPlan,
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    let head_count = usize::try_from(plan.kernel_shape.head_count).map_err(|_| {
        invalid_input(format!(
            "stack_heads head count does not fit in usize: {}",
            plan.kernel_shape.head_count
        ))
    })?;
    if inputs.len() != head_count {
        return Err(invalid_input(format!(
            "stack_heads expected {head_count} inputs, got {}",
            inputs.len()
        )));
    }

    let expected_input_shape = tiled_allocation_shape(&plan.input_shape)?;
    let input_tile_count = tiled_shape_tile_count(&plan.input_shape)?;
    for (index, input) in inputs.iter().enumerate() {
        if input.dtype != dtype {
            return Err(invalid_input(format!(
                "stack_heads input {index} requires {:?}, got {:?}",
                dtype, input.dtype
            )));
        }
        if input.shape != expected_input_shape {
            return Err(invalid_input(format!(
                "stack_heads input {index} allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
                input.shape, expected_input_shape, plan.input_shape
            )));
        }
        if input.num_tiles != input_tile_count {
            return Err(invalid_input(format!(
                "stack_heads input {index} tile count mismatch: got {}, expected {input_tile_count}",
                input.num_tiles
            )));
        }
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
    let kernel = StackHeadsKernel {
        input_addrs: inputs
            .iter()
            .enumerate()
            .map(|(index, input)| u32_addr(input.addr, &format!("input {index} address")))
            .collect::<io::Result<Vec<_>>>()?,
        output_addr: u32_addr(output.addr, "output address")?,
        key: StackHeadsProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn broadcast_kernel_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[i64],
) -> io::Result<BroadcastKernelShape> {
    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let input_rank = input_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    let tile_count = tiled_shape_tile_count(output_shape)?;

    let input_shape_u32 = u32_shape(input_shape, "input shape")?;
    let output_shape_u32 = u32_shape(output_shape, "output shape")?;
    let broadcast_dimensions_u32 = u32_broadcast_dimensions(broadcast_dimensions)?;
    let direct_copy =
        is_direct_copy_broadcast(input_shape, output_shape, &broadcast_dimensions_u32);

    Ok(BroadcastKernelShape {
        input_shape: input_shape_u32,
        output_shape: output_shape_u32,
        broadcast_dimensions: broadcast_dimensions_u32,
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

fn repeat_heads_kernel_shape(
    input_shape: &[usize],
    output_shape: &[usize],
) -> io::Result<RepeatHeadsKernelShape> {
    if input_shape.len() != 3 || output_shape.len() != 3 {
        return Err(invalid_input(format!(
            "repeat_heads expects rank-3 input and output, got {input_shape:?} -> {output_shape:?}"
        )));
    }
    if input_shape[0] != output_shape[0] || input_shape[2] != output_shape[2] {
        return Err(invalid_input(format!(
            "repeat_heads requires matching sequence/head_dim, got {input_shape:?} -> {output_shape:?}"
        )));
    }
    if input_shape[1] == 0
        || output_shape[1] == 0
        || output_shape[1] % input_shape[1] != 0
        || output_shape[1] < input_shape[1]
    {
        return Err(invalid_input(format!(
            "repeat_heads output head count must be a positive multiple of input heads, got {input_shape:?} -> {output_shape:?}"
        )));
    }

    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let input_rank = input_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    let tile_count = tiled_shape_tile_count(output_shape)?;

    Ok(RepeatHeadsKernelShape {
        input_shape: u32_shape(input_shape, "input shape")?,
        output_shape: u32_shape(output_shape, "output shape")?,
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
        repeat_factor: u32_arg(output_shape[1] / input_shape[1], "repeat factor")?,
        tile_count: u32_arg(tile_count, "tile count")?,
    })
}

fn stack_heads_kernel_shape(
    input_shape: &[usize],
    output_shape: &[usize],
    head_count: usize,
    flatten: bool,
) -> io::Result<StackHeadsKernelShape> {
    if input_shape.len() != 2 {
        return Err(invalid_input(format!(
            "stack_heads expects rank-2 inputs, got {input_shape:?}"
        )));
    }
    if head_count < 2 {
        return Err(invalid_input(format!(
            "stack_heads requires at least two heads, got {head_count}"
        )));
    }
    let expected_output = if flatten {
        if input_shape[1] % TILE_C != 0 {
            return Err(invalid_input(format!(
                "flattened stack_heads requires input cols to be tile-aligned, got {input_shape:?}"
            )));
        }
        vec![input_shape[0], head_count * input_shape[1]]
    } else {
        vec![input_shape[0], head_count, input_shape[1]]
    };
    if output_shape != expected_output.as_slice() {
        return Err(invalid_input(format!(
            "stack_heads output shape mismatch: expected {expected_output:?}, got {output_shape:?}"
        )));
    }

    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let input_rank = input_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    let tile_count = tiled_shape_tile_count(output_shape)?;

    Ok(StackHeadsKernelShape {
        input_shape: u32_shape(input_shape, "input shape")?,
        output_shape: u32_shape(output_shape, "output shape")?,
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
        head_count: u32_arg(head_count, "head count")?,
        tile_count: u32_arg(tile_count, "tile count")?,
        flatten,
    })
}

fn broadcast_program(key: BroadcastProgramKey) -> io::Result<Program> {
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
        reader_kernel: broadcast_reader_source(key.dtype, &key.shape)?,
        writer_kernel: BROADCAST_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!(
            "broadcast_in_dim_{:?}_{}_{}",
            key.dtype,
            key.shape.input_shape.len(),
            key.shape.output_shape.len()
        ),
        ..Program::new(runtime_args)
    })
}

fn repeat_heads_program(key: RepeatHeadsProgramKey) -> io::Result<Program> {
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
        reader_kernel: repeat_heads_reader_source(key.dtype, &key.shape)?,
        writer_kernel: BROADCAST_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("repeat_heads_{:?}", key.dtype),
        ..Program::new(runtime_args)
    })
}

fn stack_heads_program(key: StackHeadsProgramKey) -> io::Result<Program> {
    let head_count = usize::try_from(key.shape.head_count).map_err(|_| {
        invalid_input(format!(
            "stack_heads head count does not fit in usize: {}",
            key.shape.head_count
        ))
    })?;
    let reader_dynamic_indices = (0..head_count).collect::<Vec<_>>();
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        reader_dynamic_indices,
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.tile_count, core_index, key.cores.len())?;
        let mut reader = vec![0; head_count];
        reader.extend([offset, n_tiles]);
        runtime_args.add_core(core, vec![0, offset, n_tiles], reader, Vec::new())?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: stack_heads_reader_source(key.dtype, &key.shape)?,
        writer_kernel: BROADCAST_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!(
            "stack_heads_{:?}_{}_{}",
            key.dtype, key.shape.head_count, key.shape.flatten as u32
        ),
        ..Program::new(runtime_args)
    })
}

fn broadcast_reader_source(dtype: DType, shape: &BroadcastKernelShape) -> io::Result<String> {
    let element_type = element_type(dtype);
    Ok(format!(
        "#define BROADCAST_INPUT_RANK {}\n\
         #define BROADCAST_OUTPUT_RANK {}\n\
         #define BROADCAST_INPUT_SHAPE {}\n\
         #define BROADCAST_OUTPUT_SHAPE {}\n\
         #define BROADCAST_DIMENSIONS {}\n\
         #define BROADCAST_INPUT_TILE_ROWS {}\n\
         #define BROADCAST_INPUT_TILES_PER_ROW {}\n\
         #define BROADCAST_OUTPUT_TILE_ROWS {}\n\
         #define BROADCAST_OUTPUT_TILES_PER_ROW {}\n\
         #define BROADCAST_DIRECT_COPY {}\n\
         #define BROADCAST_ELEMENT_TYPE {element_type}\n\
         {BROADCAST_READER}",
        shape.input_shape.len(),
        shape.output_shape.len(),
        cpp_u32_array(&shape.input_shape),
        cpp_u32_array(&shape.output_shape),
        cpp_u32_array(&shape.broadcast_dimensions),
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
        shape.direct_copy as u32,
    ))
}

fn repeat_heads_reader_source(dtype: DType, shape: &RepeatHeadsKernelShape) -> io::Result<String> {
    let element_type = element_type(dtype);
    Ok(format!(
        "#define REPEAT_INPUT_SHAPE {}\n\
         #define REPEAT_OUTPUT_SHAPE {}\n\
         #define REPEAT_INPUT_TILE_ROWS {}\n\
         #define REPEAT_INPUT_TILES_PER_ROW {}\n\
         #define REPEAT_OUTPUT_TILE_ROWS {}\n\
         #define REPEAT_OUTPUT_TILES_PER_ROW {}\n\
         #define REPEAT_FACTOR {}\n\
         #define REPEAT_ELEMENT_TYPE {element_type}\n\
         {REPEAT_HEADS_READER}",
        cpp_u32_array(&shape.input_shape),
        cpp_u32_array(&shape.output_shape),
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
        shape.repeat_factor,
    ))
}

fn stack_heads_reader_source(dtype: DType, shape: &StackHeadsKernelShape) -> io::Result<String> {
    let element_type = element_type(dtype);
    Ok(format!(
        "#define STACK_INPUT_SHAPE {}\n\
         #define STACK_OUTPUT_SHAPE {}\n\
         #define STACK_INPUT_TILE_ROWS {}\n\
         #define STACK_INPUT_TILES_PER_ROW {}\n\
         #define STACK_OUTPUT_TILE_ROWS {}\n\
         #define STACK_OUTPUT_TILES_PER_ROW {}\n\
         #define STACK_HEAD_COUNT {}\n\
         #define STACK_FLATTEN {}\n\
         #define STACK_ELEMENT_TYPE {element_type}\n\
         {STACK_HEADS_READER}",
        cpp_u32_array(&shape.input_shape),
        cpp_u32_array(&shape.output_shape),
        shape.input_tile_rows,
        shape.input_tiles_per_row,
        shape.output_tile_rows,
        shape.output_tiles_per_row,
        shape.head_count,
        shape.flatten as u32,
    ))
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

fn u32_shape(shape: &[usize], name: &str) -> io::Result<Vec<u32>> {
    shape
        .iter()
        .enumerate()
        .map(|(index, &dim)| u32_arg(dim, &format!("{name} dimension {index}")))
        .collect()
}

fn u32_broadcast_dimensions(broadcast_dimensions: &[i64]) -> io::Result<Vec<u32>> {
    broadcast_dimensions
        .iter()
        .enumerate()
        .map(|(index, &dim)| {
            u32::try_from(dim).map_err(|_| {
                invalid_input(format!(
                    "broadcast dimension {index} does not fit in u32: {dim}"
                ))
            })
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

fn is_direct_copy_broadcast(
    input_shape: &[usize],
    output_shape: &[usize],
    broadcast_dimensions: &[u32],
) -> bool {
    input_shape == output_shape
        && broadcast_dimensions
            .iter()
            .enumerate()
            .all(|(index, &dim)| dim == index as u32)
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
    fn broadcast_plan_describes_rank1_column_case() {
        let plan = BroadcastInDimPlan::new(&[32], &[32, 1], &[0]).expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(
            plan.kernel_shape(),
            &BroadcastKernelShape {
                input_shape: vec![32],
                output_shape: vec![32, 1],
                broadcast_dimensions: vec![0],
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
    fn broadcast_plan_allows_degenerate_matrix_dimensions() {
        let plan = BroadcastInDimPlan::new(&[1, 4], &[8, 4], &[0, 1]).expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(plan.kernel_shape().input_shape, vec![1, 4]);
        assert_eq!(plan.kernel_shape().output_shape, vec![8, 4]);
    }

    #[test]
    fn broadcast_plan_allows_rank1_transpose_plus_broadcast() {
        let plan = BroadcastInDimPlan::new(&[4], &[4, 8], &[0]).expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![32, 32]);
        assert_eq!(plan.kernel_shape().broadcast_dimensions, vec![0]);
    }

    #[test]
    fn broadcast_plan_supports_inserted_middle_dimension() {
        let plan = BroadcastInDimPlan::new(&[18, 2, 32], &[18, 2, 2, 32], &[0, 1, 3])
            .expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![18, 2, 32, 32]);
        assert_eq!(plan.kernel_shape().input_shape, vec![18, 2, 32]);
        assert_eq!(plan.kernel_shape().output_shape, vec![18, 2, 2, 32]);
        assert_eq!(plan.kernel_shape().broadcast_dimensions, vec![0, 1, 3]);
        assert_eq!(plan.kernel_shape().tile_count, 36);
    }

    #[test]
    fn repeat_heads_plan_describes_kv_head_expansion() {
        let plan = RepeatHeadsPlan::new(&[42, 8, 128], &[42, 16, 128]).expect("valid repeat heads");

        assert_eq!(plan.output_allocation_shape, vec![42, 32, 128]);
        assert_eq!(
            plan.kernel_shape(),
            &RepeatHeadsKernelShape {
                input_shape: vec![42, 8, 128],
                output_shape: vec![42, 16, 128],
                input_tile_rows: 1,
                input_tiles_per_row: 4,
                output_tile_rows: 1,
                output_tiles_per_row: 4,
                repeat_factor: 2,
                tile_count: 42 * 4,
            }
        );
    }

    #[test]
    fn stack_heads_plan_describes_attention_score_stack() {
        let plan =
            StackHeadsPlan::new(&[42, 42], &[42, 16, 42], 16, false).expect("valid stack heads");

        assert_eq!(plan.output_allocation_shape, vec![42, 32, 64]);
        assert_eq!(
            plan.kernel_shape(),
            &StackHeadsKernelShape {
                input_shape: vec![42, 42],
                output_shape: vec![42, 16, 42],
                input_tile_rows: 2,
                input_tiles_per_row: 2,
                output_tile_rows: 1,
                output_tiles_per_row: 2,
                head_count: 16,
                tile_count: 42 * 2,
                flatten: false,
            }
        );
    }

    #[test]
    fn stack_heads_plan_describes_flattened_value_stack() {
        let plan =
            StackHeadsPlan::new(&[42, 128], &[42, 2048], 16, true).expect("valid stack heads");

        assert_eq!(plan.output_allocation_shape, vec![64, 2048]);
        assert_eq!(plan.kernel_shape().tile_count, 2 * 64);
        assert!(plan.kernel_shape().flatten);
    }

    #[test]
    fn broadcast_plan_supports_batched_column_broadcast() {
        let plan = BroadcastInDimPlan::new(&[18, 4, 1], &[18, 4, 32], &[0, 1, 2])
            .expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![18, 32, 32]);
        assert_eq!(plan.kernel_shape().tile_count, 18);
    }

    #[test]
    fn broadcast_plan_supports_scalar_to_rank3() {
        let plan = BroadcastInDimPlan::new(&[], &[2, 3, 4], &[]).expect("valid broadcast");

        assert_eq!(plan.output_allocation_shape, vec![2, 32, 32]);
        assert!(plan.kernel_shape().input_shape.is_empty());
        assert_eq!(plan.kernel_shape().output_shape, vec![2, 3, 4]);
        assert_eq!(plan.kernel_shape().tile_count, 2);
    }

    #[test]
    fn broadcast_plan_rejects_incompatible_mapped_dimensions() {
        let err = BroadcastInDimPlan::new(&[4], &[8, 1], &[0])
            .expect_err("incompatible broadcast should fail");

        assert!(err.to_string().contains("incompatible"));
    }

    #[test]
    fn broadcast_reader_source_uses_dummy_arrays_for_scalars() {
        let plan = BroadcastInDimPlan::new(&[], &[], &[]).expect("valid broadcast");
        let source = broadcast_reader_source(DType::Float16B, plan.kernel_shape()).expect("source");

        assert!(source.contains("#define BROADCAST_INPUT_RANK 0"));
        assert!(source.contains("#define BROADCAST_OUTPUT_RANK 0"));
        assert!(source.contains("#define BROADCAST_INPUT_SHAPE {1u}"));
        assert!(source.contains("#define BROADCAST_DIMENSIONS {1u}"));
    }

    #[test]
    fn broadcast_program_splits_tiles_across_cores() {
        let plan = BroadcastInDimPlan::new(&[5, 32, 32], &[5, 32, 32], &[0, 1, 2])
            .expect("valid broadcast");
        let program = broadcast_program(BroadcastProgramKey {
            cores: vec![
                CoreCoord { x: 1, y: 2 },
                CoreCoord { x: 1, y: 3 },
                CoreCoord { x: 1, y: 4 },
            ],
            dtype: DType::Float16B,
            shape: plan.kernel_shape().clone(),
        })
        .expect("broadcast program");

        assert_eq!(program.runtime_args.cores().len(), 3);
        assert_eq!(program.runtime_args.section_sizes(), (12, 12, 0));
        assert!(program.compute_kernel.is_empty());
        assert!(plan.kernel_shape().direct_copy);
        assert!(program
            .reader_kernel
            .contains("#define BROADCAST_OUTPUT_RANK 3"));
        assert!(program
            .reader_kernel
            .contains("#define BROADCAST_DIRECT_COPY 1"));

        let blobs = program.runtime_args.blobs();
        assert_eq!((arg_u32(&blobs[0], 1), arg_u32(&blobs[0], 2)), (0, 2));
        assert_eq!((arg_u32(&blobs[1], 1), arg_u32(&blobs[1], 2)), (2, 2));
        assert_eq!((arg_u32(&blobs[2], 1), arg_u32(&blobs[2], 2)), (4, 1));
        assert_eq!((arg_u32(&blobs[0], 4), arg_u32(&blobs[0], 5)), (0, 2));
        assert_eq!((arg_u32(&blobs[1], 4), arg_u32(&blobs[1], 5)), (2, 2));
        assert_eq!((arg_u32(&blobs[2], 4), arg_u32(&blobs[2], 5)), (4, 1));
    }
}
