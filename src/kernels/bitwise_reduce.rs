use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::executable::ReduceReducer;
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const READER: &str = include_str!("../../kernels/bitwise_reduce_lastdim_reader.cc");
const WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_INPUT_ADDR_INDEX: usize = 0;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum BitwiseReduceOp {
    And,
    Or,
}

impl BitwiseReduceOp {
    fn from_reducer(reducer: ReduceReducer) -> io::Result<Self> {
        match reducer {
            ReduceReducer::And => Ok(Self::And),
            ReduceReducer::Or => Ok(Self::Or),
            _ => Err(invalid_input(format!(
                "bitwise reduce supports only and/or reducers, got {reducer:?}"
            ))),
        }
    }

    fn op_value(self) -> u32 {
        match self {
            Self::And => 0,
            Self::Or => 1,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct BitwiseReduceShape {
    input_shape: Vec<u32>,
    output_shape: Vec<u32>,
    input_tile_rows: u32,
    input_tiles_per_row: u32,
    output_tile_rows: u32,
    output_tiles_per_row: u32,
    output_tiles: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BitwiseReducePlan {
    input_shape: Vec<usize>,
    output_allocation_shape: Vec<usize>,
    shape: BitwiseReduceShape,
    op: BitwiseReduceOp,
    dtype: DType,
    identity: u32,
}

impl BitwiseReducePlan {
    pub(crate) fn new(
        dtype: DType,
        input_shape: &[usize],
        output_shape: &[usize],
        dimensions: &[i64],
        reducer: ReduceReducer,
        identity: u32,
    ) -> io::Result<Self> {
        if !matches!(
            dtype,
            DType::Int32 | DType::UInt32 | DType::UInt16 | DType::UInt8
        ) {
            return Err(invalid_input(format!(
                "bitwise reduce does not support dtype {dtype:?}"
            )));
        }
        if input_shape.len() < 2 {
            return Err(invalid_input(format!(
                "bitwise reduce requires rank >= 2 input, got {input_shape:?}"
            )));
        }
        let reduce_dim = input_shape.len() - 1;
        if dimensions != [reduce_dim as i64] {
            return Err(invalid_input(format!(
                "bitwise reduce currently supports only the last dimension, got dimensions {dimensions:?} for shape {input_shape:?}"
            )));
        }
        let expected_output = &input_shape[..input_shape.len() - 1];
        if output_shape != expected_output {
            return Err(invalid_input(format!(
                "bitwise reduce output shape mismatch: expected {:?}, got {:?}",
                expected_output, output_shape
            )));
        }

        let output_allocation_shape = tiled_allocation_shape(output_shape)?;
        let shape = bitwise_reduce_shape(input_shape, output_shape)?;
        Ok(Self {
            input_shape: input_shape.to_vec(),
            output_allocation_shape,
            shape,
            op: BitwiseReduceOp::from_reducer(reducer)?,
            dtype,
            identity,
        })
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct BitwiseReduceProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    op: BitwiseReduceOp,
    identity: u32,
    shape: BitwiseReduceShape,
}

struct BitwiseReduceKernel {
    input_addr: u32,
    output_addr: u32,
    key: BitwiseReduceProgramKey,
}

impl Kernel<BitwiseReduceProgramKey> for BitwiseReduceKernel {
    fn program_key(&self) -> BitwiseReduceProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        bitwise_reduce_program(self.key.clone())
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

pub(crate) fn reduce(
    device: &mut Device,
    input: &DramBuffer,
    plan: &BitwiseReducePlan,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_input(input, plan)?;
    let output_tiles = usize::try_from(plan.shape.output_tiles).map_err(|_| {
        invalid_input(format!(
            "bitwise reduce output tile count does not fit in usize: {}",
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
    let kernel = BitwiseReduceKernel {
        input_addr: u32_addr(input.addr, "bitwise reduce input address")?,
        output_addr: u32_addr(output.addr, "bitwise reduce output address")?,
        key: BitwiseReduceProgramKey {
            cores,
            dtype: plan.dtype,
            op: plan.op,
            identity: plan.identity,
            shape: plan.shape.clone(),
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_input(input: &DramBuffer, plan: &BitwiseReducePlan) -> io::Result<()> {
    if input.dtype != plan.dtype {
        return Err(invalid_input(format!(
            "bitwise reduce input requires {:?}, got {:?}",
            plan.dtype, input.dtype
        )));
    }
    let expected_shape = tiled_allocation_shape(&plan.input_shape)?;
    if input.shape != expected_shape {
        return Err(invalid_input(format!(
            "bitwise reduce input allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            input.shape, expected_shape, plan.input_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(&plan.input_shape)?;
    if input.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "bitwise reduce input tile count mismatch: got {}, expected {expected_tiles}",
            input.num_tiles
        )));
    }
    Ok(())
}

fn bitwise_reduce_shape(
    input_shape: &[usize],
    output_shape: &[usize],
) -> io::Result<BitwiseReduceShape> {
    let input_allocation_shape = tiled_allocation_shape(input_shape)?;
    let output_allocation_shape = tiled_allocation_shape(output_shape)?;
    let input_rank = input_allocation_shape.len();
    let output_rank = output_allocation_shape.len();
    Ok(BitwiseReduceShape {
        input_shape: u32_shape(input_shape, "bitwise reduce input shape")?,
        output_shape: u32_shape(output_shape, "bitwise reduce output shape")?,
        input_tile_rows: u32_arg(
            input_allocation_shape[input_rank - 2] / TILE_R,
            "bitwise reduce input tile rows",
        )?,
        input_tiles_per_row: u32_arg(
            input_allocation_shape[input_rank - 1] / TILE_C,
            "bitwise reduce input tiles per row",
        )?,
        output_tile_rows: u32_arg(
            output_allocation_shape[output_rank - 2] / TILE_R,
            "bitwise reduce output tile rows",
        )?,
        output_tiles_per_row: u32_arg(
            output_allocation_shape[output_rank - 1] / TILE_C,
            "bitwise reduce output tiles per row",
        )?,
        output_tiles: u32_arg(
            tiled_shape_tile_count(output_shape)?,
            "bitwise reduce output tile count",
        )?,
    })
}

fn bitwise_reduce_program(key: BitwiseReduceProgramKey) -> io::Result<Program> {
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
        reader_kernel: bitwise_reduce_reader_source(&key)?,
        writer_kernel: WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![CBConfig::new(0, key.dtype), CBConfig::new(16, key.dtype)],
            ..CompileConfig::default()
        },
        name: format!("bitwise_reduce_{:?}_{:?}", key.op, key.dtype),
        ..Program::new(runtime_args)
    })
}

fn bitwise_reduce_reader_source(key: &BitwiseReduceProgramKey) -> io::Result<String> {
    Ok(format!(
        "#define BITWISE_REDUCE_RANK {}\n\
         #define BITWISE_REDUCE_INPUT_SHAPE {}\n\
         #define BITWISE_REDUCE_OUTPUT_SHAPE {}\n\
         #define BITWISE_REDUCE_INPUT_TILE_ROWS {}\n\
         #define BITWISE_REDUCE_INPUT_TILES_PER_ROW {}\n\
         #define BITWISE_REDUCE_OUTPUT_TILE_ROWS {}\n\
         #define BITWISE_REDUCE_OUTPUT_TILES_PER_ROW {}\n\
         #define BITWISE_REDUCE_OP {}\n\
         #define BITWISE_REDUCE_IDENTITY {}\n\
         #define BITWISE_REDUCE_ELEMENT_TYPE {}\n\
         {READER}",
        key.shape.input_shape.len(),
        cpp_u32_array(&key.shape.input_shape),
        cpp_u32_array(&key.shape.output_shape),
        key.shape.input_tile_rows,
        key.shape.input_tiles_per_row,
        key.shape.output_tile_rows,
        key.shape.output_tiles_per_row,
        key.op.op_value(),
        key.identity,
        element_type(key.dtype),
    ))
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
        DType::Int32 | DType::UInt32 => "uint32_t",
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
