use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer};
use crate::executable::BitwiseBinaryKind;
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const BITWISE_READER: &str = include_str!("../../kernels/bitwise_binary_reader.cc");
const BITWISE_WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_LHS_ADDR_INDEX: usize = 0;
const READER_RHS_ADDR_INDEX: usize = 1;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct BitwiseProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    kind: BitwiseBinaryKind,
    tile_count: u32,
}

struct BitwiseKernel {
    lhs_addr: u32,
    rhs_addr: u32,
    output_addr: u32,
    key: BitwiseProgramKey,
}

impl Kernel<BitwiseProgramKey> for BitwiseKernel {
    fn program_key(&self) -> BitwiseProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        bitwise_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_LHS_ADDR_INDEX => Some(self.lhs_addr),
            READER_RHS_ADDR_INDEX => Some(self.rhs_addr),
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

pub(crate) fn bitwise_binary(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    shape: &[usize],
    dtype: DType,
    kind: BitwiseBinaryKind,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate(lhs, rhs, shape, dtype)?;
    let tile_count = tiled_shape_tile_count(shape)?;
    let tile_count_u32 = u32_arg(tile_count, "bitwise tile count")?;
    let cores = select_worker_cores(device.cores_ref(), tile_count)?;
    let output = device.alloc(tile_count, dtype, &tiled_allocation_shape(shape)?, name)?;
    let kernel = BitwiseKernel {
        lhs_addr: u32_addr(lhs.addr, "bitwise lhs address")?,
        rhs_addr: u32_addr(rhs.addr, "bitwise rhs address")?,
        output_addr: u32_addr(output.addr, "bitwise output address")?,
        key: BitwiseProgramKey {
            cores,
            dtype,
            kind,
            tile_count: tile_count_u32,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate(lhs: &DramBuffer, rhs: &DramBuffer, shape: &[usize], dtype: DType) -> io::Result<()> {
    if !matches!(
        dtype,
        DType::Int32 | DType::UInt32 | DType::UInt16 | DType::UInt8
    ) {
        return Err(invalid_input(format!(
            "bitwise binary does not support dtype {dtype:?}"
        )));
    }
    if lhs.dtype != dtype || rhs.dtype != dtype {
        return Err(invalid_input(format!(
            "bitwise inputs require {:?}, got {:?} and {:?}",
            dtype, lhs.dtype, rhs.dtype
        )));
    }
    let allocation_shape = tiled_allocation_shape(shape)?;
    if lhs.shape != allocation_shape || rhs.shape != allocation_shape {
        return Err(invalid_input(format!(
            "bitwise allocation shape mismatch: expected {:?}, got lhs {:?}, rhs {:?}",
            allocation_shape, lhs.shape, rhs.shape
        )));
    }
    let tile_count = tiled_shape_tile_count(shape)?;
    if lhs.num_tiles != tile_count || rhs.num_tiles != tile_count {
        return Err(invalid_input(format!(
            "bitwise tile count mismatch: expected {tile_count}, got lhs {}, rhs {}",
            lhs.num_tiles, rhs.num_tiles
        )));
    }
    Ok(())
}

fn bitwise_program(key: BitwiseProgramKey) -> io::Result<Program> {
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        vec![READER_LHS_ADDR_INDEX, READER_RHS_ADDR_INDEX],
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) = split_tile_range(key.tile_count, core_index, key.cores.len())?;
        runtime_args.add_core(
            core,
            vec![0, offset, n_tiles],
            vec![0, 0, offset, n_tiles],
            Vec::new(),
        )?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: bitwise_reader_source(key.kind, key.dtype)?,
        writer_kernel: BITWISE_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.dtype),
                CBConfig::new(1, key.dtype),
                CBConfig::new(16, key.dtype),
            ],
            ..CompileConfig::default()
        },
        name: format!("bitwise_{:?}_{:?}", key.kind, key.dtype),
        ..Program::new(runtime_args)
    })
}

fn bitwise_reader_source(kind: BitwiseBinaryKind, dtype: DType) -> io::Result<String> {
    Ok(format!(
        "#define BITWISE_OP {}\n\
         #define BITWISE_ELEMENT_TYPE {}\n\
         #define BITWISE_SIGNED_ELEMENT_TYPE {}\n\
         #define BITWISE_UNSIGNED_ELEMENT_TYPE {}\n\
         #define BITWISE_BIT_WIDTH {}\n\
         {BITWISE_READER}",
        bitwise_op_value(kind),
        element_type(dtype),
        signed_element_type(dtype),
        unsigned_element_type(dtype),
        bit_width(dtype),
    ))
}

fn bitwise_op_value(kind: BitwiseBinaryKind) -> u32 {
    match kind {
        BitwiseBinaryKind::And => 0,
        BitwiseBinaryKind::Or => 1,
        BitwiseBinaryKind::Xor => 2,
        BitwiseBinaryKind::ShiftLeft => 3,
        BitwiseBinaryKind::ShiftRightLogical => 4,
        BitwiseBinaryKind::ShiftRightArithmetic => 5,
    }
}

fn element_type(dtype: DType) -> &'static str {
    match dtype {
        DType::Int32 | DType::UInt32 => "uint32_t",
        DType::UInt16 => "uint16_t",
        DType::UInt8 => "uint8_t",
        _ => "uint32_t",
    }
}

fn signed_element_type(dtype: DType) -> &'static str {
    match dtype {
        DType::UInt16 => "int16_t",
        DType::UInt8 => "int8_t",
        _ => "int32_t",
    }
}

fn unsigned_element_type(dtype: DType) -> &'static str {
    match dtype {
        DType::UInt16 => "uint16_t",
        DType::UInt8 => "uint8_t",
        _ => "uint32_t",
    }
}

fn bit_width(dtype: DType) -> u32 {
    match dtype {
        DType::UInt16 => 16,
        DType::UInt8 => 8,
        _ => 32,
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
