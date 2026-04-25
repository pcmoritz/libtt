use crate::device::Device;
use crate::dispatch::{CBConfig, CoreSelection, Program};
use crate::dram::{DType, DramBuffer};
use std::io;

const BF16_READER: &str = include_str!("../../kernels/add_reader.cc");
const BF16_WRITER: &str = include_str!("../../kernels/add_writer.cc");
const BF16_COMPUTE: &str = include_str!("../../kernels/add_compute.cc");

pub(crate) fn eltwise_add_bf16(
    device: &mut Device,
    lhs: &DramBuffer,
    rhs: &DramBuffer,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    if lhs.dtype != DType::Float16B || rhs.dtype != DType::Float16B {
        return Err(invalid_input(format!(
            "eltwise_add_bf16 requires bf16 inputs, got {:?} and {:?}",
            lhs.dtype, rhs.dtype
        )));
    }
    if lhs.num_tiles != rhs.num_tiles {
        return Err(invalid_input(format!(
            "input tile counts must match, got {} and {}",
            lhs.num_tiles, rhs.num_tiles
        )));
    }
    if lhs.shape != rhs.shape {
        return Err(invalid_input(format!(
            "input shapes must match, got {:?} and {:?}",
            lhs.shape, rhs.shape
        )));
    }

    let lhs_addr = u32_arg(lhs.addr, "lhs address")?;
    let rhs_addr = u32_arg(rhs.addr, "rhs address")?;
    let tile_count = u32::try_from(lhs.num_tiles)
        .map_err(|_| invalid_input(format!("tile count does not fit in u32: {}", lhs.num_tiles)))?;
    let output = device.alloc(lhs.num_tiles, DType::Float16B, lhs.shape.as_deref(), name)?;
    let output_addr = u32_arg(output.addr, "output address")?;

    let program = bf16_program(lhs_addr, rhs_addr, output_addr, tile_count);
    device.run_program(&program)?;
    Ok(output)
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn u32_arg(value: u64, name: &str) -> io::Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid_input(format!("{name} does not fit in u32: 0x{value:x}")))
}

fn bf16_program(lhs_addr: u32, rhs_addr: u32, output_addr: u32, tile_count: u32) -> Program {
    Program {
        cores: CoreSelection::Count(1),
        reader_kernel: BF16_READER.to_owned(),
        compute_kernel: BF16_COMPUTE.to_owned(),
        writer_kernel: BF16_WRITER.to_owned(),
        cbs: vec![
            CBConfig::new(0, DType::Float16B),
            CBConfig::new(1, DType::Float16B),
            CBConfig::new(16, DType::Float16B),
        ],
        name: "eltwise_add_bf16".to_owned(),
        reader_args: vec![lhs_addr, rhs_addr, 0, tile_count],
        writer_args: vec![output_addr, 0, tile_count],
        compute_args: vec![tile_count],
        ..Program::default()
    }
}
