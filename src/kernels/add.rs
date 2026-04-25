use crate::dispatch::{CBConfig, CoreSelection, Program};
use crate::dram::DType;

const BF16_READER: &str = include_str!("../../kernels/add_reader.cc");
const BF16_WRITER: &str = include_str!("../../kernels/add_writer.cc");
const BF16_COMPUTE: &str = include_str!("../../kernels/add_compute.cc");

pub(crate) fn bf16_program(
    lhs_addr: u32,
    rhs_addr: u32,
    output_addr: u32,
    tile_count: u32,
) -> Program {
    Program {
        cores: CoreSelection::Count(1),
        reader_kernel: BF16_READER.to_owned(),
        compute_kernel: BF16_COMPUTE.to_owned(),
        writer_kernel: BF16_WRITER.to_owned(),
        cbs: vec![
            CBConfig {
                index: 0,
                dtype: DType::Float16B,
                tiles: 2,
            },
            CBConfig {
                index: 1,
                dtype: DType::Float16B,
                tiles: 2,
            },
            CBConfig {
                index: 16,
                dtype: DType::Float16B,
                tiles: 2,
            },
        ],
        name: "eltwise_add_bf16".to_owned(),
        reader_args: vec![lhs_addr, rhs_addr, 0, tile_count],
        writer_args: vec![output_addr, 0, tile_count],
        compute_args: vec![tile_count],
        ..Program::default()
    }
}
