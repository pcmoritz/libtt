use crate::device::Device;
use crate::dispatch::{CBConfig, CompileConfig, Program};
use crate::dram::{
    tiled_allocation_shape, tiled_shape_tile_count, DType, DramBuffer, TILE_C, TILE_R,
};
use crate::hw::CoreCoord;
use crate::kernels::kernel::{select_worker_cores, split_tile_range, Kernel, RuntimeArgsBuilder};
use std::io;

const DYNAMIC_UPDATE_SLICE_READER: &str =
    include_str!("../../kernels/dynamic_update_slice_reader.cc");
const DYNAMIC_UPDATE_SLICE_WRITER: &str = include_str!("../../kernels/broadcast_writer.cc");
const READER_OPERAND_ADDR_INDEX: usize = 0;
const READER_UPDATE_ADDR_INDEX: usize = 1;
const READER_START_INDEX_ADDR_BASE: usize = 2;
const WRITER_OUTPUT_ADDR_INDEX: usize = 0;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct DynamicUpdateSliceShape {
    operand_shape: Vec<u32>,
    update_shape: Vec<u32>,
    operand_tile_rows: u32,
    operand_tiles_per_row: u32,
    update_tile_rows: u32,
    update_tiles_per_row: u32,
    output_tiles: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DynamicUpdateSliceProgramKey {
    cores: Vec<CoreCoord>,
    dtype: DType,
    shape: DynamicUpdateSliceShape,
}

struct DynamicUpdateSliceKernel {
    operand_addr: u32,
    update_addr: u32,
    start_index_addrs: Vec<u32>,
    output_addr: u32,
    key: DynamicUpdateSliceProgramKey,
}

impl Kernel<DynamicUpdateSliceProgramKey> for DynamicUpdateSliceKernel {
    fn program_key(&self) -> DynamicUpdateSliceProgramKey {
        self.key.clone()
    }

    fn build_program(&self) -> io::Result<Program> {
        dynamic_update_slice_program(self.key.clone())
    }

    #[inline]
    fn reader_runtime_arg(&self, _core: CoreCoord, index: usize) -> Option<u32> {
        match index {
            READER_OPERAND_ADDR_INDEX => Some(self.operand_addr),
            READER_UPDATE_ADDR_INDEX => Some(self.update_addr),
            index if index >= READER_START_INDEX_ADDR_BASE => self
                .start_index_addrs
                .get(index - READER_START_INDEX_ADDR_BASE)
                .copied(),
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

pub(crate) fn dynamic_update_slice(
    device: &mut Device,
    operand: &DramBuffer,
    update: &DramBuffer,
    start_indices: &[DramBuffer],
    operand_shape: &[usize],
    update_shape: &[usize],
    start_indices_shapes: &[Vec<usize>],
    dtype: DType,
    name: impl Into<String>,
) -> io::Result<DramBuffer> {
    validate_buffers(
        operand,
        update,
        start_indices,
        operand_shape,
        update_shape,
        start_indices_shapes,
        dtype,
    )?;

    let shape = dynamic_update_slice_shape(operand_shape, update_shape)?;
    let output_tiles = usize::try_from(shape.output_tiles).map_err(|_| {
        invalid_input(format!(
            "dynamic_update_slice output tile count does not fit in usize: {}",
            shape.output_tiles
        ))
    })?;
    let cores = select_worker_cores(device.cores_ref(), output_tiles)?;
    let output = device.alloc(
        output_tiles,
        dtype,
        &tiled_allocation_shape(operand_shape)?,
        name,
    )?;
    let kernel = DynamicUpdateSliceKernel {
        operand_addr: u32_addr(operand.addr, "dynamic_update_slice operand address")?,
        update_addr: u32_addr(update.addr, "dynamic_update_slice update address")?,
        start_index_addrs: start_indices
            .iter()
            .enumerate()
            .map(|(index, start_index)| {
                u32_addr(
                    start_index.addr,
                    &format!("dynamic_update_slice start index {index} address"),
                )
            })
            .collect::<io::Result<Vec<_>>>()?,
        output_addr: u32_addr(output.addr, "dynamic_update_slice output address")?,
        key: DynamicUpdateSliceProgramKey {
            cores,
            dtype,
            shape,
        },
    };
    kernel.run(device)?;
    Ok(output)
}

fn validate_buffers(
    operand: &DramBuffer,
    update: &DramBuffer,
    start_indices: &[DramBuffer],
    operand_shape: &[usize],
    update_shape: &[usize],
    start_indices_shapes: &[Vec<usize>],
    dtype: DType,
) -> io::Result<()> {
    let rank = operand_shape.len();
    if rank != 1 {
        return Err(invalid_input(format!(
            "dynamic_update_slice currently supports rank-1 updates, got rank {rank}"
        )));
    }
    if update_shape.len() != rank {
        return Err(invalid_input(format!(
            "dynamic_update_slice update rank {} must match operand rank {rank}",
            update_shape.len()
        )));
    }
    if start_indices.len() != rank || start_indices_shapes.len() != rank {
        return Err(invalid_input(format!(
            "dynamic_update_slice requires one scalar start index per rank dimension: rank={rank}, buffers={}, shapes={}",
            start_indices.len(),
            start_indices_shapes.len()
        )));
    }
    if operand.dtype != dtype {
        return Err(invalid_input(format!(
            "dynamic_update_slice operand requires {:?}, got {:?}",
            dtype, operand.dtype
        )));
    }
    if update.dtype != dtype {
        return Err(invalid_input(format!(
            "dynamic_update_slice update requires {:?}, got {:?}",
            dtype, update.dtype
        )));
    }

    for dim in 0..rank {
        if update_shape[dim] > operand_shape[dim] {
            return Err(invalid_input(format!(
                "dynamic_update_slice update dimension {dim} size {} exceeds operand size {}",
                update_shape[dim], operand_shape[dim]
            )));
        }
    }

    validate_allocation(operand, operand_shape, "dynamic_update_slice operand")?;
    validate_allocation(update, update_shape, "dynamic_update_slice update")?;
    for (index, (start_index, start_index_shape)) in start_indices
        .iter()
        .zip(start_indices_shapes)
        .enumerate()
    {
        if start_index.dtype != DType::Int32 {
            return Err(invalid_input(format!(
                "dynamic_update_slice start index {index} requires Int32, got {:?}",
                start_index.dtype
            )));
        }
        if !start_index_shape.is_empty() {
            return Err(invalid_input(format!(
                "dynamic_update_slice start index {index} must be scalar, got {start_index_shape:?}"
            )));
        }
        validate_allocation(
            start_index,
            start_index_shape,
            &format!("dynamic_update_slice start index {index}"),
        )?;
    }
    Ok(())
}

fn validate_allocation(buffer: &DramBuffer, logical_shape: &[usize], name: &str) -> io::Result<()> {
    let expected_shape = tiled_allocation_shape(logical_shape)?;
    if buffer.shape != expected_shape {
        return Err(invalid_input(format!(
            "{name} allocation shape mismatch: got {:?}, expected {:?} for logical shape {:?}",
            buffer.shape, expected_shape, logical_shape
        )));
    }
    let expected_tiles = tiled_shape_tile_count(logical_shape)?;
    if buffer.num_tiles != expected_tiles {
        return Err(invalid_input(format!(
            "{name} tile count mismatch: got {}, expected {expected_tiles}",
            buffer.num_tiles
        )));
    }
    Ok(())
}

fn dynamic_update_slice_shape(
    operand_shape: &[usize],
    update_shape: &[usize],
) -> io::Result<DynamicUpdateSliceShape> {
    let operand_allocation_shape = tiled_allocation_shape(operand_shape)?;
    let update_allocation_shape = tiled_allocation_shape(update_shape)?;
    let operand_rank = operand_allocation_shape.len();
    let update_rank = update_allocation_shape.len();
    let output_tiles = tiled_shape_tile_count(operand_shape)?;
    Ok(DynamicUpdateSliceShape {
        operand_shape: u32_shape(operand_shape, "dynamic_update_slice operand shape")?,
        update_shape: u32_shape(update_shape, "dynamic_update_slice update shape")?,
        operand_tile_rows: u32_arg(
            operand_allocation_shape[operand_rank - 2] / TILE_R,
            "dynamic_update_slice operand tile rows",
        )?,
        operand_tiles_per_row: u32_arg(
            operand_allocation_shape[operand_rank - 1] / TILE_C,
            "dynamic_update_slice operand tiles per row",
        )?,
        update_tile_rows: u32_arg(
            update_allocation_shape[update_rank - 2] / TILE_R,
            "dynamic_update_slice update tile rows",
        )?,
        update_tiles_per_row: u32_arg(
            update_allocation_shape[update_rank - 1] / TILE_C,
            "dynamic_update_slice update tiles per row",
        )?,
        output_tiles: u32_arg(output_tiles, "dynamic_update_slice output tile count")?,
    })
}

fn dynamic_update_slice_program(key: DynamicUpdateSliceProgramKey) -> io::Result<Program> {
    let rank = key.shape.operand_shape.len();
    let mut reader_dynamic_indices = vec![READER_OPERAND_ADDR_INDEX, READER_UPDATE_ADDR_INDEX];
    reader_dynamic_indices.extend(READER_START_INDEX_ADDR_BASE..READER_START_INDEX_ADDR_BASE + rank);
    let mut runtime_args = RuntimeArgsBuilder::new(
        0,
        vec![WRITER_OUTPUT_ADDR_INDEX],
        reader_dynamic_indices,
        Vec::new(),
    );
    for (core_index, &core) in key.cores.iter().enumerate() {
        let (offset, n_tiles) =
            split_tile_range(key.shape.output_tiles, core_index, key.cores.len())?;
        let mut reader_args = vec![0, 0];
        reader_args.extend(std::iter::repeat(0).take(rank));
        reader_args.push(offset);
        reader_args.push(n_tiles);
        runtime_args.add_core(core, vec![0, offset, n_tiles], reader_args, Vec::new())?;
    }
    let runtime_args = runtime_args.build()?;
    Ok(Program {
        reader_kernel: dynamic_update_slice_reader_source(key.dtype, &key.shape)?,
        writer_kernel: DYNAMIC_UPDATE_SLICE_WRITER.to_owned(),
        compile: CompileConfig {
            cbs: vec![
                CBConfig::new(0, key.dtype),
                CBConfig::new(1, DType::Int32),
                CBConfig::new(2, key.dtype),
                CBConfig::new(16, key.dtype),
            ],
            ..CompileConfig::default()
        },
        name: format!(
            "dynamic_update_slice_{:?}_{}",
            key.dtype,
            key.shape.operand_shape.len()
        ),
        ..Program::new(runtime_args)
    })
}

fn dynamic_update_slice_reader_source(
    dtype: DType,
    shape: &DynamicUpdateSliceShape,
) -> io::Result<String> {
    let element_type = element_type(dtype);
    Ok(format!(
        "#define DUS_RANK {}\n\
         #define DUS_OPERAND_SHAPE {}\n\
         #define DUS_UPDATE_SHAPE {}\n\
         #define DUS_OPERAND_TILE_ROWS {}\n\
         #define DUS_OPERAND_TILES_PER_ROW {}\n\
         #define DUS_UPDATE_TILE_ROWS {}\n\
         #define DUS_UPDATE_TILES_PER_ROW {}\n\
         #define DUS_ELEMENT_TYPE {element_type}\n\
         {DYNAMIC_UPDATE_SLICE_READER}",
        shape.operand_shape.len(),
        cpp_u32_array(&shape.operand_shape),
        cpp_u32_array(&shape.update_shape),
        shape.operand_tile_rows,
        shape.operand_tiles_per_row,
        shape.update_tile_rows,
        shape.update_tiles_per_row,
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
    fn program_places_start_index_addresses_before_tile_range() {
        let shape = dynamic_update_slice_shape(&[64], &[2]).expect("shape");
        let program = dynamic_update_slice_program(DynamicUpdateSliceProgramKey {
            cores: vec![CoreCoord { x: 1, y: 2 }],
            dtype: DType::Int32,
            shape,
        })
        .expect("program");

        assert_eq!(program.runtime_args.section_sizes(), (12, 20, 0));
        assert!(program
            .reader_kernel
            .contains("#define DUS_OPERAND_SHAPE {64u}"));
        let blobs = program.runtime_args.blobs();
        assert_eq!(arg_u32(&blobs[0], 6), 0);
        assert_eq!(arg_u32(&blobs[0], 7), 2);
    }
}
