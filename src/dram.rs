use crate::device::{load_device, Device};
use crate::hw::{align_up, CoreCoord, Dram, DramTile};
use crate::linux::{NocOrdering, TlbWindow};
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) const TILE_R: usize = 32;
pub(crate) const TILE_C: usize = 32;
const FACE_R: usize = 16;
const FACE_C: usize = 16;
type Shape = Vec<usize>;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum DType {
    Float32 = 0,
    Float16 = 1,
    Float16B = 5,
    Int32 = 8,
    UInt16 = 9,
    Int8 = 14,
    UInt32 = 24,
    UInt8 = 30,
}

impl DType {
    pub(crate) fn bytes_per_element(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 | Self::UInt32 => 4,
            Self::Float16 | Self::Float16B | Self::UInt16 => 2,
            Self::Int8 | Self::UInt8 => 1,
        }
    }

    pub(crate) fn tile_size(self) -> usize {
        TILE_R * TILE_C * self.bytes_per_element()
    }
}

pub(crate) fn tiled_allocation_shape(shape: &[usize]) -> io::Result<Vec<usize>> {
    match shape.len() {
        0 => Ok(vec![TILE_R, TILE_C]),
        1 => Ok(vec![TILE_R, round_up_to_tile_dim(shape[0], TILE_C)?]),
        _ => {
            let mut allocation_shape = shape.to_vec();
            let rank = allocation_shape.len();
            allocation_shape[rank - 2] = round_up_to_tile_dim(allocation_shape[rank - 2], TILE_R)?;
            allocation_shape[rank - 1] = round_up_to_tile_dim(allocation_shape[rank - 1], TILE_C)?;
            Ok(allocation_shape)
        }
    }
}

pub(crate) fn tiled_shape_tile_count(shape: &[usize]) -> io::Result<usize> {
    let allocation_shape = tiled_allocation_shape(shape)?;
    let rows = allocation_shape[allocation_shape.len() - 2];
    let cols = allocation_shape[allocation_shape.len() - 1];
    let tiles_per_batch = (rows / TILE_R)
        .checked_mul(cols / TILE_C)
        .ok_or_else(|| invalid_input("shape tile count is too large"))?;
    allocation_shape[..allocation_shape.len() - 2]
        .iter()
        .try_fold(tiles_per_batch, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| invalid_input("shape tile count is too large"))
}

fn round_up_to_tile_dim(value: usize, tile_dim: usize) -> io::Result<usize> {
    value
        .max(1)
        .checked_next_multiple_of(tile_dim)
        .ok_or_else(|| invalid_input("shape dimension overflow"))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[derive(Clone, Debug)]
pub struct DramBuffer {
    pub name: String,
    pub addr: u64,
    pub num_tiles: usize,
    pub dtype: DType,
    /// Physical allocation shape. The last two dimensions are tile-aligned.
    pub shape: Shape,
    _allocation: Option<Arc<DramAllocation>>,
}

impl PartialEq for DramBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.addr == other.addr
            && self.num_tiles == other.num_tiles
            && self.dtype == other.dtype
            && self.shape == other.shape
    }
}

impl Eq for DramBuffer {}

impl DramBuffer {
    pub(crate) fn page_size(&self) -> usize {
        self.dtype.tile_size()
    }

    pub(crate) fn size(&self) -> usize {
        self.num_tiles * self.page_size()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct DramAllocation {
    local_hardware_id: usize,
    addr: u64,
    size: u64,
}

impl Drop for DramAllocation {
    fn drop(&mut self) {
        free_allocation(self.local_hardware_id, self.addr, self.size);
    }
}

pub struct Allocator {
    window: TlbWindow,
    bank_tiles: Vec<DramTile>,
    local_hardware_id: usize,
    bank_count: usize,
}

#[derive(Debug)]
struct DeviceAllocatorState {
    next: u64,
    free: Vec<FreeBlock>,
}

impl DeviceAllocatorState {
    fn new() -> Self {
        Self {
            next: Dram::WRITE_OFFSET,
            free: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreeBlock {
    addr: u64,
    size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllocatorStats {
    pub(crate) bytes_in_use: u64,
    pub(crate) bytes_limit: u64,
    pub(crate) largest_free_block_bytes: u64,
}

static ALLOCATOR_STATE_BY_DEVICE: OnceLock<Mutex<HashMap<usize, DeviceAllocatorState>>> =
    OnceLock::new();

impl Allocator {
    pub fn open(local_hardware_id: usize) -> io::Result<Self> {
        let (path, info) = load_device(local_hardware_id)?;
        Self::from_device_with_path(path, &info)
    }

    pub(crate) fn from_device(device: &Device) -> io::Result<Self> {
        Self::from_device_with_path(device.path.clone(), device)
    }

    pub(crate) fn from_existing_device(
        path: PathBuf,
        local_hardware_id: usize,
        dram_tiles: &[DramTile],
    ) -> io::Result<Self> {
        Self::from_parts(path, local_hardware_id, dram_tiles)
    }

    fn from_device_with_path(path: PathBuf, device: &Device) -> io::Result<Self> {
        Self::from_parts(path, device.local_hardware_id, &device.dram_tiles)
    }

    fn from_parts(
        path: PathBuf,
        local_hardware_id: usize,
        dram_tiles: &[DramTile],
    ) -> io::Result<Self> {
        let bank_tiles = dram_tiles
            .iter()
            .step_by(Dram::TILES_PER_BANK)
            .copied()
            .collect::<Vec<_>>();
        if bank_tiles.is_empty() {
            return Err(io::Error::other("no active DRAM bank tiles discovered"));
        }
        let first = bank_tiles
            .first()
            .copied()
            .ok_or_else(|| io::Error::other("no active DRAM bank tiles discovered"))?;
        let bank_count = bank_tiles.len();
        let mut window = TlbWindow::open(path.as_path(), Dram::TLB_SIZE_4G, true)?;
        window.target(
            CoreCoord {
                x: first.x,
                y: first.y,
            },
            None,
            0,
            NocOrdering::Strict,
        )?;
        Ok(Self {
            window,
            bank_tiles,
            local_hardware_id,
            bank_count,
        })
    }

    pub fn alloc(
        &mut self,
        num_tiles: usize,
        dtype: DType,
        name: impl Into<String>,
        shape: Shape,
    ) -> io::Result<DramBuffer> {
        validate_allocation_shape(num_tiles, &shape)?;
        let (addr, allocation_size, _next) =
            allocate_allocation_range(self.local_hardware_id, num_tiles, dtype, self.bank_count)?;
        Ok(DramBuffer {
            name: name.into(),
            addr,
            num_tiles,
            dtype,
            shape,
            _allocation: (allocation_size > 0).then(|| {
                Arc::new(DramAllocation {
                    local_hardware_id: self.local_hardware_id,
                    addr,
                    size: allocation_size,
                })
            }),
        })
    }

    pub fn alloc_write(
        &mut self,
        data: &[u8],
        dtype: DType,
        shape: Shape,
        name: impl Into<String>,
    ) -> io::Result<DramBuffer> {
        validate_tile_multiple(data.len(), dtype)?;
        let buf = self.alloc(data.len() / dtype.tile_size(), dtype, name, shape)?;
        self.write(&buf, data)?;
        Ok(buf)
    }

    pub(crate) fn alloc_for_host_data(
        &mut self,
        data: &[u8],
        dtype: DType,
        shape: Shape,
        name: impl Into<String>,
    ) -> io::Result<DramBuffer> {
        validate_tiled_shape(data, dtype, &shape)?;
        let num_tiles = data.len() / dtype.tile_size();
        self.alloc(num_tiles, dtype, name, shape)
    }

    pub fn write(&mut self, buf: &DramBuffer, data: &[u8]) -> io::Result<()> {
        if data.len() > buf.size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "buffer write exceeds allocation: {} > {}",
                    data.len(),
                    buf.size()
                ),
            ));
        }
        let page_count = data.len().div_ceil(buf.page_size());

        for (bank_index, tile) in self.bank_tiles.iter().enumerate() {
            let bank_data =
                collect_bank_data(data, buf.page_size(), bank_index, self.bank_tiles.len());
            if bank_data.is_empty() {
                continue;
            }

            self.window.target(
                CoreCoord {
                    x: tile.x,
                    y: tile.y,
                },
                None,
                0,
                NocOrdering::Posted,
            )?;
            self.window.write(buf.addr as usize, &bank_data)?;
        }

        if page_count > 0 {
            self.barrier()?;
        }
        Ok(())
    }

    pub(crate) fn write_host_data(&mut self, buf: &DramBuffer, data: &[u8]) -> io::Result<()> {
        let payload = tilize(data, buf.dtype, &buf.shape)?;
        self.write(buf, &payload)
    }

    pub fn read(&mut self, buf: &DramBuffer) -> io::Result<Vec<u8>> {
        let mut result = vec![0u8; buf.size()];
        let page_count = buf.size().div_ceil(buf.page_size());

        for (bank_index, tile) in self.bank_tiles.iter().enumerate() {
            let bank_pages = (bank_index..page_count)
                .step_by(self.bank_tiles.len())
                .count();
            if bank_pages == 0 {
                continue;
            }

            self.window.target(
                CoreCoord {
                    x: tile.x,
                    y: tile.y,
                },
                None,
                0,
                NocOrdering::Relaxed,
            )?;
            let bank_data = self
                .window
                .read(buf.addr as usize, bank_pages * buf.page_size())?;
            scatter_bank_data(
                &mut result,
                buf.page_size(),
                bank_index,
                self.bank_tiles.len(),
                &bank_data,
            );
        }

        Ok(result)
    }

    pub(crate) fn read_host_data(&mut self, buf: &DramBuffer) -> io::Result<Vec<u8>> {
        let payload = self.read(buf)?;
        untilize(&payload, buf.dtype, &buf.shape)
    }

    fn barrier(&mut self) -> io::Result<()> {
        for flag in Dram::BARRIER_FLAGS {
            for tile in &self.bank_tiles {
                self.window.target(
                    CoreCoord {
                        x: tile.x,
                        y: tile.y,
                    },
                    None,
                    0,
                    NocOrdering::Strict,
                )?;
                self.window.write32(Dram::BARRIER_BASE, flag)?;
                while self.window.read32(Dram::BARRIER_BASE)? != flag {}
            }
        }
        Ok(())
    }
}

fn allocate_allocation_range(
    local_hardware_id: usize,
    num_tiles: usize,
    dtype: DType,
    bank_count: usize,
) -> io::Result<(u64, u64, u64)> {
    let allocation_size = allocation_range_size(num_tiles, dtype, bank_count)?;
    let state = ALLOCATOR_STATE_BY_DEVICE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut state = state.lock().expect("allocator state lock poisoned");
    let device_state = state
        .entry(local_hardware_id)
        .or_insert_with(DeviceAllocatorState::new);
    if allocation_size == 0 {
        return Ok((device_state.next, allocation_size, device_state.next));
    }
    if let Some(addr) = allocate_free_block(&mut device_state.free, allocation_size) {
        return Ok((addr, allocation_size, device_state.next));
    }

    let (addr, next) = next_allocation_range(device_state.next, num_tiles, dtype, bank_count)?;
    device_state.next = next;
    Ok((addr, allocation_size, next))
}

fn allocate_free_block(free: &mut Vec<FreeBlock>, size: u64) -> Option<u64> {
    let index = free
        .iter()
        .enumerate()
        .filter(|(_, block)| block.size >= size)
        .min_by_key(|(_, block)| block.size)
        .map(|(index, _)| index)?;
    let addr = free[index].addr;
    if free[index].size == size {
        free.remove(index);
    } else {
        free[index].addr += size;
        free[index].size -= size;
    }
    Some(addr)
}

fn free_allocation(local_hardware_id: usize, addr: u64, size: u64) {
    if size == 0 {
        return;
    }
    let state = ALLOCATOR_STATE_BY_DEVICE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut state) = state.lock() {
        let device_state = state
            .entry(local_hardware_id)
            .or_insert_with(DeviceAllocatorState::new);
        insert_free_block(&mut device_state.free, FreeBlock { addr, size });
    }
}

pub(crate) fn allocator_stats(local_hardware_id: usize, bank_count: usize) -> AllocatorStats {
    let bank_count = bank_count as u64;
    let tail_bytes_per_bank = Dram::TLB_SIZE_4G.saturating_sub(Dram::WRITE_OFFSET);
    let bytes_limit = tail_bytes_per_bank.saturating_mul(bank_count);
    let default_free = tail_bytes_per_bank.saturating_mul(bank_count);
    let Some(state) = ALLOCATOR_STATE_BY_DEVICE.get() else {
        return AllocatorStats {
            bytes_in_use: 0,
            bytes_limit,
            largest_free_block_bytes: default_free,
        };
    };
    let Ok(state) = state.lock() else {
        return AllocatorStats {
            bytes_in_use: 0,
            bytes_limit,
            largest_free_block_bytes: default_free,
        };
    };
    let Some(device_state) = state.get(&local_hardware_id) else {
        return AllocatorStats {
            bytes_in_use: 0,
            bytes_limit,
            largest_free_block_bytes: default_free,
        };
    };

    let allocated_span = device_state.next.saturating_sub(Dram::WRITE_OFFSET);
    let free_bytes = device_state
        .free
        .iter()
        .map(|block| block.size)
        .fold(0u64, u64::saturating_add);
    let tail_free = Dram::TLB_SIZE_4G.saturating_sub(device_state.next);
    let largest_free_per_bank = device_state
        .free
        .iter()
        .map(|block| block.size)
        .chain(std::iter::once(tail_free))
        .max()
        .unwrap_or(0);

    AllocatorStats {
        bytes_in_use: allocated_span
            .saturating_sub(free_bytes)
            .saturating_mul(bank_count),
        bytes_limit,
        largest_free_block_bytes: largest_free_per_bank.saturating_mul(bank_count),
    }
}

fn insert_free_block(free: &mut Vec<FreeBlock>, block: FreeBlock) {
    if block.size == 0 {
        return;
    }
    free.push(block);
    free.sort_unstable_by_key(|block| block.addr);

    let mut coalesced = Vec::<FreeBlock>::with_capacity(free.len());
    for block in free.drain(..) {
        if let Some(last) = coalesced.last_mut() {
            let last_end = last.addr.saturating_add(last.size);
            if last_end >= block.addr {
                let block_end = block.addr.saturating_add(block.size);
                last.size = block_end.max(last_end).saturating_sub(last.addr);
                continue;
            }
        }
        coalesced.push(block);
    }
    *free = coalesced;
}

pub(crate) fn tilize(data: &[u8], dtype: DType, shape: &[usize]) -> io::Result<Vec<u8>> {
    let Layout {
        batch,
        rows,
        cols,
        bytes_per_element,
    } = Layout::new(data, dtype, shape)?;

    let mut out = Vec::with_capacity(data.len());
    let tiles_per_row = cols / TILE_C;
    let tile_rows = rows / TILE_R;

    for batch_index in 0..batch {
        for tile_row in 0..tile_rows {
            for tile_col in 0..tiles_per_row {
                for face_row in 0..2 {
                    for face_col in 0..2 {
                        for row in 0..FACE_R {
                            for col in 0..FACE_C {
                                let source_index = element_offset(
                                    batch_index,
                                    tile_row * TILE_R + face_row * FACE_R + row,
                                    tile_col * TILE_C + face_col * FACE_C + col,
                                    rows,
                                    cols,
                                    bytes_per_element,
                                );
                                out.extend_from_slice(
                                    &data[source_index..source_index + bytes_per_element],
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(out)
}

pub(crate) fn untilize(data: &[u8], dtype: DType, shape: &[usize]) -> io::Result<Vec<u8>> {
    let Layout {
        batch,
        rows,
        cols,
        bytes_per_element,
    } = Layout::new(data, dtype, shape)?;

    let mut out = vec![0u8; data.len()];
    let tiles_per_row = cols / TILE_C;
    let tile_rows = rows / TILE_R;
    let mut cursor = 0usize;

    for batch_index in 0..batch {
        for tile_row in 0..tile_rows {
            for tile_col in 0..tiles_per_row {
                for face_row in 0..2 {
                    for face_col in 0..2 {
                        for row in 0..FACE_R {
                            for col in 0..FACE_C {
                                let target_index = element_offset(
                                    batch_index,
                                    tile_row * TILE_R + face_row * FACE_R + row,
                                    tile_col * TILE_C + face_col * FACE_C + col,
                                    rows,
                                    cols,
                                    bytes_per_element,
                                );
                                out[target_index..target_index + bytes_per_element]
                                    .copy_from_slice(&data[cursor..cursor + bytes_per_element]);
                                cursor += bytes_per_element;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(out)
}

fn collect_bank_data(
    data: &[u8],
    page_size: usize,
    bank_index: usize,
    bank_count: usize,
) -> Vec<u8> {
    let page_count = data.len().div_ceil(page_size);
    let mut out = Vec::new();

    for page in (bank_index..page_count).step_by(bank_count) {
        let start = page * page_size;
        let end = data.len().min(start + page_size);
        out.extend_from_slice(&data[start..end]);
    }

    out
}

fn scatter_bank_data(
    out: &mut [u8],
    page_size: usize,
    bank_index: usize,
    bank_count: usize,
    bank_data: &[u8],
) {
    let page_count = out.len().div_ceil(page_size);

    for (slot, page) in (bank_index..page_count).step_by(bank_count).enumerate() {
        let out_start = page * page_size;
        let len = (out.len() - out_start).min(page_size);
        let bank_start = slot * page_size;
        out[out_start..out_start + len].copy_from_slice(&bank_data[bank_start..bank_start + len]);
    }
}

#[allow(clippy::manual_is_multiple_of)]
fn validate_tile_multiple(len: usize, dtype: DType) -> io::Result<()> {
    if len % dtype.tile_size() == 0 {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "data length {} is not a multiple of tile size {}",
                len,
                dtype.tile_size()
            ),
        ))
    }
}

fn next_allocation_range(
    next: u64,
    num_tiles: usize,
    dtype: DType,
    bank_count: usize,
) -> io::Result<(u64, u64)> {
    let allocation_size = allocation_range_size(num_tiles, dtype, bank_count)?;
    let aligned_end = next
        .checked_add(allocation_size)
        .ok_or_else(|| io::Error::other("dram allocation address overflow"))?;
    if aligned_end > Dram::TLB_SIZE_4G {
        return Err(io::Error::other(format!(
            "dram allocation exceeds per-bank address space: end=0x{aligned_end:x} limit=0x{:x}",
            Dram::TLB_SIZE_4G
        )));
    }
    Ok((next, aligned_end))
}

fn allocation_range_size(num_tiles: usize, dtype: DType, bank_count: usize) -> io::Result<u64> {
    if bank_count == 0 {
        return Err(io::Error::other("dram allocation requires at least one bank"));
    }
    let pages_per_bank = num_tiles.div_ceil(bank_count);
    let allocation_size = (pages_per_bank as u64)
        .checked_mul(dtype.tile_size() as u64)
        .ok_or_else(|| io::Error::other("dram allocation size overflow"))?;
    Ok(align_up(allocation_size, Dram::ALIGNMENT as u64))
}

fn element_offset(
    batch_index: usize,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    bytes_per_element: usize,
) -> usize {
    ((batch_index * rows + row) * cols + col) * bytes_per_element
}

struct Layout {
    batch: usize,
    rows: usize,
    cols: usize,
    bytes_per_element: usize,
}

impl Layout {
    fn new(data: &[u8], dtype: DType, shape: &[usize]) -> io::Result<Self> {
        let bytes_per_element = dtype.bytes_per_element();
        let (batch, rows, cols, expected_len) = validate_tiled_shape(data, dtype, shape)?;
        debug_assert_eq!(expected_len, data.len());
        Ok(Self {
            batch,
            rows,
            cols,
            bytes_per_element,
        })
    }
}

#[allow(clippy::manual_is_multiple_of)]
fn validate_tiled_shape(
    data: &[u8],
    dtype: DType,
    shape: &[usize],
) -> io::Result<(usize, usize, usize, usize)> {
    if shape.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "shape must have at least two dimensions",
        ));
    }

    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    if rows % TILE_R != 0 || cols % TILE_C != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("shape rows/cols must be multiples of {TILE_R}x{TILE_C}"),
        ));
    }

    let batch = shape[..shape.len() - 2]
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "shape is too large"))?;
    let element_count = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "shape is too large"))?;
    let expected_len = element_count
        .checked_mul(dtype.bytes_per_element())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "shape byte size is too large")
        })?;

    if data.len() != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "data length {} does not match shape byte size {}",
                data.len(),
                expected_len
            ),
        ));
    }

    Ok((batch, rows, cols, expected_len))
}

fn validate_allocation_shape(num_tiles: usize, shape: &[usize]) -> io::Result<()> {
    if shape.len() < 2 {
        return Err(invalid_input(
            "dram buffer allocation shape must have at least two dimensions",
        ));
    }
    if shape != tiled_allocation_shape(shape)?.as_slice() {
        return Err(invalid_input(format!(
            "dram buffer shape must be a tiled allocation shape, got {shape:?}"
        )));
    }
    let shape_tiles = tiled_shape_tile_count(shape)?;
    if shape_tiles != num_tiles {
        return Err(invalid_input(format!(
            "dram buffer tile count mismatch: shape {shape:?} requires {shape_tiles} tiles, got {num_tiles}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_from_u16(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bytes_from_u32(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn tilize_roundtrips_u16_tensor() {
        let values = (0..(64 * 64) as u16).collect::<Vec<_>>();
        let bytes = bytes_from_u16(&values);
        let tiled = tilize(&bytes, DType::Float16, &[64, 64]).expect("tilize should succeed");
        let untiled = untilize(&tiled, DType::Float16, &[64, 64]).expect("untilize should succeed");
        assert_eq!(untiled, bytes);
    }

    #[test]
    fn tilize_roundtrips_batched_u32_tensor() {
        let values = (0..(2 * 32 * 64) as u32).collect::<Vec<_>>();
        let bytes = bytes_from_u32(&values);
        let tiled = tilize(&bytes, DType::UInt32, &[2, 32, 64]).expect("tilize should succeed");
        let untiled =
            untilize(&tiled, DType::UInt32, &[2, 32, 64]).expect("untilize should succeed");
        assert_eq!(untiled, bytes);
    }

    #[test]
    fn buffer_size_matches_tile_count() {
        let buffer = DramBuffer {
            name: "weights".to_owned(),
            addr: Dram::WRITE_OFFSET,
            num_tiles: 3,
            dtype: DType::Float16,
            shape: vec![32, 96],
            _allocation: None,
        };

        assert_eq!(buffer.page_size(), 2048);
        assert_eq!(buffer.size(), 6144);
    }

    #[test]
    fn allocation_shape_validation_rejects_logical_shape() {
        let err = validate_allocation_shape(1, &[3, 2])
            .expect_err("logical shape must not be accepted as allocation shape");
        assert!(err.to_string().contains("tiled allocation shape"));
    }

    #[test]
    fn allocation_shape_validation_checks_tile_count() {
        let err = validate_allocation_shape(1, &[32, 64])
            .expect_err("shape tile count must match allocation tile count");
        assert!(err.to_string().contains("tile count mismatch"));
    }

    #[test]
    fn collect_bank_data_interleaves_pages() {
        let data = (0u8..10).collect::<Vec<_>>();
        assert_eq!(collect_bank_data(&data, 2, 0, 2), vec![0, 1, 4, 5, 8, 9]);
        assert_eq!(collect_bank_data(&data, 2, 1, 2), vec![2, 3, 6, 7]);
    }

    #[test]
    fn scatter_bank_data_restores_page_order() {
        let mut out = vec![0u8; 10];
        scatter_bank_data(&mut out, 2, 0, 2, &[0, 1, 4, 5, 8, 9]);
        scatter_bank_data(&mut out, 2, 1, 2, &[2, 3, 6, 7]);
        assert_eq!(out, (0u8..10).collect::<Vec<_>>());
    }

    #[test]
    fn collect_and_scatter_bank_data_roundtrip() {
        let input = (0u8..17).collect::<Vec<_>>();
        let page_size = 3;
        let bank_count = 3;
        let mut out = vec![0u8; input.len()];

        for bank_index in 0..bank_count {
            let bank_data = collect_bank_data(&input, page_size, bank_index, bank_count);
            scatter_bank_data(&mut out, page_size, bank_index, bank_count, &bank_data);
        }

        assert_eq!(out, input);
    }

    #[test]
    fn allocation_range_errors_when_capacity_is_exceeded() {
        let err = next_allocation_range(Dram::TLB_SIZE_4G, 1, DType::Float16, 1)
            .expect_err("allocation should exceed the per-bank address space");
        assert!(err.to_string().contains("exceeds per-bank address space"));
    }

    #[test]
    fn free_list_reuses_released_blocks() {
        let local_hardware_id = usize::MAX;
        reset_allocator_for_test(local_hardware_id);

        let (addr, size, _) =
            allocate_allocation_range(local_hardware_id, 1, DType::Float16, 1).unwrap();
        free_allocation(local_hardware_id, addr, size);

        let (reused, _, _) =
            allocate_allocation_range(local_hardware_id, 1, DType::Float16, 1).unwrap();
        assert_eq!(reused, addr);

        reset_allocator_for_test(local_hardware_id);
    }

    #[test]
    fn dram_buffer_clone_frees_on_last_drop() {
        let local_hardware_id = usize::MAX - 1;
        reset_allocator_for_test(local_hardware_id);
        let (addr, size, _) =
            allocate_allocation_range(local_hardware_id, 1, DType::Float16, 1).unwrap();
        let buffer = DramBuffer {
            name: "tmp".to_owned(),
            addr,
            num_tiles: 1,
            dtype: DType::Float16,
            shape: vec![32, 32],
            _allocation: Some(Arc::new(DramAllocation {
                local_hardware_id,
                addr,
                size,
            })),
        };
        let clone = buffer.clone();

        drop(buffer);
        assert_eq!(free_bytes_for_test(local_hardware_id), 0);
        drop(clone);
        assert_eq!(free_bytes_for_test(local_hardware_id), size);

        reset_allocator_for_test(local_hardware_id);
    }

    fn reset_allocator_for_test(local_hardware_id: usize) {
        let state = ALLOCATOR_STATE_BY_DEVICE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut state = state.lock().expect("allocator state lock poisoned");
        state.remove(&local_hardware_id);
    }

    fn free_bytes_for_test(local_hardware_id: usize) -> u64 {
        let state = ALLOCATOR_STATE_BY_DEVICE.get_or_init(|| Mutex::new(HashMap::new()));
        let state = state.lock().expect("allocator state lock poisoned");
        state
            .get(&local_hardware_id)
            .map(|device_state| device_state.free.iter().map(|block| block.size).sum())
            .unwrap_or(0)
    }
}
