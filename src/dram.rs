use crate::device::CoreCoord;
use crate::device::{Device, DramTile, load_device};
use crate::linux::{NocOrdering, TlbWindow};
use std::io;
use std::path::PathBuf;

const TILE_R: usize = 32;
const TILE_C: usize = 32;
const FACE_R: usize = 16;
const FACE_C: usize = 16;
const DRAM_TILES_PER_BANK: usize = 3;
const DRAM_ALIGNMENT: usize = 64;
const TLB_SIZE_4G: u64 = 1 << 32;
const DRAM_BARRIER_BASE: usize = 0;
const DRAM_BARRIER_FLAGS: [u32; 2] = [0xaa, 0xbb];

type Shape = Vec<usize>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DramBuffer {
    pub name: String,
    pub addr: u64,
    pub num_tiles: usize,
    pub dtype: DType,
    pub shape: Option<Shape>,
}

impl DramBuffer {
    pub(crate) fn page_size(&self) -> usize {
        self.dtype.tile_size()
    }

    pub(crate) fn size(&self) -> usize {
        self.num_tiles * self.page_size()
    }
}

pub struct Allocator {
    window: TlbWindow,
    bank_tiles: Vec<DramTile>,
    next: u64,
    bank_count: usize,
}

impl Allocator {
    pub fn open(local_hardware_id: usize) -> io::Result<Self> {
        let (path, info) = load_device(local_hardware_id);
        Self::from_device_with_path(path, &info)
    }

    pub(crate) fn from_device(device: &Device) -> io::Result<Self> {
        Self::from_device_with_path(device.path.clone(), device)
    }

    fn from_device_with_path(path: PathBuf, device: &Device) -> io::Result<Self> {
        let bank_tiles = allocator_bank_tiles(&device.dram_tiles)?;
        let first = bank_tiles
            .first()
            .copied()
            .ok_or_else(|| io::Error::other("no active DRAM bank tiles discovered"))?;
        let bank_count = bank_tiles.len();
        let window = TlbWindow::open(
            path.as_path(),
            CoreCoord {
                x: first.x,
                y: first.y,
            },
            0,
            TLB_SIZE_4G,
            true,
        )?;
        Ok(Self {
            window,
            bank_tiles,
            next: 0x40,
            bank_count,
        })
    }

    pub fn alloc(
        &mut self,
        num_tiles: usize,
        dtype: DType,
        name: impl Into<String>,
        shape: Option<Shape>,
    ) -> io::Result<DramBuffer> {
        if self.bank_count == 0 {
            return Err(io::Error::other("allocator has no active DRAM banks"));
        }

        let pages_per_bank = num_tiles.div_ceil(self.bank_count);
        let addr = self.next;
        self.next = align_up(
            addr + (pages_per_bank as u64) * (dtype.tile_size() as u64),
            DRAM_ALIGNMENT as u64,
        );
        Ok(DramBuffer {
            name: name.into(),
            addr,
            num_tiles,
            dtype,
            shape,
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
        let buf = self.alloc(data.len() / dtype.tile_size(), dtype, name, Some(shape))?;
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
        self.alloc(num_tiles, dtype, name, Some(shape))
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
        let payload = match &buf.shape {
            Some(shape) => tilize(data, buf.dtype, shape)?,
            None => data.to_vec(),
        };
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
        match &buf.shape {
            Some(shape) => untilize(&payload, buf.dtype, shape),
            None => Ok(payload),
        }
    }

    pub fn read_raw_bank_pages(&mut self, addr: u64, page_size: usize) -> io::Result<Vec<u8>> {
        let mut result = vec![0u8; page_size * self.bank_tiles.len()];

        for (bank_index, tile) in self.bank_tiles.iter().enumerate() {
            self.window.target(
                CoreCoord {
                    x: tile.x,
                    y: tile.y,
                },
                None,
                0,
                NocOrdering::Relaxed,
            )?;
            let bank_data = self.window.read(addr as usize, page_size)?;
            let offset = bank_index * page_size;
            result[offset..offset + page_size].copy_from_slice(&bank_data);
        }

        Ok(result)
    }

    fn barrier(&mut self) -> io::Result<()> {
        for flag in DRAM_BARRIER_FLAGS {
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
                self.window.write32(DRAM_BARRIER_BASE, flag)?;
                while self.window.read32(DRAM_BARRIER_BASE)? != flag {}
            }
        }
        Ok(())
    }
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

fn allocator_bank_tiles(dram_tiles: &[DramTile]) -> io::Result<Vec<DramTile>> {
    let bank_tiles = dram_tiles
        .iter()
        .step_by(DRAM_TILES_PER_BANK)
        .copied()
        .collect::<Vec<_>>();
    if bank_tiles.is_empty() {
        Err(io::Error::other("no active DRAM bank tiles discovered"))
    } else {
        Ok(bank_tiles)
    }
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

fn align_up(value: u64, align: u64) -> u64 {
    value.div_ceil(align) * align
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
            addr: 0x40,
            num_tiles: 3,
            dtype: DType::Float16,
            shape: Some(vec![32, 96]),
        };

        assert_eq!(buffer.page_size(), 2048);
        assert_eq!(buffer.size(), 6144);
    }

    #[test]
    fn allocator_bank_tiles_picks_one_tile_per_bank() {
        let tiles = vec![
            DramTile {
                bank: 0,
                x: 0,
                y: 0,
            },
            DramTile {
                bank: 0,
                x: 0,
                y: 1,
            },
            DramTile {
                bank: 0,
                x: 0,
                y: 2,
            },
            DramTile {
                bank: 1,
                x: 9,
                y: 0,
            },
            DramTile {
                bank: 1,
                x: 9,
                y: 1,
            },
            DramTile {
                bank: 1,
                x: 9,
                y: 2,
            },
        ];

        let bank_tiles = allocator_bank_tiles(&tiles).expect("bank tiles should exist");
        assert_eq!(bank_tiles.len(), 2);
        assert_eq!(bank_tiles[0].bank, 0);
        assert_eq!(bank_tiles[1].bank, 1);
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
}
