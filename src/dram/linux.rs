use crate::device::{CoreCoord, DramTile};
use std::ffi::{c_int, c_ulong, c_void};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::ptr;

const TLB_SIZE_4G: u64 = 1 << 32;
const TT_IOCTL_BASE: c_ulong = 0xFA << 8;
const TT_IOCTL_ALLOC_TLB: c_ulong = TT_IOCTL_BASE | 11;
const TT_IOCTL_FREE_TLB: c_ulong = TT_IOCTL_BASE | 12;
const TT_IOCTL_CONFIG_TLB: c_ulong = TT_IOCTL_BASE | 13;
const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;
const DRAM_BARRIER_BASE: usize = 0;
const DRAM_BARRIER_FLAGS: [u32; 2] = [0xaa, 0xbb];

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct AllocIn {
    size: u64,
    reserved: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct AllocOut {
    id: u32,
    reserved0: u32,
    mmap_offset_uc: u64,
    mmap_offset_wc: u64,
    reserved1: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct AllocTlbIo {
    input: AllocIn,
    output: AllocOut,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct FreeIn {
    id: u32,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct NocTlbConfig {
    addr: u64,
    x_end: u16,
    y_end: u16,
    x_start: u16,
    y_start: u16,
    noc: u8,
    mcast: u8,
    ordering: u8,
    linked: u8,
    static_vc: u8,
    reserved0: [u8; 3],
    reserved1: [u32; 2],
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct ConfigIn {
    id: u32,
    reserved: u32,
    config: NocTlbConfig,
}

#[derive(Clone, Copy)]
enum NocOrdering {
    Relaxed = 0,
    Strict = 1,
    Posted = 2,
}

pub(super) struct AllocatorBackend {
    window: TlbWindow,
    bank_tiles: Vec<DramTile>,
}

impl AllocatorBackend {
    pub(super) fn open(path: PathBuf, bank_tiles: Vec<DramTile>) -> io::Result<Self> {
        let first = bank_tiles
            .first()
            .copied()
            .ok_or_else(|| io::Error::other("no active DRAM bank tiles discovered"))?;
        let window = TlbWindow::open(
            path,
            CoreCoord {
                x: first.x,
                y: first.y,
            },
            TLB_SIZE_4G,
            true,
        )?;
        Ok(Self { window, bank_tiles })
    }

    pub(super) fn write(&mut self, addr: u64, page_size: usize, data: &[u8]) -> io::Result<()> {
        let page_count = data.len().div_ceil(page_size);

        for (bank_index, tile) in self.bank_tiles.iter().enumerate() {
            let bank_data = collect_bank_data(data, page_size, bank_index, self.bank_tiles.len());
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
            self.window.write(addr as usize, &bank_data)?;
        }

        if page_count > 0 {
            self.barrier()?;
        }
        Ok(())
    }

    pub(super) fn read(&mut self, addr: u64, page_size: usize, size: usize) -> io::Result<Vec<u8>> {
        let mut result = vec![0u8; size];
        let page_count = size.div_ceil(page_size);

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
            let bank_data = self.window.read(addr as usize, bank_pages * page_size)?;
            scatter_bank_data(
                &mut result,
                page_size,
                bank_index,
                self.bank_tiles.len(),
                &bank_data,
            );
        }

        Ok(result)
    }

    pub(super) fn read_raw_bank_pages(
        &mut self,
        addr: u64,
        page_size: usize,
    ) -> io::Result<Vec<u8>> {
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

struct TlbWindow {
    file: File,
    id: u32,
    mapping: Option<MappedRegion>,
}

impl TlbWindow {
    fn open(path: PathBuf, start: CoreCoord, size: u64, wc: bool) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|err| io::Error::new(err.kind(), format!("open {}: {err}", path.display())))?;

        let len = usize::try_from(size)
            .map_err(|_| io::Error::other(format!("TLB size {size} does not fit in usize")))?;

        let mut alloc = AllocTlbIo {
            input: AllocIn { size, reserved: 0 },
            output: AllocOut::default(),
        };
        ioctl_call(file.as_raw_fd(), TT_IOCTL_ALLOC_TLB, &mut alloc)?;

        let id = unsafe { ptr::addr_of!(alloc.output.id).read_unaligned() };
        let offset = if wc {
            unsafe { ptr::addr_of!(alloc.output.mmap_offset_wc).read_unaligned() }
        } else {
            unsafe { ptr::addr_of!(alloc.output.mmap_offset_uc).read_unaligned() }
        };

        let mut window = Self {
            file,
            id,
            mapping: None,
        };
        window.mapping = Some(MappedRegion::map(window.file.as_raw_fd(), len, offset)?);
        window.target(start, None, 0, NocOrdering::Strict)?;
        Ok(window)
    }

    fn target(
        &mut self,
        start: CoreCoord,
        end: Option<CoreCoord>,
        addr: u64,
        ordering: NocOrdering,
    ) -> io::Result<()> {
        let end = end.unwrap_or(start);
        let mut config = ConfigIn {
            id: self.id,
            reserved: 0,
            config: NocTlbConfig {
                addr,
                x_end: end.x as u16,
                y_end: end.y as u16,
                x_start: start.x as u16,
                y_start: start.y as u16,
                noc: 0,
                mcast: u8::from(end != start),
                ordering: ordering as u8,
                linked: 0,
                static_vc: 0,
                reserved0: [0; 3],
                reserved1: [0; 2],
            },
        };
        ioctl_call(self.file.as_raw_fd(), TT_IOCTL_CONFIG_TLB, &mut config)
    }

    fn read32(&self, offset: usize) -> io::Result<u32> {
        self.mapping
            .as_ref()
            .expect("TLB mapping should exist")
            .read32(offset)
    }

    fn write32(&mut self, offset: usize, value: u32) -> io::Result<()> {
        self.mapping
            .as_ref()
            .expect("TLB mapping should exist")
            .write(offset, &value.to_le_bytes())
    }

    fn write(&mut self, offset: usize, data: &[u8]) -> io::Result<()> {
        self.mapping
            .as_ref()
            .expect("TLB mapping should exist")
            .write(offset, data)
    }

    fn read(&self, offset: usize, len: usize) -> io::Result<Vec<u8>> {
        self.mapping
            .as_ref()
            .expect("TLB mapping should exist")
            .read(offset, len)
    }
}

impl Drop for TlbWindow {
    fn drop(&mut self) {
        drop(self.mapping.take());
        let mut free = FreeIn { id: self.id };
        let _ = ioctl_call(self.file.as_raw_fd(), TT_IOCTL_FREE_TLB, &mut free);
    }
}

struct MappedRegion {
    addr: *mut u8,
    len: usize,
}

impl MappedRegion {
    fn map(fd: c_int, len: usize, offset: u64) -> io::Result<Self> {
        let addr = unsafe {
            mmap(
                ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                offset as i64,
            )
        };
        if addr as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            addr: addr.cast::<u8>(),
            len,
        })
    }

    fn read32(&self, offset: usize) -> io::Result<u32> {
        let bytes = self.read(offset, 4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("slice length is fixed"),
        ))
    }

    fn write(&self, offset: usize, data: &[u8]) -> io::Result<()> {
        self.check_range(offset, data.len())?;
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.addr.add(offset), data.len());
        }
        Ok(())
    }

    fn read(&self, offset: usize, len: usize) -> io::Result<Vec<u8>> {
        self.check_range(offset, len)?;
        let bytes = unsafe { std::slice::from_raw_parts(self.addr.add(offset), len) };
        Ok(bytes.to_vec())
    }

    fn check_range(&self, offset: usize, len: usize) -> io::Result<()> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| io::Error::other("TLB access overflow"))?;
        if end > self.len {
            Err(io::Error::other(format!(
                "TLB access out of range: offset=0x{offset:x} len=0x{len:x} mapping_len=0x{:x}",
                self.len
            )))
        } else {
            Ok(())
        }
    }
}

impl Drop for MappedRegion {
    fn drop(&mut self) {
        let _ = unsafe { munmap(self.addr.cast::<c_void>(), self.len) };
    }
}

fn ioctl_call<T>(fd: c_int, request: c_ulong, data: &mut T) -> io::Result<()> {
    let result = unsafe { ioctl(fd, request, data as *mut T) };
    if result == -1 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

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
