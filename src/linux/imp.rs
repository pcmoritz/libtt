use crate::hw::{align_up, CoreCoord};
use std::ffi::{c_int, c_ulong, c_void};
use std::fs::{File, OpenOptions};
use std::io;
use std::mem::size_of;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;

const TT_IOCTL_BASE: c_ulong = 0xFA << 8;
const TT_IOCTL_ALLOC_TLB: c_ulong = TT_IOCTL_BASE | 11;
const TT_IOCTL_FREE_TLB: c_ulong = TT_IOCTL_BASE | 12;
const TT_IOCTL_CONFIG_TLB: c_ulong = TT_IOCTL_BASE | 13;
const TT_IOCTL_PIN_PAGES: c_ulong = TT_IOCTL_BASE | 7;
const TT_IOCTL_UNPIN_PAGES: c_ulong = TT_IOCTL_BASE | 10;
const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;
const MAP_ANONYMOUS: c_int = 0x20;
const MAP_HUGETLB: c_int = 0x40000;
const MAP_HUGE_1GB: c_int = 30 << 26;
const PAGE_SIZE: usize = 4096;
const HUGE_PAGE_SIZE: usize = 1 << 30;
const PIN_NOC_DMA: u32 = 2;

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

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct PinIn {
    output_size_bytes: u32,
    flags: u32,
    virtual_address: u64,
    size: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct PinOut {
    physical_address: u64,
    noc_address: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct PinIo {
    input: PinIn,
    output: PinOut,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
struct UnpinIn {
    virtual_address: u64,
    size: u64,
    reserved: u64,
}

#[derive(Clone, Copy)]
pub(crate) enum NocOrdering {
    Relaxed = 0,
    Strict = 1,
    Posted = 2,
}

pub(crate) struct TlbWindow {
    file: File,
    id: u32,
    mapping: Option<MappedRegion>,
}

impl TlbWindow {
    pub(crate) fn open(path: &Path, size: u64, wc: bool) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|err| io::Error::new(err.kind(), format!("open {}: {err}", path.display())))?;
        Self::new(file, size, wc)
    }

    fn new(file: File, size: u64, wc: bool) -> io::Result<Self> {
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
        window.mapping = Some(MappedRegion::map(
            len,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            window.file.as_raw_fd(),
            offset,
        )?);
        Ok(window)
    }

    pub(crate) fn target(
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

    pub(crate) fn read32(&self, offset: usize) -> io::Result<u32> {
        self.mapping
            .as_ref()
            .expect("TLB mapping should exist while window is alive")
            .read32(offset)
    }

    pub(crate) fn write32(&mut self, offset: usize, value: u32) -> io::Result<()> {
        self.mapping
            .as_ref()
            .expect("TLB mapping should exist while window is alive")
            .write(offset, &value.to_le_bytes())
    }

    pub(crate) fn write(&mut self, offset: usize, data: &[u8]) -> io::Result<()> {
        self.mapping
            .as_ref()
            .expect("TLB mapping should exist while window is alive")
            .write(offset, data)
    }

    pub(crate) fn read(&self, offset: usize, len: usize) -> io::Result<Vec<u8>> {
        self.mapping
            .as_ref()
            .expect("TLB mapping should exist while window is alive")
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

pub(crate) struct PinnedMemory {
    file: File,
    mapping: MappedRegion,
    noc_addr: u64,
}

impl PinnedMemory {
    pub(crate) fn new(path: &Path, size: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|err| io::Error::new(err.kind(), format!("open {}: {err}", path.display())))?;
        let len = align_up(size as u64, HUGE_PAGE_SIZE as u64) as usize;
        let mapping = MappedRegion::map(
            len,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
            -1,
            0,
        )
        .map_err(|err| {
            io::Error::new(
                err.kind(),
                format!("map pinned memory size=0x{size:x} len=0x{len:x}: {err}"),
            )
        })?;
        let virtual_address = mapping.addr as u64;
        let size = mapping.len;
        if virtual_address % PAGE_SIZE as u64 != 0 || size % PAGE_SIZE != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "pinned memory must be page-aligned and page-sized: va=0x{virtual_address:x} size=0x{size:x}"
                ),
            ));
        }
        let mut pin = PinIo {
            input: PinIn {
                output_size_bytes: size_of::<PinOut>() as u32,
                flags: PIN_NOC_DMA,
                virtual_address,
                size: size as u64,
            },
            output: PinOut::default(),
        };
        ioctl_call(file.as_raw_fd(), TT_IOCTL_PIN_PAGES, &mut pin).map_err(|err| {
            io::Error::new(
                err.kind(),
                format!(
                    "pin NOC DMA pages failed for {}: va=0x{virtual_address:x} size=0x{size:x} flags={PIN_NOC_DMA}: {err}",
                    path.display()
                ),
            )
        })?;
        let noc_addr = unsafe { ptr::addr_of!(pin.output.noc_address).read_unaligned() };
        Ok(Self {
            file,
            mapping,
            noc_addr,
        })
    }

    pub(crate) fn noc_addr(&self) -> u64 {
        self.noc_addr
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        self.mapping.as_mut_slice()
    }

    pub(crate) fn read32(&self, offset: usize) -> io::Result<u32> {
        self.mapping.read32(offset)
    }

    pub(crate) fn write32(&mut self, offset: usize, value: u32) -> io::Result<()> {
        self.mapping.write(offset, &value.to_le_bytes())
    }
}

impl Drop for PinnedMemory {
    fn drop(&mut self) {
        let mut unpin = UnpinIn {
            virtual_address: self.mapping.addr as u64,
            size: self.mapping.len as u64,
            reserved: 0,
        };
        let _ = ioctl_call(self.file.as_raw_fd(), TT_IOCTL_UNPIN_PAGES, &mut unpin);
    }
}

struct MappedRegion {
    addr: *mut u8,
    len: usize,
}

impl MappedRegion {
    fn map(len: usize, prot: c_int, flags: c_int, fd: c_int, offset: u64) -> io::Result<Self> {
        let addr = unsafe { mmap(ptr::null_mut(), len, prot, flags, fd, offset as i64) };
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

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.addr, self.len) }
    }

    fn check_range(&self, offset: usize, len: usize) -> io::Result<()> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| io::Error::other("mapping access overflow"))?;
        if end > self.len {
            Err(io::Error::other(format!(
                "mapping access out of range: offset=0x{offset:x} len=0x{len:x} mapping_len=0x{:x}",
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
