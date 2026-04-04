use super::{CoreCoord, ProbeInfo, log};
use std::ffi::{c_int, c_ulong, c_void};
use std::fs::{self, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr;

const DEFAULT_TENSIX_ENABLED: u32 = 0x3fff;
const DEFAULT_GDDR_ENABLED: u32 = 0xff;
const ARC_TILE: CoreCoord = CoreCoord { x: 8, y: 0 };
const ARC_NOC_BASE: u64 = 0x8000_0000;
const ARC_SCRATCH_RAM_13: usize = 0x30434;
const TAG_TENSIX_ENABLED_COL: u16 = 34;
const TAG_GDDR_ENABLED: u16 = 36;
const TLB_SIZE_2M: u64 = 1 << 21;
const TT_IOCTL_BASE: c_ulong = 0xFA << 8;
const TT_IOCTL_ALLOC_TLB: c_ulong = TT_IOCTL_BASE | 11;
const TT_IOCTL_FREE_TLB: c_ulong = TT_IOCTL_BASE | 12;
const TT_IOCTL_CONFIG_TLB: c_ulong = TT_IOCTL_BASE | 13;
const TT_IOCTL_SET_POWER_STATE: c_ulong = TT_IOCTL_BASE | 15;
const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x01;

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
struct PowerStateIn {
    argsz: u32,
    flags: u32,
    reserved0: u8,
    validity: u8,
    power_flags: u16,
    power_settings: [u16; 14],
}

pub(super) fn detect_probe_info(local_hardware_id: usize, path: &Path) -> Option<ProbeInfo> {
    match probe_info_for_device(local_hardware_id, path) {
        Ok(probe) => probe,
        Err(err) => {
            log(format!(
                "linux probe local_hardware_id={local_hardware_id} failed: {err}"
            ));
            None
        }
    }
}

fn probe_info_for_device(local_hardware_id: usize, path: &Path) -> io::Result<Option<ProbeInfo>> {
    let card_type = read_card_type(local_hardware_id)?;
    log(format!(
        "linux probe local_hardware_id={local_hardware_id} path={} card_type={card_type}",
        path.display()
    ));

    let probe = ProbeDevice::open(path)?;
    let (gddr_enabled_mask, tensix_enabled_col_mask) = probe.read_arc_enabled_masks()?;
    log(format!(
        "linux probe local_hardware_id={local_hardware_id} tensix_enabled_col_mask=0x{tensix_enabled_col_mask:08x} gddr_enabled_mask=0x{gddr_enabled_mask:08x}"
    ));

    Ok(Some(ProbeInfo {
        arch: card_type,
        tensix_enabled_col_mask,
        gddr_enabled_mask,
    }))
}

fn read_card_type(local_hardware_id: usize) -> io::Result<String> {
    let path = PathBuf::from(format!(
        "/sys/class/tenstorrent/tenstorrent!{local_hardware_id}/tt_card_type"
    ));
    let card_type = fs::read_to_string(&path)
        .map_err(|err| io::Error::new(err.kind(), format!("read {}: {err}", path.display())))?;
    Ok(card_type.trim().to_owned())
}

struct ProbeDevice {
    file: std::fs::File,
}

impl ProbeDevice {
    fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|err| io::Error::new(err.kind(), format!("open {}: {err}", path.display())))?;
        log(format!("linux probe opened {}", path.display()));
        Ok(Self { file })
    }

    fn read_arc_enabled_masks(&self) -> io::Result<(u32, u32)> {
        let mut power_state = PowerStateIn {
            argsz: std::mem::size_of::<PowerStateIn>() as u32,
            validity: 1,
            power_flags: 1,
            ..PowerStateIn::default()
        };
        if let Err(err) = ioctl_call(self.file.as_raw_fd(), TT_IOCTL_SET_POWER_STATE, &mut power_state) {
            log(format!("linux probe set_power_state ioctl failed: {err}"));
        }

        let mut arc = TlbWindow::new(&self.file, ARC_TILE, ARC_NOC_BASE)?;
        let telemetry_ptr = arc.read32(ARC_SCRATCH_RAM_13)? as u64;
        let (csm_base, csm_offset) = align_down(telemetry_ptr, TLB_SIZE_2M);
        log(format!(
            "linux probe telemetry_ptr=0x{telemetry_ptr:x} csm_base=0x{csm_base:x} csm_offset=0x{csm_offset:x}"
        ));
        arc.target(ARC_TILE, None, csm_base)?;

        let entry_count = arc.read32((csm_offset + 4) as usize)? as usize;
        log(format!("linux probe telemetry entry_count={entry_count}"));
        if entry_count == 0 || entry_count > 4096 {
            return Err(io::Error::other(format!(
                "invalid ARC telemetry entry_count {entry_count}"
            )));
        }

        let tags_base = csm_offset + 8;
        let data_base = tags_base + (entry_count as u64) * 4;
        let mut tensix_data_offset = None;
        let mut gddr_data_offset = None;

        for index in 0..entry_count {
            let tag_offset = arc.read32((tags_base + (index as u64) * 4) as usize)?;
            let tag = (tag_offset & 0xffff) as u16;
            let data_offset_words = (tag_offset >> 16) & 0xffff;

            if tag == TAG_TENSIX_ENABLED_COL {
                tensix_data_offset = Some(data_offset_words);
            } else if tag == TAG_GDDR_ENABLED {
                gddr_data_offset = Some(data_offset_words);
            }
        }

        let tensix_enabled_col_mask = match tensix_data_offset {
            Some(offset_words) => arc.read32((data_base + (offset_words as u64) * 4) as usize)?,
            None => DEFAULT_TENSIX_ENABLED,
        };
        let gddr_enabled_mask = match gddr_data_offset {
            Some(offset_words) => arc.read32((data_base + (offset_words as u64) * 4) as usize)?,
            None => DEFAULT_GDDR_ENABLED,
        };
        Ok((gddr_enabled_mask, tensix_enabled_col_mask))
    }
}

struct TlbWindow<'a> {
    file: &'a std::fs::File,
    id: u32,
    mapping: Option<MappedRegion>,
}

impl<'a> TlbWindow<'a> {
    fn new(file: &'a std::fs::File, start: CoreCoord, addr: u64) -> io::Result<Self> {
        let mut alloc = AllocTlbIo {
            input: AllocIn {
                size: TLB_SIZE_2M,
                reserved: 0,
            },
            output: AllocOut::default(),
        };
        ioctl_call(file.as_raw_fd(), TT_IOCTL_ALLOC_TLB, &mut alloc)?;

        let id = unsafe { ptr::addr_of!(alloc.output.id).read_unaligned() };
        let mmap_offset_uc = unsafe { ptr::addr_of!(alloc.output.mmap_offset_uc).read_unaligned() };
        let mapping = MappedRegion::map(file.as_raw_fd(), TLB_SIZE_2M as usize, mmap_offset_uc)?;
        let mut window = Self {
            file,
            id,
            mapping: Some(mapping),
        };
        window.target(start, None, addr)?;
        Ok(window)
    }

    fn target(&mut self, start: CoreCoord, end: Option<CoreCoord>, addr: u64) -> io::Result<()> {
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
                ordering: 1,
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
            .expect("TLB mapping should exist while window is alive")
            .read32(offset)
    }
}

impl Drop for TlbWindow<'_> {
    fn drop(&mut self) {
        drop(self.mapping.take());
        let mut free = FreeIn { id: self.id };
        if let Err(err) = ioctl_call(self.file.as_raw_fd(), TT_IOCTL_FREE_TLB, &mut free) {
            log(format!("linux probe free_tlb ioctl failed: {err}"));
        }
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
        let end = offset
            .checked_add(4)
            .ok_or_else(|| io::Error::other("TLB read overflow"))?;
        if end > self.len {
            return Err(io::Error::other(format!(
                "TLB read out of range: offset=0x{offset:x} len=0x{:x}",
                self.len
            )));
        }

        let bytes = unsafe { std::slice::from_raw_parts(self.addr.add(offset), 4) };
        Ok(u32::from_le_bytes(bytes.try_into().expect("slice length is fixed")))
    }
}

impl Drop for MappedRegion {
    fn drop(&mut self) {
        let result = unsafe { munmap(self.addr.cast::<c_void>(), self.len) };
        if result != 0 {
            log(format!(
                "linux probe munmap failed: {}",
                io::Error::last_os_error()
            ));
        }
    }
}

fn align_down(value: u64, alignment: u64) -> (u64, u64) {
    let base = value & !(alignment - 1);
    (base, value - base)
}

fn ioctl_call<T>(fd: c_int, request: c_ulong, data: &mut T) -> io::Result<()> {
    let result = unsafe { ioctl(fd, request, data as *mut T) };
    if result == -1 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}
