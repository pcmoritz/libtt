use crate::hw::CoreCoord;
use std::io;
use std::path::Path;

#[derive(Clone, Copy)]
pub(crate) enum NocOrdering {
    Relaxed = 0,
    Strict = 1,
    Posted = 2,
}

pub(crate) struct TlbWindow;
#[allow(dead_code)]
pub(crate) struct Sysmem;

impl TlbWindow {
    pub(crate) fn open(
        _path: &Path,
        _start: CoreCoord,
        _addr: u64,
        _size: u64,
        _wc: bool,
    ) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn target(
        &mut self,
        _start: CoreCoord,
        _end: Option<CoreCoord>,
        _addr: u64,
        _ordering: NocOrdering,
    ) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn read32(&self, _offset: usize) -> io::Result<u32> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn write32(&mut self, _offset: usize, _value: u32) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn write(&mut self, _offset: usize, _data: &[u8]) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn read(&self, _offset: usize, _len: usize) -> io::Result<Vec<u8>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }
}

#[allow(dead_code)]
impl Sysmem {
    pub(crate) const DEFAULT_SIZE: usize = 1 << 30;
    pub(crate) const PCIE_NOC_XY: u16 = (24 << 6) | 19;

    pub(crate) fn open(_local_hardware_id: usize) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn with_size(_local_hardware_id: usize, _size: usize) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn size(&self) -> usize {
        0
    }

    pub(crate) fn physical_address(&self) -> u64 {
        0
    }

    pub(crate) fn noc_addr(&self) -> u64 {
        0
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &[]
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut []
    }

    pub(crate) fn write(&mut self, _offset: usize, _data: &[u8]) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn read(&self, _offset: usize, _len: usize) -> io::Result<Vec<u8>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }
}
