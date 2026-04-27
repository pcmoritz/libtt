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

pub(crate) struct PinnedMemory;

impl TlbWindow {
    pub(crate) fn open(_path: &Path, _size: u64, _wc: bool) -> io::Result<Self> {
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

impl PinnedMemory {
    pub(crate) fn new(_path: &Path, _size: usize) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "Tenstorrent Linux backend is only available on Linux",
        ))
    }

    pub(crate) fn noc_addr(&self) -> u64 {
        0
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        &[]
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut []
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
}
