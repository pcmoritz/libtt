use crate::device::CoreCoord;
use std::io;
use std::path::Path;

#[derive(Clone, Copy)]
pub(crate) enum NocOrdering {
    Relaxed = 0,
    Strict = 1,
    Posted = 2,
}

pub(crate) struct TlbWindow;

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
