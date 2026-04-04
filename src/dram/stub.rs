use std::io;
use std::path::PathBuf;

pub(super) struct AllocatorBackend;

impl AllocatorBackend {
    pub(super) fn open(
        _path: PathBuf,
        _bank_tiles: Vec<crate::device::DramTile>,
    ) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "dram allocator is only available on Linux",
        ))
    }

    pub(super) fn write(&mut self, _addr: u64, _page_size: usize, _data: &[u8]) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "dram allocator is only available on Linux",
        ))
    }

    pub(super) fn read(
        &mut self,
        _addr: u64,
        _page_size: usize,
        _size: usize,
    ) -> io::Result<Vec<u8>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "dram allocator is only available on Linux",
        ))
    }

    pub(super) fn read_raw_bank_pages(
        &mut self,
        _addr: u64,
        _page_size: usize,
    ) -> io::Result<Vec<u8>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "dram allocator is only available on Linux",
        ))
    }
}
