use crate::linux::Sysmem as LinuxSysmem;
use std::io;
use std::path::PathBuf;

pub struct Sysmem {
    inner: LinuxSysmem,
}

impl Sysmem {
    pub const DEFAULT_SIZE: usize = 1 << 30;
    pub const PCIE_NOC_XY: u16 = (24 << 6) | 19;

    pub fn open(local_hardware_id: usize) -> io::Result<Self> {
        Self::with_size(local_hardware_id, Self::DEFAULT_SIZE)
    }

    pub fn with_size(local_hardware_id: usize, size: usize) -> io::Result<Self> {
        let path = PathBuf::from(format!("/dev/tenstorrent/{local_hardware_id}"));
        Ok(Self {
            inner: LinuxSysmem::open(path.as_path(), size)?,
        })
    }

    pub fn size(&self) -> usize {
        self.inner.size()
    }

    pub fn physical_address(&self) -> u64 {
        self.inner.physical_address()
    }

    pub fn noc_addr(&self) -> u64 {
        self.inner.noc_address()
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.inner.as_mut_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.inner.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut_slice()
    }

    pub fn write(&mut self, offset: usize, data: &[u8]) -> io::Result<()> {
        self.inner.write(offset, data)
    }

    pub fn read(&self, offset: usize, len: usize) -> io::Result<Vec<u8>> {
        self.inner.read(offset, len)
    }
}
