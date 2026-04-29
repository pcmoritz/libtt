use crate::device::Device;
use crate::dispatch::Program;
use crate::hw::CoreCoord;
use std::io;

pub(crate) trait Kernel {
    fn program(&self, device: &Device) -> io::Result<Program>;

    fn reader_runtime_arg(&self, _core: CoreCoord, _index: usize) -> io::Result<Option<u32>> {
        Ok(None)
    }

    fn writer_runtime_arg(&self, _core: CoreCoord, _index: usize) -> io::Result<Option<u32>> {
        Ok(None)
    }

    fn compute_runtime_arg(&self, _core: CoreCoord, _index: usize) -> io::Result<Option<u32>> {
        Ok(None)
    }
}
