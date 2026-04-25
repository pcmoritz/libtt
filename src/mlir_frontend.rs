#[cfg(libtt_mlir_frontend)]
use std::ffi::{c_char, c_void};
#[cfg(libtt_mlir_frontend)]
use std::ptr;

#[cfg(libtt_mlir_frontend)]
unsafe extern "C" {
    fn TT_MlirAnalyzeProgram(
        format: *const c_char,
        format_size: usize,
        code: *const c_char,
        code_size: usize,
        alloc_output: unsafe extern "C" fn(usize, *mut c_void) -> *mut c_char,
        user_data: *mut c_void,
    ) -> bool;
}

#[cfg(libtt_mlir_frontend)]
pub(crate) struct AnalysisHandle {
    data: Vec<u8>,
}

#[cfg(libtt_mlir_frontend)]
unsafe extern "C" fn alloc_output(size: usize, user_data: *mut c_void) -> *mut c_char {
    if user_data.is_null() {
        return ptr::null_mut();
    }
    let data = unsafe { &mut *user_data.cast::<Vec<u8>>() };
    data.clear();
    if data.try_reserve_exact(size).is_err() {
        return ptr::null_mut();
    }
    data.resize(size, 0);
    data.as_mut_ptr().cast()
}

#[cfg(libtt_mlir_frontend)]
impl AnalysisHandle {
    pub(crate) fn analyze(
        format: *const c_char,
        format_size: usize,
        code: *const c_char,
        code_size: usize,
    ) -> Option<Self> {
        let mut data = Vec::new();
        let ok = unsafe {
            TT_MlirAnalyzeProgram(
                format,
                format_size,
                code,
                code_size,
                alloc_output,
                (&mut data as *mut Vec<u8>).cast(),
            )
        };
        ok.then_some(Self { data })
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.data
    }
}
