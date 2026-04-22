#[cfg(libtt_mlir_frontend)]
use std::ffi::c_char;

#[cfg(libtt_mlir_frontend)]
#[repr(C)]
pub(crate) struct TT_MlirAnalysis {
    pub(crate) status: i32,
    pub(crate) output_type: i32,
    pub(crate) num_output_dims: usize,
    pub(crate) output_dims: *mut i64,
    pub(crate) optimized_program: *mut c_char,
    pub(crate) optimized_program_size: usize,
    pub(crate) error_message: *mut c_char,
}

#[cfg(libtt_mlir_frontend)]
unsafe extern "C" {
    fn TT_MlirAnalyzeProgram(
        format: *const c_char,
        format_size: usize,
        code: *const c_char,
        code_size: usize,
    ) -> *mut TT_MlirAnalysis;
    fn TT_MlirAnalysisDestroy(analysis: *mut TT_MlirAnalysis);
}

#[cfg(libtt_mlir_frontend)]
pub(crate) const STATUS_PARSE_ERROR: i32 = 1;
#[cfg(libtt_mlir_frontend)]
pub(crate) const STATUS_UNSUPPORTED: i32 = 2;

#[cfg(libtt_mlir_frontend)]
pub(crate) const ELEMENT_TYPE_BF16: i32 = 1;
#[cfg(libtt_mlir_frontend)]
pub(crate) const ELEMENT_TYPE_F16: i32 = 2;
#[cfg(libtt_mlir_frontend)]
pub(crate) const ELEMENT_TYPE_F32: i32 = 3;
#[cfg(libtt_mlir_frontend)]
pub(crate) const ELEMENT_TYPE_U32: i32 = 4;
#[cfg(libtt_mlir_frontend)]
pub(crate) const ELEMENT_TYPE_U16: i32 = 5;
#[cfg(libtt_mlir_frontend)]
pub(crate) const ELEMENT_TYPE_U8: i32 = 6;
#[cfg(libtt_mlir_frontend)]
pub(crate) const ELEMENT_TYPE_S32: i32 = 7;
#[cfg(libtt_mlir_frontend)]
pub(crate) const ELEMENT_TYPE_S8: i32 = 8;

#[cfg(libtt_mlir_frontend)]
pub(crate) struct AnalysisHandle {
    raw: *mut TT_MlirAnalysis,
}

#[cfg(libtt_mlir_frontend)]
impl AnalysisHandle {
    pub(crate) fn analyze(
        format: *const c_char,
        format_size: usize,
        code: *const c_char,
        code_size: usize,
    ) -> Option<Self> {
        let raw = unsafe { TT_MlirAnalyzeProgram(format, format_size, code, code_size) };
        (!raw.is_null()).then_some(Self { raw })
    }

    pub(crate) fn analysis(&self) -> &TT_MlirAnalysis {
        unsafe { &*self.raw }
    }
}

#[cfg(libtt_mlir_frontend)]
impl Drop for AnalysisHandle {
    fn drop(&mut self) {
        unsafe {
            TT_MlirAnalysisDestroy(self.raw);
        }
    }
}
