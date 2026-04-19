#![allow(dead_code)]

mod device;
mod dram;
mod hw;
mod linux;
mod log;

use device::Device;
use dram::{DType, DramBuffer};
use std::ffi::{CString, c_char, c_void};
use std::io;
use std::ptr;
use std::slice;

const PJRT_BUFFER_TYPE_INVALID: i32 = 0;
const PJRT_BUFFER_TYPE_S8: i32 = 2;
const PJRT_BUFFER_TYPE_S32: i32 = 4;
const PJRT_BUFFER_TYPE_U8: i32 = 6;
const PJRT_BUFFER_TYPE_U16: i32 = 7;
const PJRT_BUFFER_TYPE_U32: i32 = 8;
const PJRT_BUFFER_TYPE_F16: i32 = 10;
const PJRT_BUFFER_TYPE_F32: i32 = 11;
const PJRT_BUFFER_TYPE_BF16: i32 = 13;

const PJRT_ERROR_CODE_INVALID_ARGUMENT: i32 = 3;
const PJRT_ERROR_CODE_RESOURCE_EXHAUSTED: i32 = 8;
const PJRT_ERROR_CODE_FAILED_PRECONDITION: i32 = 9;
const PJRT_ERROR_CODE_UNIMPLEMENTED: i32 = 12;
const PJRT_ERROR_CODE_INTERNAL: i32 = 13;

#[repr(C)]
pub struct TTRustError {
    pub code: i32,
    pub message: *mut c_char,
}

#[repr(C)]
pub struct TTDeviceInfo {
    pub id: i32,
    pub local_hardware_id: i32,
    pub arch: *const c_char,
    pub device_kind: *const c_char,
    pub device_debug_string: *const c_char,
    pub device_to_string: *const c_char,
    pub memory_debug_string: *const c_char,
    pub memory_to_string: *const c_char,
}

struct OwnedDeviceInfo {
    arch: CString,
    device_kind: CString,
    device_debug_string: CString,
    device_to_string: CString,
    memory_debug_string: CString,
    memory_to_string: CString,
}

#[repr(C)]
pub struct TTRustDiscovery {
    _owned: Vec<OwnedDeviceInfo>,
    raw: Vec<TTDeviceInfo>,
}

#[repr(C)]
pub struct TTRustBufferHandle {
    local_hardware_id: usize,
    buffer_type: i32,
    dims: Vec<i64>,
    dram_buffer: Option<DramBuffer>,
    deleted: bool,
}

fn cstring_lossy(value: impl AsRef<str>) -> CString {
    let sanitized = value.as_ref().replace('\0', "?");
    CString::new(sanitized).expect("CString::new should succeed after sanitizing NULs")
}

fn rust_error(code: i32, message: impl AsRef<str>) -> *mut TTRustError {
    let message = cstring_lossy(message);
    Box::into_raw(Box::new(TTRustError {
        code,
        message: message.into_raw(),
    }))
}

fn invalid_argument(message: impl AsRef<str>) -> *mut TTRustError {
    rust_error(PJRT_ERROR_CODE_INVALID_ARGUMENT, message)
}

fn unimplemented(message: impl AsRef<str>) -> *mut TTRustError {
    rust_error(PJRT_ERROR_CODE_UNIMPLEMENTED, message)
}

fn failed_precondition(message: impl AsRef<str>) -> *mut TTRustError {
    rust_error(PJRT_ERROR_CODE_FAILED_PRECONDITION, message)
}

fn resource_exhausted(message: impl AsRef<str>) -> *mut TTRustError {
    rust_error(PJRT_ERROR_CODE_RESOURCE_EXHAUSTED, message)
}

fn io_error(err: io::Error) -> *mut TTRustError {
    let code = match err.kind() {
        io::ErrorKind::InvalidInput => PJRT_ERROR_CODE_INVALID_ARGUMENT,
        io::ErrorKind::OutOfMemory => PJRT_ERROR_CODE_RESOURCE_EXHAUSTED,
        _ => PJRT_ERROR_CODE_INTERNAL,
    };
    rust_error(code, err.to_string())
}

fn checked_dims<'a>(ptr: *const i64, len: usize) -> Result<&'a [i64], *mut TTRustError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(invalid_argument("dims must not be null when num_dims > 0"));
    }
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

fn dims_i64_to_usize(dims: &[i64]) -> Result<Vec<usize>, *mut TTRustError> {
    dims.iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| invalid_argument("shape dimensions must be >= 0"))
        })
        .collect()
}

fn pjrt_buffer_type_to_dtype(buffer_type: i32) -> Result<DType, *mut TTRustError> {
    match buffer_type {
        PJRT_BUFFER_TYPE_S8 => Ok(DType::Int8),
        PJRT_BUFFER_TYPE_S32 => Ok(DType::Int32),
        PJRT_BUFFER_TYPE_U8 => Ok(DType::UInt8),
        PJRT_BUFFER_TYPE_U16 => Ok(DType::UInt16),
        PJRT_BUFFER_TYPE_U32 => Ok(DType::UInt32),
        PJRT_BUFFER_TYPE_F16 => Ok(DType::Float16),
        PJRT_BUFFER_TYPE_F32 => Ok(DType::Float32),
        PJRT_BUFFER_TYPE_BF16 => Ok(DType::Float16B),
        PJRT_BUFFER_TYPE_INVALID => Err(invalid_argument("invalid PJRT buffer type")),
        _ => Err(unimplemented(format!(
            "unsupported PJRT buffer type {buffer_type}"
        ))),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tt_rust_discovery_create() -> *mut TTRustDiscovery {
    let discovered = Device::discover();
    let mut owned = Vec::with_capacity(discovered.len());
    let mut raw = Vec::with_capacity(discovered.len());

    for info in discovered {
        let owned_info = OwnedDeviceInfo {
            arch: cstring_lossy(info.arch()),
            device_kind: cstring_lossy(info.device_kind()),
            device_debug_string: cstring_lossy(info.device_debug_string()),
            device_to_string: cstring_lossy(info.device_to_string()),
            memory_debug_string: cstring_lossy(info.memory_debug_string()),
            memory_to_string: cstring_lossy(info.memory_to_string()),
        };
        raw.push(TTDeviceInfo {
            id: info.id as i32,
            local_hardware_id: info.local_hardware_id() as i32,
            arch: owned_info.arch.as_ptr(),
            device_kind: owned_info.device_kind.as_ptr(),
            device_debug_string: owned_info.device_debug_string.as_ptr(),
            device_to_string: owned_info.device_to_string.as_ptr(),
            memory_debug_string: owned_info.memory_debug_string.as_ptr(),
            memory_to_string: owned_info.memory_to_string.as_ptr(),
        });
        owned.push(owned_info);
    }

    Box::into_raw(Box::new(TTRustDiscovery { _owned: owned, raw }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_discovery_destroy(discovery: *mut TTRustDiscovery) {
    if !discovery.is_null() {
        unsafe {
            drop(Box::from_raw(discovery));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_discovery_len(discovery: *const TTRustDiscovery) -> usize {
    unsafe { discovery.as_ref() }
        .map(|value| value.raw.len())
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_discovery_devices(
    discovery: *const TTRustDiscovery,
) -> *const TTDeviceInfo {
    unsafe { discovery.as_ref() }
        .map(|value| value.raw.as_ptr())
        .unwrap_or(ptr::null())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_buffer_from_host(
    local_hardware_id: usize,
    buffer_type: i32,
    dims: *const i64,
    num_dims: usize,
    data: *const c_void,
    data_len: usize,
    out_buffer: *mut *mut TTRustBufferHandle,
) -> *mut TTRustError {
    if out_buffer.is_null() {
        return invalid_argument("out_buffer must not be null");
    }
    let dtype = match pjrt_buffer_type_to_dtype(buffer_type) {
        Ok(dtype) => dtype,
        Err(err) => return err,
    };
    let dims_i64 = match checked_dims(dims, num_dims) {
        Ok(dims) => dims.to_vec(),
        Err(err) => return err,
    };
    let shape = match dims_i64_to_usize(&dims_i64) {
        Ok(shape) => shape,
        Err(err) => return err,
    };
    if data_len > 0 && data.is_null() {
        return invalid_argument("data must not be null");
    }
    let data = if data_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(data.cast::<u8>(), data_len) }
    };

    let mut device = match Device::open(local_hardware_id) {
        Ok(device) => device,
        Err(err) => return io_error(err),
    };
    let dram_buffer = match device.alloc_write(data, dtype, &shape, "pjrt") {
        Ok(buffer) => buffer,
        Err(err) => return io_error(err),
    };

    unsafe {
        *out_buffer = Box::into_raw(Box::new(TTRustBufferHandle {
            local_hardware_id,
            buffer_type,
            dims: dims_i64,
            dram_buffer: Some(dram_buffer),
            deleted: false,
        }));
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_buffer_destroy(buffer: *mut TTRustBufferHandle) {
    if !buffer.is_null() {
        unsafe {
            drop(Box::from_raw(buffer));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_buffer_delete(buffer: *mut TTRustBufferHandle) {
    if let Some(buffer) = unsafe { buffer.as_mut() } {
        buffer.deleted = true;
        buffer.dram_buffer = None;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_buffer_is_deleted(buffer: *const TTRustBufferHandle) -> bool {
    unsafe { buffer.as_ref() }
        .map(|buffer| buffer.deleted)
        .unwrap_or(true)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_buffer_size(buffer: *const TTRustBufferHandle) -> usize {
    unsafe { buffer.as_ref() }
        .and_then(|buffer| buffer.dram_buffer.as_ref())
        .map(DramBuffer::size)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_buffer_read(
    buffer: *const TTRustBufferHandle,
    dst: *mut c_void,
    dst_len: usize,
    out_len: *mut usize,
) -> *mut TTRustError {
    let Some(buffer) = (unsafe { buffer.as_ref() }) else {
        return invalid_argument("buffer must not be null");
    };
    let Some(dram_buffer) = buffer.dram_buffer.as_ref() else {
        return failed_precondition("buffer has been deleted");
    };
    if out_len.is_null() {
        return invalid_argument("out_len must not be null");
    }

    let mut device = match Device::open(buffer.local_hardware_id) {
        Ok(device) => device,
        Err(err) => return io_error(err),
    };
    let data = match device.dram_read(dram_buffer) {
        Ok(data) => data,
        Err(err) => return io_error(err),
    };

    unsafe { *out_len = data.len() };

    if dst.is_null() {
        return ptr::null_mut();
    }
    if dst_len < data.len() {
        return resource_exhausted("destination buffer is too small");
    }
    unsafe {
        ptr::copy_nonoverlapping(data.as_ptr(), dst.cast::<u8>(), data.len());
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn tt_rust_error_destroy(error: *mut TTRustError) {
    if let Some(error) = unsafe { error.as_mut() } {
        if !error.message.is_null() {
            let _ = unsafe { CString::from_raw(error.message) };
            error.message = ptr::null_mut();
        }
    }
    if !error.is_null() {
        unsafe {
            drop(Box::from_raw(error));
        }
    }
}
