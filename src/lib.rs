#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::missing_safety_doc
)]

pub mod compiler;
mod cq;
pub mod device;
pub mod dispatch;
pub mod dram;
mod executable;
mod hw;
mod kernels;
mod linux;
mod log;
mod mlir_frontend;

use device::Device;
use dram::{DType, DramBuffer};
#[cfg(libtt_mlir_frontend)]
use executable_proto::tt::analysis_result::Status as MlirAnalysisStatus;
use log::log;
use std::ffi::{c_char, CString};
use std::io;
use std::mem::size_of;
use std::ptr;
use std::slice;
use std::sync::Once;

include!("pjrt_bindings.rs");

#[repr(C)]
pub struct PJRT_Error {
    code: PJRT_Error_Code,
    message: CString,
}

#[repr(C)]
pub struct PJRT_DeviceDescription {
    id: i32,
    process_index: i32,
    device_kind: CString,
    debug_string: CString,
    to_string: CString,
}

#[repr(C)]
pub struct PJRT_TopologyDescription {
    platform_name: CString,
    platform_version: CString,
    device_description_ptrs: Vec<*mut PJRT_DeviceDescription>,
}

#[repr(C)]
pub struct PJRT_Memory {
    id: i32,
    kind: CString,
    debug_string: CString,
    to_string: CString,
    device_ptrs: Vec<*mut PJRT_Device>,
}

#[repr(C)]
pub struct PJRT_Device {
    id: i32,
    local_hardware_id: i32,
    description: *mut PJRT_DeviceDescription,
    addressable: bool,
    default_memory: *mut PJRT_Memory,
    memory_ptrs: Vec<*mut PJRT_Memory>,
    runtime: Device,
}

#[repr(C)]
pub struct PJRT_Event {
    error: Option<(PJRT_Error_Code, String)>,
}

#[derive(Clone)]
#[repr(C)]
pub struct PJRT_Buffer {
    buffer_type: PJRT_Buffer_Type,
    dims: Vec<i64>,
    device: *mut PJRT_Device,
    memory: *mut PJRT_Memory,
    local_hardware_id: usize,
    dram_buffer: Option<DramBuffer>,
    deleted: bool,
}

#[repr(C)]
pub struct PJRT_Layouts_MemoryLayout {
    serialized: CString,
}

#[repr(C)]
pub struct PJRT_Layouts_SerializedLayout {
    serialized: CString,
}

#[derive(Clone)]
struct ExecutableMetadata {
    name: CString,
    fingerprint: CString,
    num_outputs: usize,
    output_types: Vec<PJRT_Buffer_Type>,
    output_dims: Vec<i64>,
    output_dim_sizes: Vec<usize>,
    _output_memory_kinds: Vec<CString>,
    output_memory_kind_ptrs: Vec<*const c_char>,
    output_memory_kind_sizes: Vec<usize>,
    executable: Option<executable::Executable>,
}

#[repr(C)]
pub struct PJRT_Executable {
    metadata: ExecutableMetadata,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable {
    metadata: ExecutableMetadata,
    addressable_devices: Vec<*mut PJRT_Device>,
    deleted: bool,
}

#[repr(C)]
pub struct PJRT_Client {
    platform_name: CString,
    platform_version: CString,
    topology: PJRT_TopologyDescription,
    device_descriptions: Vec<PJRT_DeviceDescription>,
    memories: Vec<PJRT_Memory>,
    devices: Vec<PJRT_Device>,
    device_ptrs: Vec<*mut PJRT_Device>,
    addressable_device_ptrs: Vec<*mut PJRT_Device>,
    memory_ptrs: Vec<*mut PJRT_Memory>,
}

unsafe impl Sync for PJRT_Api {}

#[cfg(libtt_mlir_frontend)]
const EXECUTABLE_NAME: &str = "tt.executable.v1";

impl PJRT_Client {
    fn new() -> Self {
        Self::new_with_devices(Device::discover())
    }

    fn new_with_devices(discovered: Vec<Device>) -> Self {
        let mut device_descriptions = Vec::with_capacity(discovered.len());

        for info in &discovered {
            device_descriptions.push(PJRT_DeviceDescription {
                id: info.id as i32,
                process_index: 0,
                device_kind: cstring_lossy(info.device_kind()),
                debug_string: cstring_lossy(info.device_debug_string()),
                to_string: cstring_lossy(info.device_to_string()),
            });
        }

        let mut memories = Vec::with_capacity(discovered.len());
        for info in &discovered {
            memories.push(PJRT_Memory {
                id: info.id as i32,
                kind: cstring_lossy("dram"),
                debug_string: cstring_lossy(info.memory_debug_string()),
                to_string: cstring_lossy(info.memory_to_string()),
                device_ptrs: Vec::with_capacity(1),
            });
        }

        let mut memory_ptrs = Vec::with_capacity(memories.len());
        for memory in &mut memories {
            memory_ptrs.push(memory as *mut PJRT_Memory);
        }

        let mut devices = Vec::with_capacity(discovered.len());
        for (index, info) in discovered.into_iter().enumerate() {
            let description = &mut device_descriptions[index] as *mut PJRT_DeviceDescription;
            let default_memory = memory_ptrs[index];
            devices.push(PJRT_Device {
                id: info.id as i32,
                local_hardware_id: info.local_hardware_id as i32,
                description,
                addressable: true,
                default_memory,
                memory_ptrs: vec![default_memory],
                runtime: info,
            });
        }

        let mut device_ptrs = Vec::with_capacity(devices.len());
        for device in &mut devices {
            device_ptrs.push(device as *mut PJRT_Device);
        }
        let addressable_device_ptrs = device_ptrs.clone();
        for (index, memory) in memories.iter_mut().enumerate() {
            memory.device_ptrs.push(device_ptrs[index]);
        }

        let topology = PJRT_TopologyDescription {
            platform_name: cstring_lossy("tt"),
            platform_version: cstring_lossy(format!("libtt {}", env!("CARGO_PKG_VERSION"))),
            device_description_ptrs: device_descriptions
                .iter_mut()
                .map(|description| description as *mut PJRT_DeviceDescription)
                .collect(),
        };

        Self {
            platform_name: cstring_lossy("tt"),
            platform_version: cstring_lossy(format!("libtt {}", env!("CARGO_PKG_VERSION"))),
            topology,
            device_descriptions,
            memories,
            devices,
            device_ptrs,
            addressable_device_ptrs,
            memory_ptrs,
        }
    }
}

fn cstring_lossy<S: AsRef<str>>(value: S) -> CString {
    let sanitized = value.as_ref().replace('\0', "?");
    CString::new(sanitized).expect("CString::new should succeed after sanitizing NULs")
}

fn pjrt_error(message: impl AsRef<str>, code: PJRT_Error_Code) -> *mut PJRT_Error {
    Box::into_raw(Box::new(PJRT_Error {
        code,
        message: cstring_lossy(message.as_ref()),
    }))
}

fn invalid_argument(message: impl AsRef<str>) -> *mut PJRT_Error {
    pjrt_error(message, PJRT_Error_Code::PJRT_Error_Code_INVALID_ARGUMENT)
}

fn unimplemented(message: impl AsRef<str>) -> *mut PJRT_Error {
    pjrt_error(message, PJRT_Error_Code::PJRT_Error_Code_UNIMPLEMENTED)
}

fn resource_exhausted(message: impl AsRef<str>) -> *mut PJRT_Error {
    pjrt_error(message, PJRT_Error_Code::PJRT_Error_Code_RESOURCE_EXHAUSTED)
}

fn io_error(err: io::Error) -> *mut PJRT_Error {
    let code = match err.kind() {
        io::ErrorKind::InvalidInput => PJRT_Error_Code::PJRT_Error_Code_INVALID_ARGUMENT,
        io::ErrorKind::TimedOut => PJRT_Error_Code::PJRT_Error_Code_DEADLINE_EXCEEDED,
        io::ErrorKind::OutOfMemory => PJRT_Error_Code::PJRT_Error_Code_RESOURCE_EXHAUSTED,
        _ => PJRT_Error_Code::PJRT_Error_Code_INTERNAL,
    };
    pjrt_error(err.to_string(), code)
}

fn failed_precondition(message: impl AsRef<str>) -> *mut PJRT_Error {
    pjrt_error(
        message,
        PJRT_Error_Code::PJRT_Error_Code_FAILED_PRECONDITION,
    )
}

fn ready_event() -> *mut PJRT_Event {
    Box::into_raw(Box::new(PJRT_Event { error: None }))
}

unsafe extern "C" fn noop_device_attributes_deleter(
    device_attributes: *mut PJRT_Device_Attributes,
) {
    if !device_attributes.is_null() {
        unsafe {
            drop(Box::from_raw(device_attributes));
        }
    }
}

unsafe extern "C" fn noop_serialized_device_assignment_deleter(
    _device_assignment: *mut PJRT_DeviceAssignmentSerialized,
) {
}

fn event_with_error(code: PJRT_Error_Code, message: impl Into<String>) -> *mut PJRT_Event {
    Box::into_raw(Box::new(PJRT_Event {
        error: Some((code, message.into())),
    }))
}

fn cloned_event_error(event: &PJRT_Event) -> *mut PJRT_Error {
    match &event.error {
        Some((code, message)) => pjrt_error(message, *code),
        None => ptr::null_mut(),
    }
}

unsafe fn checked_mut<'a, T>(ptr: *mut T, name: &str) -> Result<&'a mut T, *mut PJRT_Error> {
    // SAFETY: caller guarantees `ptr` originates from the C ABI.
    unsafe { ptr.as_mut() }.ok_or_else(|| invalid_argument(format!("{name} must not be null")))
}

unsafe fn checked_ref<'a, T>(ptr: *const T, name: &str) -> Result<&'a T, *mut PJRT_Error> {
    // SAFETY: caller guarantees `ptr` originates from the C ABI.
    unsafe { ptr.as_ref() }.ok_or_else(|| invalid_argument(format!("{name} must not be null")))
}

fn pjrt_buffer_type_to_dtype(buffer_type: PJRT_Buffer_Type) -> Result<DType, *mut PJRT_Error> {
    match buffer_type {
        PJRT_Buffer_Type::PJRT_Buffer_Type_S8 => Ok(DType::Int8),
        PJRT_Buffer_Type::PJRT_Buffer_Type_PRED => Ok(DType::UInt8),
        PJRT_Buffer_Type::PJRT_Buffer_Type_S32 => Ok(DType::Int32),
        PJRT_Buffer_Type::PJRT_Buffer_Type_U8 => Ok(DType::UInt8),
        PJRT_Buffer_Type::PJRT_Buffer_Type_U16 => Ok(DType::UInt16),
        PJRT_Buffer_Type::PJRT_Buffer_Type_U32 => Ok(DType::UInt32),
        PJRT_Buffer_Type::PJRT_Buffer_Type_F16 => Ok(DType::Float16),
        PJRT_Buffer_Type::PJRT_Buffer_Type_F32 => Ok(DType::Float32),
        PJRT_Buffer_Type::PJRT_Buffer_Type_BF16 => Ok(DType::Float16B),
        PJRT_Buffer_Type::PJRT_Buffer_Type_INVALID => {
            Err(invalid_argument("invalid PJRT buffer type"))
        }
        _ => Err(unimplemented(format!(
            "unsupported PJRT buffer type {buffer_type:?}"
        ))),
    }
}

fn dtype_to_pjrt_buffer_type(dtype: DType) -> PJRT_Buffer_Type {
    match dtype {
        DType::Int8 => PJRT_Buffer_Type::PJRT_Buffer_Type_S8,
        DType::Int32 => PJRT_Buffer_Type::PJRT_Buffer_Type_S32,
        DType::UInt8 => PJRT_Buffer_Type::PJRT_Buffer_Type_U8,
        DType::UInt16 => PJRT_Buffer_Type::PJRT_Buffer_Type_U16,
        DType::UInt32 => PJRT_Buffer_Type::PJRT_Buffer_Type_U32,
        DType::Float16 => PJRT_Buffer_Type::PJRT_Buffer_Type_F16,
        DType::Float32 => PJRT_Buffer_Type::PJRT_Buffer_Type_F32,
        DType::Float16B => PJRT_Buffer_Type::PJRT_Buffer_Type_BF16,
    }
}

fn dims_i64_to_usize(dims: &[i64]) -> Result<Vec<usize>, *mut PJRT_Error> {
    dims.iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| invalid_argument("shape dimensions must be >= 0"))
        })
        .collect()
}

unsafe fn checked_i64_slice<'a>(
    ptr: *const i64,
    len: usize,
    field: &str,
) -> Result<&'a [i64], *mut PJRT_Error> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(invalid_argument(format!(
            "{field} must not be null when length > 0"
        )));
    }
    // SAFETY: caller owns `ptr` for `len` elements during the call.
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

fn host_byte_size(dtype: DType, dims: &[usize]) -> Result<usize, *mut PJRT_Error> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .and_then(|elements| elements.checked_mul(dtype.bytes_per_element()))
        .ok_or_else(|| resource_exhausted("host buffer size overflow"))
}

fn padded_host_data(
    data: &[u8],
    dtype: DType,
    logical_shape: &[usize],
    allocation_shape: &[usize],
) -> Result<Option<Vec<u8>>, *mut PJRT_Error> {
    if logical_shape == allocation_shape {
        return Ok(None);
    }

    let allocation_size = host_byte_size(dtype, allocation_shape)?;
    if data.len() > allocation_size {
        return Err(invalid_argument(
            "logical buffer is larger than allocation buffer",
        ));
    }
    let mut padded = vec![0u8; allocation_size];
    if logical_shape.len() < 2 {
        padded[..data.len()].copy_from_slice(data);
        return Ok(Some(padded));
    }

    copy_between_host_shapes(
        data,
        &mut padded,
        dtype,
        logical_shape,
        allocation_shape,
        logical_shape,
    )?;
    Ok(Some(padded))
}

fn copy_between_host_shapes(
    source: &[u8],
    target: &mut [u8],
    dtype: DType,
    source_shape: &[usize],
    target_shape: &[usize],
    copy_shape: &[usize],
) -> Result<(), *mut PJRT_Error> {
    if source_shape.len() != target_shape.len() || source_shape.len() != copy_shape.len() {
        return Err(invalid_argument(
            "rank must match when copying between padded host shapes",
        ));
    }
    if copy_shape.len() < 2 {
        return Err(invalid_argument(
            "padded host shape copy requires rank >= 2",
        ));
    }
    if copy_shape
        .iter()
        .zip(source_shape.iter().zip(target_shape))
        .any(|(copy, (source, target))| copy > source || copy > target)
    {
        return Err(invalid_argument(
            "copy shape exceeds source or target shape",
        ));
    }

    let source_size = host_byte_size(dtype, source_shape)?;
    let target_size = host_byte_size(dtype, target_shape)?;
    if source.len() != source_size || target.len() != target_size {
        return Err(invalid_argument(
            "host buffer size does not match row-major shape",
        ));
    }

    let rank = copy_shape.len();
    let copy_rows = copy_shape[rank - 2];
    let copy_cols = copy_shape[rank - 1];
    let source_rows = source_shape[rank - 2];
    let source_cols = source_shape[rank - 1];
    let target_rows = target_shape[rank - 2];
    let target_cols = target_shape[rank - 1];
    let batch = copy_shape[..rank - 2]
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| resource_exhausted("shape dimensions overflow"))?;
    let bytes_per_element = dtype.bytes_per_element();
    let row_bytes = copy_cols
        .checked_mul(bytes_per_element)
        .ok_or_else(|| resource_exhausted("shape dimensions overflow"))?;

    for batch_index in 0..batch {
        for row in 0..copy_rows {
            let source_element = (batch_index * source_rows + row)
                .checked_mul(source_cols)
                .ok_or_else(|| resource_exhausted("shape dimensions overflow"))?;
            let target_element = (batch_index * target_rows + row)
                .checked_mul(target_cols)
                .ok_or_else(|| resource_exhausted("shape dimensions overflow"))?;
            let source_start = source_element
                .checked_mul(bytes_per_element)
                .ok_or_else(|| resource_exhausted("shape dimensions overflow"))?;
            let target_start = target_element
                .checked_mul(bytes_per_element)
                .ok_or_else(|| resource_exhausted("shape dimensions overflow"))?;
            let source_end = source_start
                .checked_add(row_bytes)
                .ok_or_else(|| resource_exhausted("shape dimensions overflow"))?;
            let target_end = target_start
                .checked_add(row_bytes)
                .ok_or_else(|| resource_exhausted("shape dimensions overflow"))?;
            target[target_start..target_end].copy_from_slice(&source[source_start..source_end]);
        }
    }

    Ok(())
}

fn validate_dense_row_major_strides(
    dtype: DType,
    dims: &[usize],
    byte_strides: *const i64,
    num_byte_strides: usize,
) -> Result<(), *mut PJRT_Error> {
    if byte_strides.is_null() && num_byte_strides == 0 {
        return Ok(());
    }
    if num_byte_strides != dims.len() {
        return Err(invalid_argument(
            "num_byte_strides must match num_dims for strided host buffers",
        ));
    }
    let strides = unsafe { checked_i64_slice(byte_strides, num_byte_strides, "byte_strides") }?;
    let mut expected = dtype.bytes_per_element();
    for (&dim, &stride) in dims.iter().rev().zip(strides.iter().rev()) {
        let stride =
            usize::try_from(stride).map_err(|_| invalid_argument("byte strides must be >= 0"))?;
        if stride != expected {
            return Err(unimplemented(
                "only dense row-major host buffers are supported",
            ));
        }
        expected = expected
            .checked_mul(dim.max(1))
            .ok_or_else(|| resource_exhausted("byte stride overflow"))?;
    }
    Ok(())
}

fn event_for_buffer(buffer: &PJRT_Buffer) -> *mut PJRT_Event {
    if buffer.deleted {
        event_with_error(
            PJRT_Error_Code::PJRT_Error_Code_FAILED_PRECONDITION,
            "buffer has been deleted",
        )
    } else {
        ready_event()
    }
}

fn read_buffer_bytes(buffer: &PJRT_Buffer) -> Result<Vec<u8>, *mut PJRT_Error> {
    let Some(dram_buffer) = buffer.dram_buffer.as_ref() else {
        return Err(pjrt_error(
            "buffer has been deleted",
            PJRT_Error_Code::PJRT_Error_Code_FAILED_PRECONDITION,
        ));
    };
    with_device_ptr(buffer.device, |device| {
        device.dram_read(dram_buffer).map_err(io_error)
    })
}

fn read_buffer_logical_bytes(buffer: &PJRT_Buffer) -> Result<Vec<u8>, *mut PJRT_Error> {
    let dtype = pjrt_buffer_type_to_dtype(buffer.buffer_type)?;
    let dims = dims_i64_to_usize(&buffer.dims)?;
    let byte_size = host_byte_size(dtype, &dims)?;
    let allocation_shape = buffer
        .dram_buffer
        .as_ref()
        .map(|dram_buffer| dram_buffer.shape.clone())
        .ok_or_else(|| {
            pjrt_error(
                "buffer has been deleted",
                PJRT_Error_Code::PJRT_Error_Code_FAILED_PRECONDITION,
            )
        })?;
    let data = read_buffer_bytes(buffer)?;
    if data.len() == byte_size {
        return Ok(data);
    }
    if buffer.dims.len() < 2 && data.len() >= byte_size {
        let mut data = data;
        data.truncate(byte_size);
        return Ok(data);
    }
    if let Some(data) = crop_padded_host_data(&data, dtype, &dims, &allocation_shape)? {
        return Ok(data);
    }
    Err(pjrt_error(
        format!(
            "readback byte size {} does not match buffer byte size {}",
            data.len(),
            byte_size
        ),
        PJRT_Error_Code::PJRT_Error_Code_INTERNAL,
    ))
}

fn crop_padded_host_data(
    data: &[u8],
    dtype: DType,
    logical_shape: &[usize],
    allocation_shape: &[usize],
) -> Result<Option<Vec<u8>>, *mut PJRT_Error> {
    if logical_shape.len() < 2 || logical_shape.len() != allocation_shape.len() {
        return Ok(None);
    }

    let logical_size = host_byte_size(dtype, logical_shape)?;
    let allocation_size = host_byte_size(dtype, allocation_shape)?;
    if data.len() != allocation_size {
        return Ok(None);
    }

    let mut out = vec![0u8; logical_size];
    copy_between_host_shapes(
        data,
        &mut out,
        dtype,
        allocation_shape,
        logical_shape,
        logical_shape,
    )?;
    Ok(Some(out))
}

fn with_device<T>(
    pjrt_device: &mut PJRT_Device,
    f: impl FnOnce(&mut Device) -> Result<T, *mut PJRT_Error>,
) -> Result<T, *mut PJRT_Error> {
    f(&mut pjrt_device.runtime)
}

fn with_device_ptr<T>(
    pjrt_device: *mut PJRT_Device,
    f: impl FnOnce(&mut Device) -> Result<T, *mut PJRT_Error>,
) -> Result<T, *mut PJRT_Error> {
    let pjrt_device = unsafe { checked_mut(pjrt_device, "device") }?;
    with_device(pjrt_device, f)
}

fn c_api_string(ptr: *const c_char, len: usize, field: &str) -> Result<String, *mut PJRT_Error> {
    if len == 0 {
        return Ok(String::new());
    }
    if ptr.is_null() {
        return Err(invalid_argument(format!(
            "{field} must not be null when size > 0"
        )));
    }
    // SAFETY: caller owns `ptr` for `len` bytes during the call.
    let bytes = unsafe { slice::from_raw_parts(ptr.cast::<u8>(), len) };
    String::from_utf8(bytes.to_vec())
        .map_err(|_| invalid_argument(format!("{field} must be valid UTF-8")))
}

fn validate_program_format(program: &PJRT_Program) -> Result<(), *mut PJRT_Error> {
    let format = c_api_string(program.format, program.format_size, "program.format")?;
    log(format!(
        "pjrt compile program format={format:?} code_size={}",
        program.code_size
    ));

    match format.as_str() {
        "mlir" | "stablehlo" => Ok(()),
        other => Err(unimplemented(format!(
            "unsupported program format {other:?}; supported formats are \"mlir\" and \"stablehlo\""
        ))),
    }
}

fn executable_metadata_from_program(
    program: &PJRT_Program,
) -> Result<ExecutableMetadata, *mut PJRT_Error> {
    #[cfg(libtt_mlir_frontend)]
    {
        let format = c_api_string(program.format, program.format_size, "program.format")?;
        if format == "mlir" || format == "stablehlo" {
            if let Some(analysis) = mlir_frontend::AnalysisHandle::analyze(
                program.format,
                program.format_size,
                program.code.cast_const(),
                program.code_size,
            ) {
                let analysis = executable::parse_analysis(analysis.bytes()).map_err(|message| {
                    pjrt_error(message, PJRT_Error_Code::PJRT_Error_Code_INTERNAL)
                })?;
                if analysis.status != MlirAnalysisStatus::Ok {
                    let message = if analysis.error_message.is_empty() {
                        format!("MLIR analysis failed with status {:?}", analysis.status)
                    } else {
                        analysis.error_message
                    };
                    return Err(match analysis.status {
                        MlirAnalysisStatus::ParseError => invalid_argument(message),
                        MlirAnalysisStatus::Unsupported => unimplemented(message),
                        _ => pjrt_error(message, PJRT_Error_Code::PJRT_Error_Code_INTERNAL),
                    });
                }

                return Ok(make_executable_metadata(
                    EXECUTABLE_NAME,
                    &analysis.outputs,
                    analysis.executable,
                ));
            }
        }
    }

    validate_program_format(program)?;
    Err(unimplemented(
        "MLIR compilation requires the libtt MLIR frontend build",
    ))
}

#[cfg(libtt_mlir_frontend)]
fn make_executable_metadata(
    name: &str,
    outputs: &[executable::ValueDesc],
    executable: Option<executable::Executable>,
) -> ExecutableMetadata {
    let output_memory_kinds = vec![cstring_lossy("dram"); outputs.len()];
    let output_memory_kind_ptrs = output_memory_kinds
        .iter()
        .map(|kind| kind.as_ptr())
        .collect::<Vec<_>>();
    let output_memory_kind_sizes = output_memory_kinds
        .iter()
        .map(|kind| kind.as_bytes().len())
        .collect::<Vec<_>>();
    let output_types = outputs
        .iter()
        .map(|output| output.element_type)
        .collect::<Vec<_>>();
    let output_dim_sizes = outputs
        .iter()
        .map(|output| output.dims.len())
        .collect::<Vec<_>>();
    let output_dims = outputs
        .iter()
        .flat_map(|output| output.dims.iter().copied())
        .collect::<Vec<_>>();
    let fingerprint = executable_fingerprint_string(name, outputs);
    ExecutableMetadata {
        name: cstring_lossy(name),
        fingerprint,
        num_outputs: outputs.len(),
        output_types,
        output_dims,
        output_dim_sizes,
        _output_memory_kinds: output_memory_kinds,
        output_memory_kind_ptrs,
        output_memory_kind_sizes,
        executable,
    }
}

fn make_executable(metadata: ExecutableMetadata) -> PJRT_Executable {
    PJRT_Executable { metadata }
}

fn make_loaded_executable(
    metadata: ExecutableMetadata,
    addressable_devices: Vec<*mut PJRT_Device>,
) -> PJRT_LoadedExecutable {
    PJRT_LoadedExecutable {
        metadata,
        addressable_devices,
        deleted: false,
    }
}

fn cloned_executable(executable: &PJRT_LoadedExecutable) -> PJRT_Executable {
    PJRT_Executable {
        metadata: executable.metadata.clone(),
    }
}

#[cfg(libtt_mlir_frontend)]
fn executable_fingerprint_string(name: &str, outputs: &[executable::ValueDesc]) -> CString {
    let outputs = outputs
        .iter()
        .map(|output| {
            let dims = output
                .dims
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join("x");
            format!("{}:{}", output.element_type as u32, dims)
        })
        .collect::<Vec<_>>()
        .join(",");
    cstring_lossy(&format!(
        "tt:executable_v1:name={name}:outputs={outputs}:v1"
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Error_Destroy(args: *mut PJRT_Error_Destroy_Args) {
    let Some(args) = (unsafe { args.as_mut() }) else {
        return;
    };
    if !args.error.is_null() {
        // SAFETY: `error` is allocated by `pjrt_error`.
        unsafe {
            drop(Box::from_raw(args.error));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Error_Message(args: *mut PJRT_Error_Message_Args) {
    let Some(args) = (unsafe { args.as_mut() }) else {
        return;
    };
    if let Some(error) = unsafe { args.error.as_ref() } {
        args.message = error.message.as_ptr();
        args.message_size = error.message.as_bytes().len();
    } else {
        args.message = ptr::null();
        args.message_size = 0;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Error_GetCode(args: *mut PJRT_Error_GetCode_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    args.code = unsafe { args.error.as_ref() }
        .map(|error| error.code)
        .unwrap_or(PJRT_Error_Code::PJRT_Error_Code_OK);
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Error_ForEachPayload(
    args: *mut PJRT_Error_ForEachPayload_Args,
) -> *mut PJRT_Error {
    let Ok(_args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    // libtt does not attach structured error payloads.
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Plugin_Initialize(
    args: *mut PJRT_Plugin_Initialize_Args,
) -> *mut PJRT_Error {
    if args.is_null() {
        return invalid_argument("args must not be null");
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Plugin_Attributes(
    args: *mut PJRT_Plugin_Attributes_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    args.attributes = ptr::null();
    args.num_attributes = 0;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Event_Destroy(args: *mut PJRT_Event_Destroy_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if !args.event.is_null() {
        // SAFETY: `event` is allocated by `ready_event` or `event_with_error`.
        unsafe {
            drop(Box::from_raw(args.event));
        }
        args.event = ptr::null_mut();
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Event_IsReady(args: *mut PJRT_Event_IsReady_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(_event) = (unsafe { checked_ref(args.event, "event") }) else {
        return invalid_argument("event must not be null");
    };
    args.is_ready = true;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Event_Error(args: *mut PJRT_Event_Error_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(event) = (unsafe { checked_ref(args.event, "event") }) else {
        return invalid_argument("event must not be null");
    };
    cloned_event_error(event)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Event_Await(args: *mut PJRT_Event_Await_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(event) = (unsafe { checked_ref(args.event, "event") }) else {
        return invalid_argument("event must not be null");
    };
    cloned_event_error(event)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Event_OnReady(args: *mut PJRT_Event_OnReady_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(event) = (unsafe { checked_ref(args.event, "event") }) else {
        return invalid_argument("event must not be null");
    };
    let Some(callback) = args.callback else {
        return invalid_argument("callback must not be null");
    };
    // SAFETY: `callback` originates from the caller and accepts ownership of the error.
    unsafe {
        callback(cloned_event_error(event), args.user_arg);
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_Create(args: *mut PJRT_Client_Create_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let client = Box::new(PJRT_Client::new());
    let client_ptr = Box::into_raw(client);
    args.client = client_ptr;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_Destroy(args: *mut PJRT_Client_Destroy_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if !args.client.is_null() {
        // SAFETY: `client` is allocated by `TT_Client_Create`.
        unsafe {
            drop(Box::from_raw(args.client));
        }
        args.client = ptr::null_mut();
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_PlatformName(
    args: *mut PJRT_Client_PlatformName_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.platform_name = client.platform_name.as_ptr();
    args.platform_name_size = client.platform_name.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_ProcessIndex(
    args: *mut PJRT_Client_ProcessIndex_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.client.is_null() {
        return invalid_argument("client must not be null");
    }
    args.process_index = 0;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_PlatformVersion(
    args: *mut PJRT_Client_PlatformVersion_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.platform_version = client.platform_version.as_ptr();
    args.platform_version_size = client.platform_version.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_TopologyDescription(
    args: *mut PJRT_Client_TopologyDescription_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_mut(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.topology = &mut client.topology;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_Devices(args: *mut PJRT_Client_Devices_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.devices = if client.device_ptrs.is_empty() {
        ptr::null()
    } else {
        client.device_ptrs.as_ptr()
    };
    args.num_devices = client.device_ptrs.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_AddressableDevices(
    args: *mut PJRT_Client_AddressableDevices_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.addressable_devices = if client.addressable_device_ptrs.is_empty() {
        ptr::null()
    } else {
        client.addressable_device_ptrs.as_ptr()
    };
    args.num_addressable_devices = client.addressable_device_ptrs.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_LookupDevice(
    args: *mut PJRT_Client_LookupDevice_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.device = client
        .device_ptrs
        .iter()
        .copied()
        .find(|device| unsafe { device.as_ref() }.is_some_and(|device| device.id == args.id))
        .unwrap_or(ptr::null_mut());
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_LookupAddressableDevice(
    args: *mut PJRT_Client_LookupAddressableDevice_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.addressable_device = client
        .addressable_device_ptrs
        .iter()
        .copied()
        .find(|device| {
            unsafe { device.as_ref() }
                .is_some_and(|device| device.local_hardware_id == args.local_hardware_id)
        })
        .unwrap_or(ptr::null_mut());
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_AddressableMemories(
    args: *mut PJRT_Client_AddressableMemories_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.addressable_memories = if client.memory_ptrs.is_empty() {
        ptr::null()
    } else {
        client.memory_ptrs.as_ptr()
    };
    args.num_addressable_memories = client.memory_ptrs.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_Compile(args: *mut PJRT_Client_Compile_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    let Ok(program) = (unsafe { checked_ref(args.program, "program") }) else {
        return invalid_argument("program must not be null");
    };
    let metadata = match executable_metadata_from_program(program) {
        Ok(compiled) => compiled,
        Err(err) => return err,
    };
    args.executable = Box::into_raw(Box::new(make_loaded_executable(
        metadata,
        client.addressable_device_ptrs.clone(),
    )));
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Compile(args: *mut PJRT_Compile_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(program) = (unsafe { checked_ref(args.program, "program") }) else {
        return invalid_argument("program must not be null");
    };
    let metadata = match executable_metadata_from_program(program) {
        Ok(compiled) => compiled,
        Err(err) => return err,
    };
    args.executable = Box::into_raw(Box::new(make_executable(metadata)));
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_DefaultDeviceAssignment(
    args: *mut PJRT_Client_DefaultDeviceAssignment_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    if args.num_replicas < 0 || args.num_partitions < 0 {
        return invalid_argument("num_replicas and num_partitions must be >= 0");
    }
    let required = usize::try_from(args.num_replicas)
        .ok()
        .and_then(|replicas| {
            usize::try_from(args.num_partitions)
                .ok()
                .and_then(|partitions| replicas.checked_mul(partitions))
        });
    let Some(required) = required else {
        return invalid_argument(
            "default device assignment size overflowed num_replicas * num_partitions",
        );
    };
    if args.default_assignment_size < required {
        return invalid_argument("default_assignment buffer is too small");
    }
    if required > 0 && args.default_assignment.is_null() {
        return invalid_argument("default_assignment must not be null");
    }
    if required > client.device_ptrs.len() {
        return invalid_argument("not enough devices for requested assignment");
    }
    for index in 0..required {
        // SAFETY: caller owns `default_assignment` for `default_assignment_size` entries.
        unsafe {
            *args.default_assignment.add(index) = index as i32;
        }
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_Destroy(
    args: *mut PJRT_Executable_Destroy_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if !args.executable.is_null() {
        unsafe {
            drop(Box::from_raw(args.executable));
        }
        args.executable = ptr::null_mut();
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_Name(
    args: *mut PJRT_Executable_Name_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.executable_name = executable.metadata.name.as_ptr();
    args.executable_name_size = executable.metadata.name.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_NumReplicas(
    args: *mut PJRT_Executable_NumReplicas_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.executable.is_null() {
        return invalid_argument("executable must not be null");
    }
    args.num_replicas = 1;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_NumPartitions(
    args: *mut PJRT_Executable_NumPartitions_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.executable.is_null() {
        return invalid_argument("executable must not be null");
    }
    args.num_partitions = 1;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_OptimizedProgram(
    args: *mut PJRT_Executable_OptimizedProgram_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    if unsafe { checked_mut(args.program, "program") }.is_err() {
        return invalid_argument("program must not be null");
    }
    if executable.metadata.executable.is_some() {
        return pjrt_error(
            "optimized program serialization is not exposed",
            PJRT_Error_Code::PJRT_Error_Code_UNIMPLEMENTED,
        );
    }
    pjrt_error(
        "optimized program is not available for this executable",
        PJRT_Error_Code::PJRT_Error_Code_UNIMPLEMENTED,
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_Fingerprint(
    args: *mut PJRT_Executable_Fingerprint_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.executable_fingerprint = executable.metadata.fingerprint.as_ptr();
    args.executable_fingerprint_size = executable.metadata.fingerprint.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_GetCompiledMemoryStats(
    args: *mut PJRT_Executable_GetCompiledMemoryStats_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.executable.is_null() {
        return invalid_argument("executable must not be null");
    }
    args.generated_code_size_in_bytes = 0;
    args.argument_size_in_bytes = 0;
    args.output_size_in_bytes = 0;
    args.alias_size_in_bytes = 0;
    args.temp_size_in_bytes = 0;
    args.host_generated_code_size_in_bytes = 0;
    args.host_argument_size_in_bytes = 0;
    args.host_output_size_in_bytes = 0;
    args.host_alias_size_in_bytes = 0;
    args.host_temp_size_in_bytes = 0;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_NumOutputs(
    args: *mut PJRT_Executable_NumOutputs_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.num_outputs = executable.metadata.num_outputs;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_OutputElementTypes(
    args: *mut PJRT_Executable_OutputElementTypes_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.output_types = executable.metadata.output_types.as_ptr().cast_mut();
    args.num_output_types = executable.metadata.output_types.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_OutputDimensions(
    args: *mut PJRT_Executable_OutputDimensions_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.dims = executable.metadata.output_dims.as_ptr();
    args.dim_sizes = executable.metadata.output_dim_sizes.as_ptr();
    args.num_outputs = executable.metadata.output_dim_sizes.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Executable_OutputMemoryKinds(
    args: *mut PJRT_Executable_OutputMemoryKinds_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.memory_kinds = executable.metadata.output_memory_kind_ptrs.as_ptr();
    args.memory_kind_sizes = executable.metadata.output_memory_kind_sizes.as_ptr();
    args.num_outputs = executable.metadata.output_memory_kind_ptrs.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_LoadedExecutable_Destroy(
    args: *mut PJRT_LoadedExecutable_Destroy_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if !args.executable.is_null() {
        unsafe {
            drop(Box::from_raw(args.executable));
        }
        args.executable = ptr::null_mut();
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_LoadedExecutable_GetExecutable(
    args: *mut PJRT_LoadedExecutable_GetExecutable_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(loaded) = (unsafe { checked_ref(args.loaded_executable, "loaded_executable") }) else {
        return invalid_argument("loaded_executable must not be null");
    };
    args.executable = Box::into_raw(Box::new(cloned_executable(loaded)));
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_LoadedExecutable_GetDeviceAssignment(
    args: *mut PJRT_LoadedExecutable_GetDeviceAssignment_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.executable.is_null() {
        return invalid_argument("executable must not be null");
    }
    args.serialized_bytes = ptr::null();
    args.serialized_bytes_size = 0;
    args.serialized_device_assignment = ptr::null_mut();
    args.serialized_device_assignment_deleter = Some(noop_serialized_device_assignment_deleter);
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_LoadedExecutable_AddressableDevices(
    args: *mut PJRT_LoadedExecutable_AddressableDevices_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.addressable_devices = if executable.addressable_devices.is_empty() {
        ptr::null()
    } else {
        executable.addressable_devices.as_ptr()
    };
    args.num_addressable_devices = executable.addressable_devices.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_LoadedExecutable_Delete(
    args: *mut PJRT_LoadedExecutable_Delete_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_mut(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    executable.deleted = true;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_LoadedExecutable_IsDeleted(
    args: *mut PJRT_LoadedExecutable_IsDeleted_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.is_deleted = executable.deleted;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_LoadedExecutable_Fingerprint(
    args: *mut PJRT_LoadedExecutable_Fingerprint_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    args.executable_fingerprint = executable.metadata.fingerprint.as_ptr();
    args.executable_fingerprint_size = executable.metadata.fingerprint.as_bytes().len();
    ptr::null_mut()
}

fn device_buffer_for_value<'a>(
    values: &'a [Option<PJRT_Buffer>],
    value_id: u32,
    field: &str,
) -> Result<&'a PJRT_Buffer, *mut PJRT_Error> {
    let index = value_id as usize;
    values
        .get(index)
        .and_then(|value| value.as_ref())
        .ok_or_else(|| invalid_argument(format!("{field} value id {value_id} is not available")))
}

fn eltwise_input<'a>(
    values: &'a [Option<PJRT_Buffer>],
    plan: &'a executable::Executable,
    value_id: u32,
    expected_dtype: DType,
    field: &str,
) -> Result<kernels::binary_eltwise::EltwiseInput<'a>, *mut PJRT_Error> {
    let index = value_id as usize;
    if let Some(buffer) = values.get(index).and_then(|value| value.as_ref()) {
        let Some(dram_buffer) = buffer.dram_buffer.as_ref() else {
            return Err(failed_precondition(format!(
                "TT executable {field} buffer has no device allocation"
            )));
        };
        return Ok(kernels::binary_eltwise::EltwiseInput::Dram(dram_buffer));
    }
    for op in &plan.ops {
        if let executable::Op::Constant {
            packed_value,
            output_id,
        } = op
        {
            if *output_id == value_id {
                let desc = plan.values.get(index).ok_or_else(|| {
                    invalid_argument(format!(
                        "{field} constant value id {value_id} is out of bounds"
                    ))
                })?;
                let dtype = pjrt_buffer_type_to_dtype(desc.element_type)?;
                if dtype != expected_dtype {
                    return Err(unimplemented(format!(
                        "{field} constant value id {value_id} has type {:?}; expected {expected_dtype:?}",
                        desc.element_type
                    )));
                }
                return Ok(kernels::binary_eltwise::EltwiseInput::Constant(
                    *packed_value,
                ));
            }
        }
    }
    Err(invalid_argument(format!(
        "{field} value id {value_id} is not available"
    )))
}

fn select_value_input<'a>(
    values: &'a [Option<PJRT_Buffer>],
    plan: &'a executable::Executable,
    value_id: u32,
    expected_dtype: DType,
    field: &str,
) -> Result<kernels::select::SelectInput<'a>, *mut PJRT_Error> {
    match eltwise_input(values, plan, value_id, expected_dtype, field)? {
        kernels::binary_eltwise::EltwiseInput::Dram(buffer) => {
            Ok(kernels::select::SelectInput::Dram(buffer))
        }
        kernels::binary_eltwise::EltwiseInput::Constant(value) => {
            Ok(kernels::select::SelectInput::Constant(value))
        }
    }
}

struct OutputContext {
    device: *mut PJRT_Device,
    memory: *mut PJRT_Memory,
    local_hardware_id: usize,
}

fn store_output_buffer(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    output_id: u32,
    expected_dims: Vec<i64>,
    dram_buffer: DramBuffer,
    context: &OutputContext,
    op: &str,
) -> Result<(), *mut PJRT_Error> {
    let output_index = output_id as usize;
    let expected = plan.values.get(output_index).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable {op} output id {output_id} is out of bounds"
        ))
    })?;
    if expected.dims != expected_dims {
        return Err(invalid_argument(format!(
            "TT executable {op} output shape mismatch: expected {:?}, got {:?}",
            expected.dims, expected_dims
        )));
    }
    let expected_dtype = pjrt_buffer_type_to_dtype(expected.element_type)?;
    if dram_buffer.dtype != expected_dtype {
        return Err(invalid_argument(format!(
            "TT executable {op} output dtype mismatch: expected {:?}, got {:?}",
            expected.element_type, dram_buffer.dtype
        )));
    }
    let logical_shape = dims_i64_to_usize(&expected.dims)?;
    let allocation_shape = dram::tiled_allocation_shape(&logical_shape).map_err(io_error)?;
    if dram_buffer.shape != allocation_shape {
        return Err(invalid_argument(format!(
            "TT executable {op} output allocation shape mismatch: expected {:?}, got {:?}",
            allocation_shape, dram_buffer.shape
        )));
    }
    values[output_index] = Some(PJRT_Buffer {
        buffer_type: expected.element_type,
        dims: expected.dims.clone(),
        device: context.device,
        memory: context.memory,
        local_hardware_id: context.local_hardware_id,
        dram_buffer: Some(dram_buffer),
        deleted: false,
    });
    Ok(())
}

fn execute_binary_eltwise(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    op: kernels::binary_eltwise::BinaryEltwiseOp,
    input_ids: [u32; 2],
    output_id: u32,
    op_name: &str,
) -> Result<(), *mut PJRT_Error> {
    let lhs_field = format!("{op_name}.lhs");
    let rhs_field = format!("{op_name}.rhs");
    let lhs_desc = plan
        .values
        .get(input_ids[0] as usize)
        .ok_or_else(|| invalid_argument(format!("{lhs_field} value id is out of bounds")))?;
    let rhs_desc = plan
        .values
        .get(input_ids[1] as usize)
        .ok_or_else(|| invalid_argument(format!("{rhs_field} value id is out of bounds")))?;
    if lhs_desc.element_type != rhs_desc.element_type {
        return Err(invalid_argument(format!(
            "TT executable {op_name} input element types must match"
        )));
    }
    if lhs_desc.dims != rhs_desc.dims {
        return Err(invalid_argument(format!(
            "TT executable {op_name} input shapes must match"
        )));
    }
    let input_dtype = pjrt_buffer_type_to_dtype(lhs_desc.element_type)?;
    let expected_output_type = match op {
        kernels::binary_eltwise::BinaryEltwiseOp::Add
        | kernels::binary_eltwise::BinaryEltwiseOp::Subtract
        | kernels::binary_eltwise::BinaryEltwiseOp::Divide
        | kernels::binary_eltwise::BinaryEltwiseOp::Multiply
        | kernels::binary_eltwise::BinaryEltwiseOp::Power
        | kernels::binary_eltwise::BinaryEltwiseOp::Max => lhs_desc.element_type,
        kernels::binary_eltwise::BinaryEltwiseOp::Compare(_) => {
            if !matches!(input_dtype, DType::Float16B | DType::Float32 | DType::Int32) {
                return Err(unimplemented(format!(
                    "TT executable compare currently supports bf16, f32, and s32 inputs, got {:?}",
                    lhs_desc.element_type
                )));
            }
            PJRT_Buffer_Type::PJRT_Buffer_Type_PRED
        }
    };
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable {op_name} output id {output_id} is out of bounds"
        ))
    })?;
    if output_desc.element_type != expected_output_type {
        return Err(invalid_argument(format!(
            "TT executable {op_name} output must be {:?}, got {:?}",
            expected_output_type, output_desc.element_type
        )));
    }

    let output_dims = lhs_desc.dims.clone();
    let shape = dims_i64_to_usize(&output_dims)?;
    let lhs_input = eltwise_input(values, plan, input_ids[0], input_dtype, &lhs_field)?;
    let rhs_input = eltwise_input(values, plan, input_ids[1], input_dtype, &rhs_field)?;
    let output_name = format!("pjrt_{op_name}");
    let output_dram = kernels::binary_eltwise::eltwise(
        device,
        op,
        lhs_input,
        rhs_input,
        input_dtype,
        &shape,
        output_name,
    )
    .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_dims,
        output_dram,
        context,
        op_name,
    )
}

fn execute_unary_eltwise(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    op: kernels::unary_eltwise::UnaryEltwiseOp,
    input_id: u32,
    output_id: u32,
    op_name: &str,
) -> Result<(), *mut PJRT_Error> {
    let input_field = format!("{op_name}.input");
    let input_desc = plan
        .values
        .get(input_id as usize)
        .ok_or_else(|| invalid_argument(format!("{input_field} value id is out of bounds")))?;
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable {op_name} output id {output_id} is out of bounds"
        ))
    })?;
    if output_desc.dims != input_desc.dims {
        return Err(invalid_argument(format!(
            "TT executable {op_name} output shape mismatch: expected {:?}, got {:?}",
            input_desc.dims, output_desc.dims
        )));
    }

    let input_dtype = pjrt_buffer_type_to_dtype(input_desc.element_type)?;
    let output_dtype = pjrt_buffer_type_to_dtype(output_desc.element_type)?;
    let output_dims = input_desc.dims.clone();
    let shape = dims_i64_to_usize(&output_dims)?;
    let input = eltwise_input(values, plan, input_id, input_dtype, &input_field)?;
    let output_name = format!("pjrt_{op_name}");
    let output_dram = kernels::unary_eltwise::eltwise(
        device,
        op,
        input,
        input_dtype,
        output_dtype,
        &shape,
        output_name,
    )
    .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_dims,
        output_dram,
        context,
        op_name,
    )
}

fn execute_reshape(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_id: u32,
    output_id: u32,
) -> Result<(), *mut PJRT_Error> {
    let input_desc = plan.values.get(input_id as usize).ok_or_else(|| {
        invalid_argument("TT executable reshape operand value id is out of bounds")
    })?;
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument("TT executable reshape output value id is out of bounds")
    })?;
    if input_desc.element_type != output_desc.element_type {
        return Err(invalid_argument(
            "TT executable reshape input and output element types must match",
        ));
    }

    let input_shape = dims_i64_to_usize(&input_desc.dims)?;
    let output_shape = dims_i64_to_usize(&output_desc.dims)?;

    let input = device_buffer_for_value(values, input_id, "reshape.operand")?;
    let Some(input_dram) = input.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable reshape operand buffer has no device allocation",
        ));
    };
    let dtype = pjrt_buffer_type_to_dtype(input_desc.element_type)?;
    let output_dims = output_desc.dims.clone();
    let output_dram = kernels::reshape::reshape(
        device,
        input_dram,
        &input_shape,
        &output_shape,
        dtype,
        "pjrt_reshape",
    )
    .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_dims,
        output_dram,
        context,
        "reshape",
    )
}

fn execute_slice(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_id: u32,
    output_id: u32,
    start_indices: &[i64],
    limit_indices: &[i64],
    strides: &[i64],
) -> Result<(), *mut PJRT_Error> {
    let input_desc = plan
        .values
        .get(input_id as usize)
        .ok_or_else(|| invalid_argument("TT executable slice operand value id is out of bounds"))?;
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable slice output id {output_id} is out of bounds"
        ))
    })?;
    if input_desc.element_type != output_desc.element_type {
        return Err(invalid_argument(
            "TT executable slice input and output element types must match",
        ));
    }

    let input_shape = dims_i64_to_usize(&input_desc.dims)?;
    let output_shape = dims_i64_to_usize(&output_desc.dims)?;
    let slice_plan = kernels::slice::SlicePlan::new(
        &input_shape,
        &output_shape,
        start_indices,
        limit_indices,
        strides,
    )
    .map_err(io_error)?;

    let input = device_buffer_for_value(values, input_id, "slice.operand")?;
    let Some(input_dram) = input.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable slice operand buffer has no device allocation",
        ));
    };
    let dtype = pjrt_buffer_type_to_dtype(input_desc.element_type)?;
    let output_dram = kernels::slice::slice(device, input_dram, &slice_plan, dtype, "pjrt_slice")
        .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_desc.dims.clone(),
        output_dram,
        context,
        "slice",
    )
}

fn execute_transpose(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_id: u32,
    output_id: u32,
    permutation: &[i64],
) -> Result<(), *mut PJRT_Error> {
    let input_desc = plan.values.get(input_id as usize).ok_or_else(|| {
        invalid_argument("TT executable transpose operand value id is out of bounds")
    })?;
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument("TT executable transpose output value id is out of bounds")
    })?;
    if input_desc.element_type != output_desc.element_type {
        return Err(invalid_argument(
            "TT executable transpose input and output element types must match",
        ));
    }
    if permutation != [1, 0] {
        return Err(unimplemented(
            "TT executable transpose currently only supports rank-2 permutation [1, 0]",
        ));
    }

    let input_shape = dims_i64_to_usize(&input_desc.dims)?;
    let output_shape = dims_i64_to_usize(&output_desc.dims)?;
    let input = device_buffer_for_value(values, input_id, "transpose.operand")?;
    let Some(input_dram) = input.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable transpose operand buffer has no device allocation",
        ));
    };
    let dtype = pjrt_buffer_type_to_dtype(input_desc.element_type)?;
    let output_dram = kernels::transpose::transpose_rank2(
        device,
        input_dram,
        &input_shape,
        &output_shape,
        dtype,
        "pjrt_transpose",
    )
    .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_desc.dims.clone(),
        output_dram,
        context,
        "transpose",
    )
}

fn execute_reduce(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_ids: &[u32],
    init_value_ids: &[u32],
    output_id: u32,
    dimensions: &[i64],
    reducer: executable::ReduceReducer,
) -> Result<(), *mut PJRT_Error> {
    let [input_id] = input_ids else {
        return Err(unimplemented(
            "TT executable reduce currently only supports one input",
        ));
    };
    let [init_value_id] = init_value_ids else {
        return Err(unimplemented(
            "TT executable reduce currently only supports one init value",
        ));
    };

    let input = device_buffer_for_value(values, *input_id, "reduce.input")?;
    let Some(input_dram) = input.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable reduce input buffer has no device allocation",
        ));
    };
    let input_desc = plan
        .values
        .get(*input_id as usize)
        .ok_or_else(|| invalid_argument("TT executable reduce input id is out of bounds"))?;
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable reduce output id {output_id} is out of bounds"
        ))
    })?;
    if output_desc.element_type != input_desc.element_type {
        return Err(invalid_argument(
            "TT executable reduce input and output element types must match",
        ));
    }
    let dtype = pjrt_buffer_type_to_dtype(input_desc.element_type)?;
    let input_shape = dims_i64_to_usize(&input_desc.dims)?;
    let output_shape = dims_i64_to_usize(&output_desc.dims)?;
    let reduce_plan =
        kernels::reduce::ReducePlan::new(dtype, &input_shape, &output_shape, dimensions, reducer)
            .map_err(io_error)?;
    if !reduce_init_is_supported(plan, *init_value_id, reducer) {
        return Err(unimplemented(
            "TT executable reduce currently requires the StableHLO init value to be the reducer identity",
        ));
    }
    let output_dram = kernels::reduce::reduce(device, input_dram, &reduce_plan, "pjrt_reduce")
        .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_desc.dims.clone(),
        output_dram,
        context,
        "reduce",
    )
}

fn execute_top_k(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_id: u32,
    values_id: u32,
    indices_id: u32,
    k: u32,
) -> Result<(), *mut PJRT_Error> {
    let input = device_buffer_for_value(values, input_id, "top_k.input")?;
    let Some(input_dram) = input.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable top_k input buffer has no device allocation",
        ));
    };
    let input_desc = plan
        .values
        .get(input_id as usize)
        .ok_or_else(|| invalid_argument("TT executable top_k input id is out of bounds"))?;
    let values_desc = plan.values.get(values_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable top_k values id {values_id} is out of bounds"
        ))
    })?;
    let indices_desc = plan.values.get(indices_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable top_k indices id {indices_id} is out of bounds"
        ))
    })?;
    if values_desc.element_type != input_desc.element_type {
        return Err(invalid_argument(format!(
            "TT executable top_k values must match input type {:?}, got {:?}",
            input_desc.element_type, values_desc.element_type
        )));
    }
    if indices_desc.element_type != PJRT_Buffer_Type::PJRT_Buffer_Type_S32 {
        return Err(invalid_argument(format!(
            "TT executable top_k indices must be S32, got {:?}",
            indices_desc.element_type
        )));
    }
    let input_shape = dims_i64_to_usize(&input_desc.dims)?;
    let output_shape = vec![i64::from(k)];
    if values_desc.dims != output_shape || indices_desc.dims != output_shape {
        return Err(invalid_argument(format!(
            "TT executable top_k output shapes must both be {:?}, got values {:?}, indices {:?}",
            output_shape, values_desc.dims, indices_desc.dims
        )));
    }
    let (values_dram, indices_dram) =
        kernels::topk::top_k(device, input_dram, &input_shape, k as usize, "pjrt_top_k")
            .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        values_id,
        values_desc.dims.clone(),
        values_dram,
        context,
        "top_k.values",
    )?;
    store_output_buffer(
        values,
        plan,
        indices_id,
        indices_desc.dims.clone(),
        indices_dram,
        context,
        "top_k.indices",
    )
}

fn reduce_init_is_supported(
    plan: &executable::Executable,
    init_value_id: u32,
    reducer: executable::ReduceReducer,
) -> bool {
    if let Some(packed_value) = constant_packed_value(plan, init_value_id) {
        return match reducer {
            executable::ReduceReducer::Add => packed_value == 0,
            executable::ReduceReducer::Max => packed_value == f32::NEG_INFINITY.to_bits(),
            executable::ReduceReducer::Mul => false,
        };
    }
    true
}

fn constant_packed_value(plan: &executable::Executable, value_id: u32) -> Option<u32> {
    plan.ops.iter().find_map(|op| {
        if let executable::Op::Constant {
            packed_value,
            output_id,
        } = op
        {
            (*output_id == value_id).then_some(*packed_value)
        } else {
            None
        }
    })
}

fn execute_identity_custom_call(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    input_id: u32,
    output_id: u32,
    call_target_name: &str,
) -> Result<(), *mut PJRT_Error> {
    let input = device_buffer_for_value(
        values,
        input_id,
        &format!("custom_call {call_target_name:?}.input"),
    )?;
    let output_index = output_id as usize;
    let expected = plan.values.get(output_index).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable custom_call {call_target_name:?} output id {output_id} is out of bounds"
        ))
    })?;
    if input.buffer_type != expected.element_type {
        return Err(invalid_argument(format!(
            "TT executable custom_call {call_target_name:?} output must be {:?}, got {:?}",
            expected.element_type, input.buffer_type
        )));
    }
    if input.dims != expected.dims {
        return Err(invalid_argument(format!(
            "TT executable custom_call {call_target_name:?} output shape mismatch: expected {:?}, got {:?}",
            expected.dims, input.dims
        )));
    }
    let output = input.clone();
    values[output_index] = Some(output);
    Ok(())
}

fn execute_select(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_ids: [u32; 3],
    output_id: u32,
) -> Result<(), *mut PJRT_Error> {
    let [pred_id, true_id, false_id] = input_ids;
    let pred_desc = plan
        .values
        .get(pred_id as usize)
        .ok_or_else(|| invalid_argument("TT executable select.pred value id is out of bounds"))?;
    let true_desc = plan.values.get(true_id as usize).ok_or_else(|| {
        invalid_argument("TT executable select.on_true value id is out of bounds")
    })?;
    let false_desc = plan.values.get(false_id as usize).ok_or_else(|| {
        invalid_argument("TT executable select.on_false value id is out of bounds")
    })?;
    if pred_desc.element_type != PJRT_Buffer_Type::PJRT_Buffer_Type_PRED {
        return Err(invalid_argument(format!(
            "TT executable select predicate must be PRED, got {:?}",
            pred_desc.element_type
        )));
    }
    if true_desc.element_type != false_desc.element_type {
        return Err(invalid_argument(
            "TT executable select value element types must match",
        ));
    }
    if pred_desc.dims != true_desc.dims || true_desc.dims != false_desc.dims {
        return Err(unimplemented(
            "TT executable select currently only supports equal-shaped operands",
        ));
    }
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable select output id {output_id} is out of bounds"
        ))
    })?;
    if output_desc.element_type != true_desc.element_type {
        return Err(invalid_argument(format!(
            "TT executable select output must be {:?}, got {:?}",
            true_desc.element_type, output_desc.element_type
        )));
    }
    if output_desc.dims != true_desc.dims {
        return Err(invalid_argument(format!(
            "TT executable select output shape mismatch: expected {:?}, got {:?}",
            true_desc.dims, output_desc.dims
        )));
    }
    let value_dtype = pjrt_buffer_type_to_dtype(true_desc.element_type)?;
    if !matches!(value_dtype, DType::Float16B | DType::Float32 | DType::Int32) {
        return Err(unimplemented(format!(
            "TT executable select currently supports bf16, f32, and s32 values, got {:?}",
            true_desc.element_type
        )));
    }
    let pred = device_buffer_for_value(values, pred_id, "select.pred")?;
    let Some(pred_dram) = pred.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable select predicate buffer has no device allocation",
        ));
    };
    let true_input = select_value_input(values, plan, true_id, value_dtype, "select.on_true")?;
    let false_input = select_value_input(values, plan, false_id, value_dtype, "select.on_false")?;
    let expected_dims = true_desc.dims.clone();
    let shape = dims_i64_to_usize(&expected_dims)?;
    let output_dram = kernels::select::select(
        device,
        pred_dram,
        true_input,
        false_input,
        value_dtype,
        &shape,
        "pjrt_select",
    )
    .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        expected_dims,
        output_dram,
        context,
        "select",
    )
}

fn execute_broadcast_in_dim(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_id: u32,
    output_id: u32,
    broadcast_dimensions: &[i64],
) -> Result<(), *mut PJRT_Error> {
    let input_desc = plan.values.get(input_id as usize).ok_or_else(|| {
        invalid_argument("TT executable broadcast operand value id is out of bounds")
    })?;
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument("TT executable broadcast output value id is out of bounds")
    })?;
    if input_desc.element_type != output_desc.element_type {
        return Err(invalid_argument(
            "TT executable broadcast input and output element types must match",
        ));
    }
    let input_shape = dims_i64_to_usize(&input_desc.dims)?;
    let output_shape = dims_i64_to_usize(&output_desc.dims)?;
    let broadcast_plan = kernels::broadcast::BroadcastInDimPlan::new(
        &input_shape,
        &output_shape,
        broadcast_dimensions,
    )
    .map_err(io_error)?;

    let input = device_buffer_for_value(values, input_id, "broadcast_in_dim.operand")?;
    let Some(input_dram) = input.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable broadcast_in_dim operand buffer has no device allocation",
        ));
    };
    let dtype = pjrt_buffer_type_to_dtype(input_desc.element_type)?;
    let output_dims = output_desc.dims.clone();
    let output_dram = kernels::broadcast::broadcast_in_dim(
        device,
        input_dram,
        &broadcast_plan,
        dtype,
        "pjrt_broadcast",
    )
    .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_dims,
        output_dram,
        context,
        "broadcast_in_dim",
    )
}

fn execute_concatenate(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_ids: &[u32],
    output_id: u32,
    dimension: u64,
) -> Result<(), *mut PJRT_Error> {
    if input_ids.len() < 2 {
        return Err(invalid_argument(
            "TT executable concatenate requires at least two inputs",
        ));
    }

    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable concatenate output id {output_id} is out of bounds"
        ))
    })?;
    let output_shape = dims_i64_to_usize(&output_desc.dims)?;
    let dimension = usize::try_from(dimension)
        .map_err(|_| invalid_argument("TT executable concatenate dimension is too large"))?;
    if dimension >= output_shape.len() {
        return Err(invalid_argument(format!(
            "TT executable concatenate dimension {dimension} is out of bounds for shape {:?}",
            output_desc.dims
        )));
    }

    let dtype = pjrt_buffer_type_to_dtype(output_desc.element_type)?;
    let mut input_shapes = Vec::with_capacity(input_ids.len());
    let mut input_buffers = Vec::with_capacity(input_ids.len());
    for (index, &input_id) in input_ids.iter().enumerate() {
        let desc = plan.values.get(input_id as usize).ok_or_else(|| {
            invalid_argument(format!(
                "TT executable concatenate input {index} id {input_id} is out of bounds"
            ))
        })?;
        if desc.element_type != output_desc.element_type {
            return Err(invalid_argument(format!(
                "TT executable concatenate input {index} element type {:?} must match output {:?}",
                desc.element_type, output_desc.element_type
            )));
        }
        let input_shape = dims_i64_to_usize(&desc.dims)?;
        input_shapes.push(input_shape);
        let input = device_buffer_for_value(values, input_id, "concatenate.operand")?;
        let Some(input_dram) = input.dram_buffer.as_ref() else {
            return Err(failed_precondition(format!(
                "TT executable concatenate input {index} buffer has no device allocation"
            )));
        };
        input_buffers.push(input_dram);
    }

    let output_dram = kernels::concatenate::concatenate(
        device,
        &input_buffers,
        &input_shapes,
        &output_shape,
        dimension,
        dtype,
        "pjrt_concatenate",
    )
    .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_desc.dims.clone(),
        output_dram,
        context,
        "concatenate",
    )
}

fn execute_gather(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    input_ids: [u32; 2],
    output_id: u32,
    dimension_numbers: &executable::GatherDimensionNumbers,
    slice_sizes: &[i64],
) -> Result<(), *mut PJRT_Error> {
    if dimension_numbers.offset_dims.as_slice() != [1]
        || dimension_numbers.collapsed_slice_dims.as_slice() != [0]
        || !dimension_numbers.operand_batching_dims.is_empty()
        || !dimension_numbers.start_indices_batching_dims.is_empty()
        || dimension_numbers.start_index_map.as_slice() != [0]
        || dimension_numbers.index_vector_dim != 1
    {
        return Err(unimplemented(
            "TT executable gather currently only supports rank-2 row gathers",
        ));
    }

    let operand = device_buffer_for_value(values, input_ids[0], "gather.operand")?.clone();
    let start_indices =
        device_buffer_for_value(values, input_ids[1], "gather.start_indices")?.clone();
    if start_indices.buffer_type != PJRT_Buffer_Type::PJRT_Buffer_Type_S32 {
        return Err(unimplemented(
            "TT executable gather currently only supports s32 start_indices",
        ));
    }
    if operand.buffer_type != PJRT_Buffer_Type::PJRT_Buffer_Type_BF16 {
        return Err(unimplemented(
            "TT executable gather currently only supports bf16 operands",
        ));
    }

    let operand_shape = dims_i64_to_usize(&operand.dims)?;
    let start_indices_shape = dims_i64_to_usize(&start_indices.dims)?;
    if operand_shape.len() != 2 {
        return Err(unimplemented(
            "TT executable gather currently only supports rank-2 operands",
        ));
    }
    if slice_sizes.len() != 2
        || slice_sizes[0] != 1
        || usize::try_from(slice_sizes[1]).ok() != Some(operand_shape[1])
    {
        return Err(unimplemented(
            "TT executable gather currently only supports slice_sizes [1, operand_width]",
        ));
    }

    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable gather output id {output_id} is out of bounds"
        ))
    })?;
    if output_desc.element_type != operand.buffer_type {
        return Err(invalid_argument(format!(
            "TT executable gather output must be {:?}, got {:?}",
            operand.buffer_type, output_desc.element_type
        )));
    }
    let output_shape = dims_i64_to_usize(&output_desc.dims)?;
    let expected_output_shape = if start_indices_shape.len() == 2 {
        vec![start_indices_shape[0], operand_shape[1]]
    } else {
        Vec::new()
    };
    if output_shape != expected_output_shape {
        return Err(invalid_argument(format!(
            "TT executable gather output shape mismatch: expected {:?}, got {:?}",
            expected_output_shape, output_shape
        )));
    }

    let Some(operand_dram) = operand.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable gather operand buffer has no device allocation",
        ));
    };
    let Some(start_indices_dram) = start_indices.dram_buffer.as_ref() else {
        return Err(failed_precondition(
            "TT executable gather start_indices buffer has no device allocation",
        ));
    };

    let output_dram = kernels::gather::gather_bf16_rows(
        device,
        operand_dram,
        start_indices_dram,
        &operand_shape,
        &start_indices_shape,
        &output_shape,
        "pjrt_gather",
    )
    .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_desc.dims.clone(),
        output_dram,
        context,
        "gather",
    )
}

fn execute_iota(
    values: &mut [Option<PJRT_Buffer>],
    plan: &executable::Executable,
    device: &mut Device,
    context: &OutputContext,
    output_id: u32,
    iota_dimension: u64,
) -> Result<(), *mut PJRT_Error> {
    let output_desc = plan.values.get(output_id as usize).ok_or_else(|| {
        invalid_argument(format!(
            "TT executable iota output id {output_id} is out of bounds"
        ))
    })?;
    let logical_shape = dims_i64_to_usize(&output_desc.dims)?;
    let iota_dimension = usize::try_from(iota_dimension)
        .map_err(|_| invalid_argument("TT executable iota dimension is too large"))?;
    let dtype = pjrt_buffer_type_to_dtype(output_desc.element_type)?;
    let output_dram =
        kernels::iota::iota(device, dtype, &logical_shape, iota_dimension, "pjrt_iota")
            .map_err(io_error)?;
    store_output_buffer(
        values,
        plan,
        output_id,
        output_desc.dims.clone(),
        output_dram,
        context,
        "iota",
    )
}

fn execute_executable_v1(
    executable: &PJRT_LoadedExecutable,
    execute_device: *mut PJRT_Device,
    target_device: &mut PJRT_Device,
    inputs: &[*mut PJRT_Buffer],
) -> Result<Vec<PJRT_Buffer>, *mut PJRT_Error> {
    let plan = executable
        .metadata
        .executable
        .as_ref()
        .ok_or_else(|| failed_precondition("loaded executable has no TT executable payload"))?;
    let mut values = vec![None; plan.values.len()];
    let target_local_hardware_id = target_device.local_hardware_id as usize;
    let output_context = OutputContext {
        device: execute_device,
        memory: target_device.default_memory,
        local_hardware_id: target_local_hardware_id,
    };
    let device = &mut target_device.runtime;

    for op in &plan.ops {
        match op {
            executable::Op::Parameter {
                parameter_index,
                output_id,
            } => {
                let parameter_index = *parameter_index;
                let output_id = *output_id;
                let input_ptr = inputs.get(parameter_index).copied().ok_or_else(|| {
                    invalid_argument(format!(
                        "TT executable parameter index {parameter_index} is out of range"
                    ))
                })?;
                let input = unsafe { checked_ref(input_ptr, "argument_lists[0][*]") }?;
                if input.deleted {
                    return Err(failed_precondition("input buffers must not be deleted"));
                }
                if input.local_hardware_id != target_local_hardware_id {
                    return Err(invalid_argument(
                        "all input buffers and execute_device must be on the same device",
                    ));
                }

                let output_index = output_id as usize;
                let expected = plan.values.get(output_index).ok_or_else(|| {
                    invalid_argument(format!(
                        "TT executable parameter output id {output_id} is out of bounds"
                    ))
                })?;
                if input.buffer_type != expected.element_type {
                    return Err(unimplemented(format!(
                        "TT executable parameter {parameter_index} expected {:?}, got {:?}",
                        expected.element_type, input.buffer_type
                    )));
                }
                if input.dims != expected.dims {
                    return Err(invalid_argument(format!(
                        "TT executable parameter {parameter_index} shape mismatch: expected {:?}, got {:?}",
                        expected.dims, input.dims
                    )));
                }
                values[output_index] = Some(input.clone());
            }
            executable::Op::Add {
                input_ids,
                output_id,
            } => {
                execute_binary_eltwise(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    kernels::binary_eltwise::BinaryEltwiseOp::Add,
                    *input_ids,
                    *output_id,
                    "add",
                )?;
            }
            executable::Op::Subtract {
                input_ids,
                output_id,
            } => {
                execute_binary_eltwise(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    kernels::binary_eltwise::BinaryEltwiseOp::Subtract,
                    *input_ids,
                    *output_id,
                    "subtract",
                )?;
            }
            executable::Op::Multiply {
                input_ids,
                output_id,
            } => {
                execute_binary_eltwise(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    kernels::binary_eltwise::BinaryEltwiseOp::Multiply,
                    *input_ids,
                    *output_id,
                    "multiply",
                )?;
            }
            executable::Op::Divide {
                input_ids,
                output_id,
            } => {
                execute_binary_eltwise(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    kernels::binary_eltwise::BinaryEltwiseOp::Divide,
                    *input_ids,
                    *output_id,
                    "divide",
                )?;
            }
            executable::Op::Power {
                input_ids,
                output_id,
            } => {
                execute_binary_eltwise(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    kernels::binary_eltwise::BinaryEltwiseOp::Power,
                    *input_ids,
                    *output_id,
                    "power",
                )?;
            }
            executable::Op::Concatenate {
                input_ids,
                output_id,
                dimension,
            } => execute_concatenate(
                &mut values,
                plan,
                device,
                &output_context,
                input_ids,
                *output_id,
                *dimension,
            )?,
            executable::Op::Cosine {
                input_id,
                output_id,
            } => execute_unary_eltwise(
                &mut values,
                plan,
                device,
                &output_context,
                kernels::unary_eltwise::UnaryEltwiseOp::Cosine,
                *input_id,
                *output_id,
                "cosine",
            )?,
            executable::Op::Sine {
                input_id,
                output_id,
            } => execute_unary_eltwise(
                &mut values,
                plan,
                device,
                &output_context,
                kernels::unary_eltwise::UnaryEltwiseOp::Sine,
                *input_id,
                *output_id,
                "sine",
            )?,
            executable::Op::Rsqrt {
                input_id,
                output_id,
            } => execute_unary_eltwise(
                &mut values,
                plan,
                device,
                &output_context,
                kernels::unary_eltwise::UnaryEltwiseOp::Rsqrt,
                *input_id,
                *output_id,
                "rsqrt",
            )?,
            executable::Op::Reshape {
                input_id,
                output_id,
            } => execute_reshape(
                &mut values,
                plan,
                device,
                &output_context,
                *input_id,
                *output_id,
            )?,
            executable::Op::Slice {
                input_id,
                output_id,
                start_indices,
                limit_indices,
                strides,
            } => execute_slice(
                &mut values,
                plan,
                device,
                &output_context,
                *input_id,
                *output_id,
                start_indices,
                limit_indices,
                strides,
            )?,
            executable::Op::Negate {
                input_id,
                output_id,
            } => execute_unary_eltwise(
                &mut values,
                plan,
                device,
                &output_context,
                kernels::unary_eltwise::UnaryEltwiseOp::Negate,
                *input_id,
                *output_id,
                "negate",
            )?,
            executable::Op::Exponential {
                input_id,
                output_id,
            } => execute_unary_eltwise(
                &mut values,
                plan,
                device,
                &output_context,
                kernels::unary_eltwise::UnaryEltwiseOp::Exponential,
                *input_id,
                *output_id,
                "exponential",
            )?,
            executable::Op::Transpose {
                input_id,
                output_id,
                permutation,
            } => execute_transpose(
                &mut values,
                plan,
                device,
                &output_context,
                *input_id,
                *output_id,
                permutation,
            )?,
            executable::Op::CustomCall {
                input_ids,
                output_id,
                call_target_name,
                ..
            } if call_target_name == "annotate_device_placement" => {
                let [input_id] = input_ids.as_slice() else {
                    return Err(invalid_argument(format!(
                        "TT executable custom_call \"annotate_device_placement\" expected one input, got {}",
                        input_ids.len()
                    )));
                };
                execute_identity_custom_call(
                    &mut values,
                    plan,
                    *input_id,
                    *output_id,
                    call_target_name,
                )?;
            }
            executable::Op::CustomCall {
                call_target_name, ..
            } => {
                return Err(unimplemented(format!(
                    "TT executable custom_call {call_target_name:?} execution is not currently supported"
                )));
            }
            executable::Op::Convert {
                input_id,
                output_id,
            } => execute_unary_eltwise(
                &mut values,
                plan,
                device,
                &output_context,
                kernels::unary_eltwise::UnaryEltwiseOp::Convert,
                *input_id,
                *output_id,
                "convert",
            )?,
            executable::Op::Reduce {
                input_ids,
                init_value_ids,
                output_id,
                dimensions,
                reducer,
            } => execute_reduce(
                &mut values,
                plan,
                device,
                &output_context,
                input_ids,
                init_value_ids,
                *output_id,
                dimensions,
                *reducer,
            )?,
            executable::Op::Max {
                input_ids,
                output_id,
            } => {
                execute_binary_eltwise(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    kernels::binary_eltwise::BinaryEltwiseOp::Max,
                    *input_ids,
                    *output_id,
                    "max",
                )?;
            }
            executable::Op::Matmul {
                input_ids,
                output_id,
                dimension_numbers,
            } => {
                let output_id = *output_id;
                let lhs = device_buffer_for_value(&values, input_ids[0], "matmul.lhs")?;
                let rhs = device_buffer_for_value(&values, input_ids[1], "matmul.rhs")?;
                if lhs.buffer_type != PJRT_Buffer_Type::PJRT_Buffer_Type_BF16
                    || rhs.buffer_type != PJRT_Buffer_Type::PJRT_Buffer_Type_BF16
                {
                    return Err(unimplemented(
                        "TT executable matmul currently only supports bf16 buffers",
                    ));
                }
                if !dimension_numbers.lhs_batching_dimensions.is_empty()
                    || !dimension_numbers.rhs_batching_dimensions.is_empty()
                    || dimension_numbers.lhs_contracting_dimensions != vec![1]
                    || dimension_numbers.rhs_contracting_dimensions != vec![0]
                {
                    return Err(unimplemented(
                        "TT executable dot_general execution currently only supports rank-2 standard matrix multiplication",
                    ));
                }
                if lhs.dims.len() != 2 || rhs.dims.len() != 2 {
                    return Err(unimplemented(
                        "TT executable matmul currently only supports rank-2 buffers",
                    ));
                }
                if lhs.dims[1] != rhs.dims[0] {
                    return Err(invalid_argument(format!(
                        "TT executable matmul shape mismatch: lhs {:?}, rhs {:?}",
                        lhs.dims, rhs.dims
                    )));
                }

                let Some(lhs_dram) = lhs.dram_buffer.as_ref() else {
                    return Err(failed_precondition(
                        "TT executable matmul lhs buffer has no device allocation",
                    ));
                };
                let Some(rhs_dram) = rhs.dram_buffer.as_ref() else {
                    return Err(failed_precondition(
                        "TT executable matmul rhs buffer has no device allocation",
                    ));
                };

                let output_dram =
                    kernels::matmul::matmul_bf16(device, lhs_dram, rhs_dram, "pjrt_matmul")
                        .map_err(io_error)?;
                let expected_dims = vec![lhs.dims[0], rhs.dims[1]];
                store_output_buffer(
                    &mut values,
                    plan,
                    output_id,
                    expected_dims,
                    output_dram,
                    &output_context,
                    "matmul",
                )?;
            }
            executable::Op::Constant { .. } => {}
            executable::Op::Compare {
                input_ids,
                output_id,
                direction,
            } => {
                execute_binary_eltwise(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    kernels::binary_eltwise::BinaryEltwiseOp::Compare(*direction),
                    *input_ids,
                    *output_id,
                    "compare",
                )?;
            }
            executable::Op::Select {
                input_ids,
                output_id,
            } => {
                execute_select(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    *input_ids,
                    *output_id,
                )?;
            }
            executable::Op::BroadcastInDim {
                input_id,
                output_id,
                broadcast_dimensions,
            } => {
                execute_broadcast_in_dim(
                    &mut values,
                    plan,
                    device,
                    &output_context,
                    *input_id,
                    *output_id,
                    broadcast_dimensions,
                )?;
            }
            executable::Op::Gather {
                input_ids,
                output_id,
                dimension_numbers,
                slice_sizes,
                ..
            } => execute_gather(
                &mut values,
                plan,
                device,
                &output_context,
                *input_ids,
                *output_id,
                dimension_numbers,
                slice_sizes,
            )?,
            executable::Op::Iota {
                output_id,
                iota_dimension,
            } => execute_iota(
                &mut values,
                plan,
                device,
                &output_context,
                *output_id,
                *iota_dimension,
            )?,
            executable::Op::TopK {
                input_id,
                values_id,
                indices_id,
                k,
            } => execute_top_k(
                &mut values,
                plan,
                device,
                &output_context,
                *input_id,
                *values_id,
                *indices_id,
                *k,
            )?,
        }
    }
    device.finish_dispatch().map_err(io_error)?;

    let mut outputs = Vec::with_capacity(plan.output_ids.len());
    for (index, &output_id) in plan.output_ids.iter().enumerate() {
        let output = device_buffer_for_value(&values, output_id, &format!("output[{index}]"))?;
        outputs.push(output.clone());
    }
    Ok(outputs)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_LoadedExecutable_Execute(
    args: *mut PJRT_LoadedExecutable_Execute_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    log(format!(
        "pjrt loaded_executable_execute entered num_devices={} num_args={}",
        args.num_devices, args.num_args
    ));
    let Ok(executable) = (unsafe { checked_ref(args.executable, "executable") }) else {
        return invalid_argument("executable must not be null");
    };
    if executable.deleted {
        return failed_precondition("executable has been deleted");
    }
    if args.num_devices != 1 {
        return unimplemented("only single-device execution is supported");
    }
    if args.num_args > 0 && args.argument_lists.is_null() {
        return invalid_argument("argument_lists must not be null when num_args > 0");
    }
    if args.output_lists.is_null() {
        return invalid_argument("output_lists must not be null");
    }

    let execute_device = if !args.execute_device.is_null() {
        args.execute_device
    } else {
        executable
            .addressable_devices
            .first()
            .copied()
            .unwrap_or(ptr::null_mut())
    };
    if execute_device.is_null() {
        return invalid_argument("no execute device available");
    }
    let Ok(target_device) = (unsafe { checked_mut(execute_device, "execute_device") }) else {
        return invalid_argument("execute_device must not be null");
    };

    let input_ptrs = if args.num_args == 0 {
        &[][..]
    } else {
        let device_args = unsafe { *args.argument_lists };
        if device_args.is_null() {
            return invalid_argument("argument_lists[0] must not be null when num_args > 0");
        }
        unsafe { slice::from_raw_parts(device_args, args.num_args) }
    };
    let output_buffers =
        match execute_executable_v1(executable, execute_device, target_device, input_ptrs) {
            Ok(outputs) => outputs,
            Err(err) => return err,
        };
    if output_buffers.len() != executable.metadata.num_outputs {
        return pjrt_error(
            format!(
                "executable produced {} outputs but metadata expects {}",
                output_buffers.len(),
                executable.metadata.num_outputs
            ),
            PJRT_Error_Code::PJRT_Error_Code_INTERNAL,
        );
    }

    let device_outputs = unsafe { *args.output_lists };
    if device_outputs.is_null() {
        return invalid_argument("output_lists[0] must not be null");
    }
    for (index, output_buffer) in output_buffers.into_iter().enumerate() {
        let output_ptr = Box::into_raw(Box::new(output_buffer));
        unsafe {
            *device_outputs.add(index) = output_ptr;
        }
    }
    if !args.device_complete_events.is_null() {
        unsafe {
            *args.device_complete_events.add(0) = ready_event();
        }
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_BufferFromHostBuffer(
    args: *mut PJRT_Client_BufferFromHostBuffer_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    if !args.device_layout.is_null() {
        return unimplemented("custom device layouts are not supported");
    }
    match args.host_buffer_semantics {
        PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kImmutableOnlyDuringCall
        | PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes
        | PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kImmutableZeroCopy
        | PJRT_HostBufferSemantics::PJRT_HostBufferSemantics_kMutableZeroCopy => {}
    }

    let dtype = match pjrt_buffer_type_to_dtype(args.type_) {
        Ok(dtype) => dtype,
        Err(err) => return err,
    };
    let dims_i64 = match unsafe { checked_i64_slice(args.dims, args.num_dims, "dims") } {
        Ok(dims) => dims,
        Err(err) => return err,
    };
    let shape = match dims_i64_to_usize(dims_i64) {
        Ok(shape) => shape,
        Err(err) => return err,
    };
    if let Err(err) =
        validate_dense_row_major_strides(dtype, &shape, args.byte_strides, args.num_byte_strides)
    {
        return err;
    }
    let byte_size = match host_byte_size(dtype, &shape) {
        Ok(size) => size,
        Err(err) => return err,
    };
    if byte_size > 0 && args.data.is_null() {
        return invalid_argument("data must not be null");
    }

    let target_device = if !args.device.is_null() {
        args.device
    } else if !args.memory.is_null() {
        match unsafe { checked_ref(args.memory, "memory") } {
            Ok(memory) => memory
                .device_ptrs
                .first()
                .copied()
                .unwrap_or(ptr::null_mut()),
            Err(err) => return err,
        }
    } else {
        client
            .addressable_device_ptrs
            .first()
            .copied()
            .unwrap_or(ptr::null_mut())
    };
    if target_device.is_null() {
        return invalid_argument("no target device available");
    }
    let target_device_ref = match unsafe { checked_mut(target_device, "device") } {
        Ok(device) => device,
        Err(err) => return err,
    };
    let target_memory = if !args.memory.is_null() {
        args.memory
    } else {
        target_device_ref.default_memory
    };
    let local_hardware_id = target_device_ref.local_hardware_id as usize;
    log(format!(
        "pjrt buffer_from_host_buffer type={:?} dims={:?} local_hardware_id={}",
        args.type_, dims_i64, local_hardware_id
    ));

    let data = if byte_size == 0 {
        &[]
    } else {
        // SAFETY: caller owns `data` for `byte_size` bytes during the call.
        unsafe { slice::from_raw_parts(args.data.cast::<u8>(), byte_size) }
    };
    let allocation_shape = match dram::tiled_allocation_shape(&shape).map_err(io_error) {
        Ok(shape) => shape,
        Err(err) => return err,
    };
    let padded_data = match padded_host_data(data, dtype, &shape, &allocation_shape) {
        Ok(data) => data,
        Err(err) => return err,
    };
    let allocation_data = padded_data.as_deref().unwrap_or(data);
    let dram_buffer = match with_device(target_device_ref, |device| {
        device
            .alloc_write(allocation_data, dtype, &allocation_shape, "pjrt")
            .map_err(io_error)
    }) {
        Ok(buffer) => buffer,
        Err(err) => return err,
    };

    args.done_with_host_buffer = ready_event();
    args.buffer = Box::into_raw(Box::new(PJRT_Buffer {
        buffer_type: dtype_to_pjrt_buffer_type(dram_buffer.dtype),
        dims: dims_i64.to_vec(),
        device: target_device,
        memory: target_memory,
        local_hardware_id,
        dram_buffer: Some(dram_buffer),
        deleted: false,
    }));
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_DeviceDescription_Id(
    args: *mut PJRT_DeviceDescription_Id_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(description) = (unsafe { checked_ref(args.device_description, "device_description") })
    else {
        return invalid_argument("device_description must not be null");
    };
    args.id = description.id;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_DeviceDescription_ProcessIndex(
    args: *mut PJRT_DeviceDescription_ProcessIndex_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(description) = (unsafe { checked_ref(args.device_description, "device_description") })
    else {
        return invalid_argument("device_description must not be null");
    };
    args.process_index = description.process_index;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_DeviceDescription_Attributes(
    args: *mut PJRT_DeviceDescription_Attributes_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.device_description.is_null() {
        return invalid_argument("device_description must not be null");
    }
    args.attributes = ptr::null();
    args.num_attributes = 0;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_DeviceDescription_Kind(
    args: *mut PJRT_DeviceDescription_Kind_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(description) = (unsafe { checked_ref(args.device_description, "device_description") })
    else {
        return invalid_argument("device_description must not be null");
    };
    args.device_kind = description.device_kind.as_ptr();
    args.device_kind_size = description.device_kind.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_DeviceDescription_DebugString(
    args: *mut PJRT_DeviceDescription_DebugString_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(description) = (unsafe { checked_ref(args.device_description, "device_description") })
    else {
        return invalid_argument("device_description must not be null");
    };
    args.debug_string = description.debug_string.as_ptr();
    args.debug_string_size = description.debug_string.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_DeviceDescription_ToString(
    args: *mut PJRT_DeviceDescription_ToString_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(description) = (unsafe { checked_ref(args.device_description, "device_description") })
    else {
        return invalid_argument("device_description must not be null");
    };
    args.to_string = description.to_string.as_ptr();
    args.to_string_size = description.to_string.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Device_GetDescription(
    args: *mut PJRT_Device_GetDescription_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(device) = (unsafe { checked_ref(args.device, "device") }) else {
        return invalid_argument("device must not be null");
    };
    args.device_description = device.description;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Device_IsAddressable(
    args: *mut PJRT_Device_IsAddressable_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(device) = (unsafe { checked_ref(args.device, "device") }) else {
        return invalid_argument("device must not be null");
    };
    args.is_addressable = device.addressable;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Device_LocalHardwareId(
    args: *mut PJRT_Device_LocalHardwareId_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(device) = (unsafe { checked_ref(args.device, "device") }) else {
        return invalid_argument("device must not be null");
    };
    args.local_hardware_id = device.local_hardware_id;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Device_AddressableMemories(
    args: *mut PJRT_Device_AddressableMemories_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(device) = (unsafe { checked_ref(args.device, "device") }) else {
        return invalid_argument("device must not be null");
    };
    args.memories = if device.memory_ptrs.is_empty() {
        ptr::null()
    } else {
        device.memory_ptrs.as_ptr()
    };
    args.num_memories = device.memory_ptrs.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Device_DefaultMemory(
    args: *mut PJRT_Device_DefaultMemory_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(device) = (unsafe { checked_ref(args.device, "device") }) else {
        return invalid_argument("device must not be null");
    };
    args.memory = device.default_memory;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Device_MemoryStats(
    args: *mut PJRT_Device_MemoryStats_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.device.is_null() {
        return invalid_argument("device must not be null");
    }
    args.bytes_in_use = 0;
    args.peak_bytes_in_use = 0;
    args.peak_bytes_in_use_is_set = false;
    args.num_allocs = 0;
    args.num_allocs_is_set = false;
    args.largest_alloc_size = 0;
    args.largest_alloc_size_is_set = false;
    args.bytes_limit = 0;
    args.bytes_limit_is_set = false;
    args.bytes_reserved = 0;
    args.bytes_reserved_is_set = false;
    args.peak_bytes_reserved = 0;
    args.peak_bytes_reserved_is_set = false;
    args.bytes_reservable_limit = 0;
    args.bytes_reservable_limit_is_set = false;
    args.largest_free_block_bytes = 0;
    args.largest_free_block_bytes_is_set = false;
    args.pool_bytes = 0;
    args.pool_bytes_is_set = false;
    args.peak_pool_bytes = 0;
    args.peak_pool_bytes_is_set = false;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Device_GetAttributes(
    args: *mut PJRT_Device_GetAttributes_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.device.is_null() {
        return invalid_argument("device must not be null");
    }
    args.attributes = ptr::null();
    args.num_attributes = 0;
    args.device_attributes = Box::into_raw(Box::new(PJRT_Device_Attributes { _unused: [] }));
    args.attributes_deleter = Some(noop_device_attributes_deleter);
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Memory_Id(args: *mut PJRT_Memory_Id_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(memory) = (unsafe { checked_ref(args.memory, "memory") }) else {
        return invalid_argument("memory must not be null");
    };
    args.id = memory.id;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Memory_Kind(args: *mut PJRT_Memory_Kind_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(memory) = (unsafe { checked_ref(args.memory, "memory") }) else {
        return invalid_argument("memory must not be null");
    };
    args.kind = memory.kind.as_ptr();
    args.kind_size = memory.kind.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Memory_Kind_Id(args: *mut PJRT_Memory_Kind_Id_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(memory) = (unsafe { checked_ref(args.memory, "memory") }) else {
        return invalid_argument("memory must not be null");
    };
    args.kind_id = memory.id;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Memory_DebugString(
    args: *mut PJRT_Memory_DebugString_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(memory) = (unsafe { checked_ref(args.memory, "memory") }) else {
        return invalid_argument("memory must not be null");
    };
    args.debug_string = memory.debug_string.as_ptr();
    args.debug_string_size = memory.debug_string.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Memory_ToString(
    args: *mut PJRT_Memory_ToString_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(memory) = (unsafe { checked_ref(args.memory, "memory") }) else {
        return invalid_argument("memory must not be null");
    };
    args.to_string = memory.to_string.as_ptr();
    args.to_string_size = memory.to_string.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Memory_AddressableByDevices(
    args: *mut PJRT_Memory_AddressableByDevices_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(memory) = (unsafe { checked_ref(args.memory, "memory") }) else {
        return invalid_argument("memory must not be null");
    };
    args.devices = if memory.device_ptrs.is_empty() {
        ptr::null()
    } else {
        memory.device_ptrs.as_ptr()
    };
    args.num_devices = memory.device_ptrs.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_Destroy(args: *mut PJRT_Buffer_Destroy_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if !args.buffer.is_null() {
        // SAFETY: `buffer` is allocated by `TT_Client_BufferFromHostBuffer`.
        unsafe {
            drop(Box::from_raw(args.buffer));
        }
        args.buffer = ptr::null_mut();
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_ElementType(
    args: *mut PJRT_Buffer_ElementType_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.type_ = buffer.buffer_type;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_Dimensions(
    args: *mut PJRT_Buffer_Dimensions_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.dims = if buffer.dims.is_empty() {
        ptr::null()
    } else {
        buffer.dims.as_ptr()
    };
    args.num_dims = buffer.dims.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_UnpaddedDimensions(
    args: *mut PJRT_Buffer_UnpaddedDimensions_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.unpadded_dims = if buffer.dims.is_empty() {
        ptr::null()
    } else {
        buffer.dims.as_ptr()
    };
    args.num_dims = buffer.dims.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_DynamicDimensionIndices(
    args: *mut PJRT_Buffer_DynamicDimensionIndices_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(_buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.dynamic_dim_indices = ptr::null();
    args.num_dynamic_dims = 0;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_OnDeviceSizeInBytes(
    args: *mut PJRT_Buffer_OnDeviceSizeInBytes_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    let Some(dram_buffer) = buffer.dram_buffer.as_ref() else {
        return pjrt_error(
            "buffer has been deleted",
            PJRT_Error_Code::PJRT_Error_Code_FAILED_PRECONDITION,
        );
    };
    args.on_device_size_in_bytes = dram_buffer.size();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_Device(args: *mut PJRT_Buffer_Device_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.device = buffer.device;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_Memory(args: *mut PJRT_Buffer_Memory_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.memory = buffer.memory;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_Delete(args: *mut PJRT_Buffer_Delete_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_mut(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    buffer.deleted = true;
    buffer.dram_buffer = None;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_IsDeleted(
    args: *mut PJRT_Buffer_IsDeleted_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.is_deleted = buffer.deleted;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_ToHostBuffer(
    args: *mut PJRT_Buffer_ToHostBuffer_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.src, "src") }) else {
        return invalid_argument("src must not be null");
    };
    let dtype = match pjrt_buffer_type_to_dtype(buffer.buffer_type) {
        Ok(dtype) => dtype,
        Err(err) => return err,
    };
    let dims = match dims_i64_to_usize(&buffer.dims) {
        Ok(dims) => dims,
        Err(err) => return err,
    };
    let byte_size = match host_byte_size(dtype, &dims) {
        Ok(size) => size,
        Err(err) => return err,
    };
    if args.dst_size < byte_size {
        return invalid_argument("dst buffer is too small");
    }
    if byte_size > 0 && args.dst.is_null() {
        return invalid_argument("dst must not be null for non-empty buffers");
    }

    let data = match read_buffer_logical_bytes(buffer) {
        Ok(data) => data,
        Err(err) => return err,
    };
    if data.len() != byte_size {
        return pjrt_error(
            format!(
                "readback byte size {} does not match buffer byte size {}",
                data.len(),
                byte_size
            ),
            PJRT_Error_Code::PJRT_Error_Code_INTERNAL,
        );
    }
    if byte_size > 0 {
        // SAFETY: caller owns `dst` for at least `dst_size` bytes and we checked capacity.
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), args.dst.cast::<u8>(), data.len());
        }
    }
    args.event = ready_event();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_IsOnCpu(args: *mut PJRT_Buffer_IsOnCpu_Args) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(_buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.is_on_cpu = false;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_ReadyEvent(
    args: *mut PJRT_Buffer_ReadyEvent_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    args.event = event_for_buffer(buffer);
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_ExecuteContext_Create(
    args: *mut PJRT_ExecuteContext_Create_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    args.context = Box::into_raw(Box::new(PJRT_ExecuteContext { _unused: [] }));
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_ExecuteContext_Destroy(
    args: *mut PJRT_ExecuteContext_Destroy_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if !args.context.is_null() {
        unsafe {
            drop(Box::from_raw(args.context));
        }
        args.context = ptr::null_mut();
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_CopyRawToHost(
    args: *mut PJRT_Buffer_CopyRawToHost_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(buffer) = (unsafe { checked_ref(args.buffer, "buffer") }) else {
        return invalid_argument("buffer must not be null");
    };
    if args.offset < 0 || args.transfer_size < 0 {
        return invalid_argument("offset and transfer_size must be non-negative");
    }
    let offset = args.offset as usize;
    let transfer_size = args.transfer_size as usize;
    if transfer_size > 0 && args.dst.is_null() {
        return invalid_argument("dst must not be null for non-empty transfers");
    }

    let data = match read_buffer_logical_bytes(buffer) {
        Ok(data) => data,
        Err(err) => return err,
    };
    let end = match offset.checked_add(transfer_size) {
        Some(end) if end <= data.len() => end,
        _ => return invalid_argument("offset + transfer_size exceeds buffer size"),
    };
    if transfer_size > 0 {
        unsafe {
            ptr::copy_nonoverlapping(
                data[offset..end].as_ptr(),
                args.dst.cast::<u8>(),
                transfer_size,
            );
        }
    }
    args.event = ready_event();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_IncreaseExternalReferenceCount(
    args: *mut PJRT_Buffer_IncreaseExternalReferenceCount_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.buffer.is_null() {
        return invalid_argument("buffer must not be null");
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_DecreaseExternalReferenceCount(
    args: *mut PJRT_Buffer_DecreaseExternalReferenceCount_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.buffer.is_null() {
        return invalid_argument("buffer must not be null");
    }
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_TopologyDescription_PlatformName(
    args: *mut PJRT_TopologyDescription_PlatformName_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(topology) = (unsafe { checked_ref(args.topology, "topology") }) else {
        return invalid_argument("topology must not be null");
    };
    args.platform_name = topology.platform_name.as_ptr();
    args.platform_name_size = topology.platform_name.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_TopologyDescription_PlatformVersion(
    args: *mut PJRT_TopologyDescription_PlatformVersion_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(topology) = (unsafe { checked_ref(args.topology, "topology") }) else {
        return invalid_argument("topology must not be null");
    };
    args.platform_version = topology.platform_version.as_ptr();
    args.platform_version_size = topology.platform_version.as_bytes().len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_TopologyDescription_GetDeviceDescriptions(
    args: *mut PJRT_TopologyDescription_GetDeviceDescriptions_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(topology) = (unsafe { checked_ref(args.topology, "topology") }) else {
        return invalid_argument("topology must not be null");
    };
    args.descriptions = if topology.device_description_ptrs.is_empty() {
        ptr::null()
    } else {
        topology.device_description_ptrs.as_ptr()
    };
    args.num_descriptions = topology.device_description_ptrs.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_TopologyDescription_Attributes(
    args: *mut PJRT_TopologyDescription_Attributes_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.topology.is_null() {
        return invalid_argument("topology must not be null");
    }
    args.attributes = ptr::null();
    args.num_attributes = 0;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_TopologyDescription_Fingerprint(
    args: *mut PJRT_TopologyDescription_Fingerprint_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(topology) = (unsafe { checked_ref(args.topology, "topology") }) else {
        return invalid_argument("topology must not be null");
    };

    let mut fingerprint = 0xcbf29ce484222325u64;
    for byte in topology.platform_name.as_bytes() {
        fingerprint ^= u64::from(*byte);
        fingerprint = fingerprint.wrapping_mul(0x100000001b3);
    }
    for byte in topology.platform_version.as_bytes() {
        fingerprint ^= u64::from(*byte);
        fingerprint = fingerprint.wrapping_mul(0x100000001b3);
    }
    fingerprint ^= topology.device_description_ptrs.len() as u64;
    fingerprint = fingerprint.wrapping_mul(0x100000001b3);

    args.fingerprint = fingerprint;
    ptr::null_mut()
}

fn build_pjrt_api() -> PJRT_Api {
    let mut api: PJRT_Api = unsafe { std::mem::zeroed() };

    api.struct_size = size_of::<PJRT_Api>();
    api.extension_start = ptr::null_mut();
    api.pjrt_api_version = PJRT_Api_Version {
        struct_size: size_of::<PJRT_Api_Version>(),
        extension_start: ptr::null_mut(),
        major_version: PJRT_API_MAJOR as _,
        minor_version: PJRT_API_MINOR as _,
    };

    api.PJRT_Error_Destroy = Some(TT_Error_Destroy);
    api.PJRT_Error_Message = Some(TT_Error_Message);
    api.PJRT_Error_GetCode = Some(TT_Error_GetCode);
    api.PJRT_Error_ForEachPayload = Some(TT_Error_ForEachPayload);
    api.PJRT_Plugin_Initialize = Some(TT_Plugin_Initialize);
    api.PJRT_Plugin_Attributes = Some(TT_Plugin_Attributes);
    api.PJRT_Event_Destroy = Some(TT_Event_Destroy);
    api.PJRT_Event_IsReady = Some(TT_Event_IsReady);
    api.PJRT_Event_Error = Some(TT_Event_Error);
    api.PJRT_Event_Await = Some(TT_Event_Await);
    api.PJRT_Event_OnReady = Some(TT_Event_OnReady);
    api.PJRT_Client_Create = Some(TT_Client_Create);
    api.PJRT_Client_Destroy = Some(TT_Client_Destroy);
    api.PJRT_Client_PlatformName = Some(TT_Client_PlatformName);
    api.PJRT_Client_ProcessIndex = Some(TT_Client_ProcessIndex);
    api.PJRT_Client_PlatformVersion = Some(TT_Client_PlatformVersion);
    api.PJRT_Client_Devices = Some(TT_Client_Devices);
    api.PJRT_Client_AddressableDevices = Some(TT_Client_AddressableDevices);
    api.PJRT_Client_LookupDevice = Some(TT_Client_LookupDevice);
    api.PJRT_Client_LookupAddressableDevice = Some(TT_Client_LookupAddressableDevice);
    api.PJRT_Client_AddressableMemories = Some(TT_Client_AddressableMemories);
    api.PJRT_Client_Compile = Some(TT_Client_Compile);
    api.PJRT_Client_DefaultDeviceAssignment = Some(TT_Client_DefaultDeviceAssignment);
    api.PJRT_Client_BufferFromHostBuffer = Some(TT_Client_BufferFromHostBuffer);
    api.PJRT_DeviceDescription_Id = Some(TT_DeviceDescription_Id);
    api.PJRT_DeviceDescription_ProcessIndex = Some(TT_DeviceDescription_ProcessIndex);
    api.PJRT_DeviceDescription_Attributes = Some(TT_DeviceDescription_Attributes);
    api.PJRT_DeviceDescription_Kind = Some(TT_DeviceDescription_Kind);
    api.PJRT_DeviceDescription_DebugString = Some(TT_DeviceDescription_DebugString);
    api.PJRT_DeviceDescription_ToString = Some(TT_DeviceDescription_ToString);
    api.PJRT_Device_GetDescription = Some(TT_Device_GetDescription);
    api.PJRT_Device_IsAddressable = Some(TT_Device_IsAddressable);
    api.PJRT_Device_LocalHardwareId = Some(TT_Device_LocalHardwareId);
    api.PJRT_Device_AddressableMemories = Some(TT_Device_AddressableMemories);
    api.PJRT_Device_DefaultMemory = Some(TT_Device_DefaultMemory);
    api.PJRT_Device_MemoryStats = Some(TT_Device_MemoryStats);
    api.PJRT_Memory_Id = Some(TT_Memory_Id);
    api.PJRT_Memory_Kind = Some(TT_Memory_Kind);
    api.PJRT_Memory_DebugString = Some(TT_Memory_DebugString);
    api.PJRT_Memory_ToString = Some(TT_Memory_ToString);
    api.PJRT_Memory_AddressableByDevices = Some(TT_Memory_AddressableByDevices);
    api.PJRT_Executable_Destroy = Some(TT_Executable_Destroy);
    api.PJRT_Executable_Name = Some(TT_Executable_Name);
    api.PJRT_Executable_NumReplicas = Some(TT_Executable_NumReplicas);
    api.PJRT_Executable_NumPartitions = Some(TT_Executable_NumPartitions);
    api.PJRT_Executable_NumOutputs = Some(TT_Executable_NumOutputs);
    api.PJRT_Executable_OutputMemoryKinds = Some(TT_Executable_OutputMemoryKinds);
    api.PJRT_Executable_OptimizedProgram = Some(TT_Executable_OptimizedProgram);
    api.PJRT_LoadedExecutable_Destroy = Some(TT_LoadedExecutable_Destroy);
    api.PJRT_LoadedExecutable_GetExecutable = Some(TT_LoadedExecutable_GetExecutable);
    api.PJRT_LoadedExecutable_AddressableDevices = Some(TT_LoadedExecutable_AddressableDevices);
    api.PJRT_LoadedExecutable_Delete = Some(TT_LoadedExecutable_Delete);
    api.PJRT_LoadedExecutable_IsDeleted = Some(TT_LoadedExecutable_IsDeleted);
    api.PJRT_LoadedExecutable_Execute = Some(TT_LoadedExecutable_Execute);
    api.PJRT_LoadedExecutable_Fingerprint = Some(TT_LoadedExecutable_Fingerprint);
    api.PJRT_Buffer_Destroy = Some(TT_Buffer_Destroy);
    api.PJRT_Buffer_ElementType = Some(TT_Buffer_ElementType);
    api.PJRT_Buffer_Dimensions = Some(TT_Buffer_Dimensions);
    api.PJRT_Buffer_UnpaddedDimensions = Some(TT_Buffer_UnpaddedDimensions);
    api.PJRT_Buffer_DynamicDimensionIndices = Some(TT_Buffer_DynamicDimensionIndices);
    api.PJRT_Buffer_OnDeviceSizeInBytes = Some(TT_Buffer_OnDeviceSizeInBytes);
    api.PJRT_Buffer_Device = Some(TT_Buffer_Device);
    api.PJRT_Buffer_Memory = Some(TT_Buffer_Memory);
    api.PJRT_Buffer_Delete = Some(TT_Buffer_Delete);
    api.PJRT_Buffer_IsDeleted = Some(TT_Buffer_IsDeleted);
    api.PJRT_Buffer_ToHostBuffer = Some(TT_Buffer_ToHostBuffer);
    api.PJRT_Buffer_IsOnCpu = Some(TT_Buffer_IsOnCpu);
    api.PJRT_Buffer_ReadyEvent = Some(TT_Buffer_ReadyEvent);
    api.PJRT_Buffer_IncreaseExternalReferenceCount = Some(TT_Buffer_IncreaseExternalReferenceCount);
    api.PJRT_Buffer_DecreaseExternalReferenceCount = Some(TT_Buffer_DecreaseExternalReferenceCount);
    api.PJRT_TopologyDescription_PlatformName = Some(TT_TopologyDescription_PlatformName);
    api.PJRT_TopologyDescription_PlatformVersion = Some(TT_TopologyDescription_PlatformVersion);
    api.PJRT_TopologyDescription_GetDeviceDescriptions =
        Some(TT_TopologyDescription_GetDeviceDescriptions);
    api.PJRT_TopologyDescription_Attributes = Some(TT_TopologyDescription_Attributes);
    api.PJRT_Compile = Some(TT_Compile);
    api.PJRT_Executable_OutputElementTypes = Some(TT_Executable_OutputElementTypes);
    api.PJRT_Executable_OutputDimensions = Some(TT_Executable_OutputDimensions);
    api.PJRT_Executable_Fingerprint = Some(TT_Executable_Fingerprint);
    api.PJRT_Client_TopologyDescription = Some(TT_Client_TopologyDescription);
    api.PJRT_Executable_GetCompiledMemoryStats = Some(TT_Executable_GetCompiledMemoryStats);
    api.PJRT_Memory_Kind_Id = Some(TT_Memory_Kind_Id);
    api.PJRT_ExecuteContext_Create = Some(TT_ExecuteContext_Create);
    api.PJRT_ExecuteContext_Destroy = Some(TT_ExecuteContext_Destroy);
    api.PJRT_Buffer_CopyRawToHost = Some(TT_Buffer_CopyRawToHost);
    api.PJRT_LoadedExecutable_GetDeviceAssignment = Some(TT_LoadedExecutable_GetDeviceAssignment);
    api.PJRT_Device_GetAttributes = Some(TT_Device_GetAttributes);
    api.PJRT_TopologyDescription_Fingerprint = Some(TT_TopologyDescription_Fingerprint);

    api
}

static INIT_PJRT_API: Once = Once::new();
static mut PJRT_API: std::mem::MaybeUninit<PJRT_Api> = std::mem::MaybeUninit::uninit();

#[unsafe(no_mangle)]
pub extern "C" fn GetPjrtApi() -> *const PJRT_Api {
    INIT_PJRT_API.call_once(|| unsafe {
        std::ptr::write(
            std::ptr::addr_of_mut!(PJRT_API),
            std::mem::MaybeUninit::new(build_pjrt_api()),
        );
    });
    std::ptr::addr_of!(PJRT_API).cast::<PJRT_Api>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, ProbeInfo};
    use std::path::PathBuf;

    fn take_error_detail(api: &PJRT_Api, error: *mut PJRT_Error) -> (PJRT_Error_Code, String) {
        assert!(!error.is_null(), "expected PJRT error");
        let mut code_args = PJRT_Error_GetCode_Args {
            struct_size: size_of::<PJRT_Error_GetCode_Args>(),
            extension_start: ptr::null_mut(),
            error,
            code: PJRT_Error_Code::PJRT_Error_Code_UNKNOWN,
        };
        let get_code = api.PJRT_Error_GetCode.expect("error get code must exist");
        let status = unsafe { get_code(&mut code_args) };
        assert!(status.is_null(), "error inspection should not fail");

        let mut message_args = PJRT_Error_Message_Args {
            struct_size: size_of::<PJRT_Error_Message_Args>(),
            extension_start: ptr::null_mut(),
            error,
            message: ptr::null(),
            message_size: 0,
        };
        let message = api.PJRT_Error_Message.expect("error message must exist");
        unsafe { message(&mut message_args) };
        let detail = if message_args.message.is_null() {
            String::from("<no message>")
        } else {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    message_args.message.cast::<u8>(),
                    message_args.message_size,
                )
            };
            String::from_utf8_lossy(bytes).into_owned()
        };

        let destroy = api.PJRT_Error_Destroy.expect("error destroy must exist");
        unsafe {
            destroy(&mut PJRT_Error_Destroy_Args {
                struct_size: size_of::<PJRT_Error_Destroy_Args>(),
                extension_start: ptr::null_mut(),
                error,
            });
        }
        (code_args.code, detail)
    }

    fn check_ok(api: &PJRT_Api, error: *mut PJRT_Error) {
        if error.is_null() {
            return;
        }

        let (code, detail) = take_error_detail(api, error);
        panic!("unexpected PJRT error {code:?}: {detail}");
    }

    #[cfg(libtt_mlir_frontend)]
    fn with_compiled_mlir_executable(code: &str, check: impl FnOnce(&executable::Executable)) {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = code.as_bytes().to_vec();
        let program = PJRT_Program {
            struct_size: size_of::<PJRT_Program>(),
            extension_start: ptr::null_mut(),
            code: code.as_mut_ptr().cast::<c_char>(),
            code_size: code.len(),
            format: format.as_mut_ptr().cast::<c_char>(),
            format_size: format.len(),
        };

        let compile = api
            .PJRT_Client_Compile
            .expect("PJRT_Client_Compile must be exported");
        let mut compile_args = PJRT_Client_Compile_Args {
            struct_size: size_of::<PJRT_Client_Compile_Args>(),
            extension_start: ptr::null_mut(),
            client,
            program: &program,
            compile_options: ptr::null(),
            compile_options_size: 0,
            executable: ptr::null_mut(),
        };
        check_ok(api, unsafe { compile(&mut compile_args) });
        assert!(!compile_args.executable.is_null());

        let get_executable = api
            .PJRT_LoadedExecutable_GetExecutable
            .expect("PJRT_LoadedExecutable_GetExecutable must be exported");
        let mut get_executable_args = PJRT_LoadedExecutable_GetExecutable_Args {
            struct_size: size_of::<PJRT_LoadedExecutable_GetExecutable_Args>(),
            extension_start: ptr::null_mut(),
            loaded_executable: compile_args.executable,
            executable: ptr::null_mut(),
        };
        check_ok(api, unsafe { get_executable(&mut get_executable_args) });

        let executable = unsafe { &*get_executable_args.executable }
            .metadata
            .executable
            .as_ref()
            .expect("compiled executable should contain a TT executable");
        check(executable);

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[test]
    fn tiled_allocation_shape_pads_scalar_and_vector_buffers() {
        assert_eq!(
            dram::tiled_allocation_shape(&[]).expect("scalar shape should pad"),
            vec![32, 32]
        );
        assert_eq!(
            dram::tiled_allocation_shape(&[5]).expect("vector shape should pad"),
            vec![32, 32]
        );
        assert_eq!(
            dram::tiled_allocation_shape(&[128]).expect("aligned vector shape should pad rank"),
            vec![32, 128]
        );
        assert_eq!(
            dram::tiled_allocation_shape(&[32, 64])
                .expect("aligned rank-2 shape should be preserved"),
            vec![32, 64]
        );
        assert_eq!(
            dram::tiled_allocation_shape(&[32, 1]).expect("rank-2 shape should pad columns"),
            vec![32, 32]
        );
        assert_eq!(
            dram::tiled_allocation_shape(&[33, 33])
                .expect("rank-2 shape should pad rows and columns"),
            vec![64, 64]
        );
    }

    #[test]
    fn padded_host_data_places_logical_payload_at_start() {
        let data = [1u8, 0, 2, 0, 3, 0];
        let padded = padded_host_data(&data, DType::UInt16, &[3], &[32, 32])
            .expect("vector payload should pad")
            .expect("padding should be required");
        assert_eq!(&padded[..data.len()], data);
        assert!(padded[data.len()..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn padded_host_data_preserves_logical_matrix_shape() {
        let data = [1u16, 2, 3, 4, 5, 6]
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let padded = padded_host_data(&data, DType::UInt16, &[3, 2], &[32, 32])
            .expect("matrix payload should pad")
            .expect("padding should be required");

        let value_at = |row: usize, col: usize| {
            let offset = (row * 32 + col) * 2;
            u16::from_le_bytes([padded[offset], padded[offset + 1]])
        };
        assert_eq!(value_at(0, 0), 1);
        assert_eq!(value_at(0, 1), 2);
        assert_eq!(value_at(1, 0), 3);
        assert_eq!(value_at(1, 1), 4);
        assert_eq!(value_at(2, 0), 5);
        assert_eq!(value_at(2, 1), 6);
        assert_eq!(value_at(0, 2), 0);
    }

    #[test]
    fn crop_padded_host_data_preserves_logical_matrix_shape() {
        let values = (0u16..(32 * 32)).collect::<Vec<_>>();
        let data = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let cropped = crop_padded_host_data(&data, DType::UInt16, &[3, 1], &[32, 32])
            .expect("crop should not fail")
            .expect("crop should be possible");
        let expected = [0u16, 32, 64]
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(cropped, expected);
    }

    #[test]
    fn get_pjrt_api_exposes_minimal_client_and_device_interface() {
        let api = unsafe { &*GetPjrtApi() };
        assert_eq!(api.pjrt_api_version.major_version, PJRT_API_MAJOR as i32);
        assert_eq!(api.pjrt_api_version.minor_version, PJRT_API_MINOR as i32);

        let plugin_init = api
            .PJRT_Plugin_Initialize
            .expect("PJRT_Plugin_Initialize must be exported");
        let mut init_args = PJRT_Plugin_Initialize_Args {
            struct_size: size_of::<PJRT_Plugin_Initialize_Args>(),
            extension_start: ptr::null_mut(),
        };
        check_ok(api, unsafe { plugin_init(&mut init_args) });

        let client_create = api
            .PJRT_Client_Create
            .expect("PJRT_Client_Create must be exported");
        let mut create_args = PJRT_Client_Create_Args {
            struct_size: size_of::<PJRT_Client_Create_Args>(),
            extension_start: ptr::null_mut(),
            create_options: ptr::null(),
            num_options: 0,
            kv_get_callback: None,
            kv_get_user_arg: ptr::null_mut(),
            kv_put_callback: None,
            kv_put_user_arg: ptr::null_mut(),
            client: ptr::null_mut(),
            kv_try_get_callback: None,
            kv_try_get_user_arg: ptr::null_mut(),
        };
        check_ok(api, unsafe { client_create(&mut create_args) });
        assert!(!create_args.client.is_null());

        let client_devices = api
            .PJRT_Client_Devices
            .expect("PJRT_Client_Devices must be exported");
        let mut devices_args = PJRT_Client_Devices_Args {
            struct_size: size_of::<PJRT_Client_Devices_Args>(),
            extension_start: ptr::null_mut(),
            client: create_args.client,
            devices: ptr::null(),
            num_devices: 0,
        };
        check_ok(api, unsafe { client_devices(&mut devices_args) });

        if devices_args.num_devices > 0 {
            let devices = unsafe {
                std::slice::from_raw_parts(devices_args.devices, devices_args.num_devices)
            };
            let first_device = devices[0];
            assert!(!first_device.is_null());

            let device_get_description = api
                .PJRT_Device_GetDescription
                .expect("PJRT_Device_GetDescription must be exported");
            let mut get_description_args = PJRT_Device_GetDescription_Args {
                struct_size: size_of::<PJRT_Device_GetDescription_Args>(),
                extension_start: ptr::null_mut(),
                device: first_device,
                device_description: ptr::null_mut(),
            };
            check_ok(api, unsafe {
                device_get_description(&mut get_description_args)
            });
            assert!(!get_description_args.device_description.is_null());

            let description_id = api
                .PJRT_DeviceDescription_Id
                .expect("PJRT_DeviceDescription_Id must be exported");
            let mut id_args = PJRT_DeviceDescription_Id_Args {
                struct_size: size_of::<PJRT_DeviceDescription_Id_Args>(),
                extension_start: ptr::null_mut(),
                device_description: get_description_args.device_description,
                id: -1,
            };
            check_ok(api, unsafe { description_id(&mut id_args) });
            assert_eq!(id_args.id, 0);

            let description_kind = api
                .PJRT_DeviceDescription_Kind
                .expect("PJRT_DeviceDescription_Kind must be exported");
            let mut kind_args = PJRT_DeviceDescription_Kind_Args {
                struct_size: size_of::<PJRT_DeviceDescription_Kind_Args>(),
                extension_start: ptr::null_mut(),
                device_description: get_description_args.device_description,
                device_kind: ptr::null(),
                device_kind_size: 0,
            };
            check_ok(api, unsafe { description_kind(&mut kind_args) });
            let kind = unsafe {
                std::slice::from_raw_parts(
                    kind_args.device_kind.cast::<u8>(),
                    kind_args.device_kind_size,
                )
            };
            assert_eq!(kind, b"Tenstorrent");

            let device_get_attributes = api
                .PJRT_Device_GetAttributes
                .expect("PJRT_Device_GetAttributes must be exported");
            let mut device_get_attributes_args = PJRT_Device_GetAttributes_Args {
                struct_size: size_of::<PJRT_Device_GetAttributes_Args>(),
                extension_start: ptr::null_mut(),
                device: first_device,
                attributes: ptr::null(),
                num_attributes: usize::MAX,
                device_attributes: ptr::null_mut(),
                attributes_deleter: None,
            };
            check_ok(api, unsafe {
                device_get_attributes(&mut device_get_attributes_args)
            });
            assert!(device_get_attributes_args.attributes.is_null());
            assert_eq!(device_get_attributes_args.num_attributes, 0);
            assert!(!device_get_attributes_args.device_attributes.is_null());
            let deleter = device_get_attributes_args
                .attributes_deleter
                .expect("attributes deleter must be returned");
            unsafe { deleter(device_get_attributes_args.device_attributes) };
        } else {
            assert!(devices_args.devices.is_null());
        }

        let client_destroy = api
            .PJRT_Client_Destroy
            .expect("PJRT_Client_Destroy must be exported");
        let mut destroy_args = PJRT_Client_Destroy_Args {
            struct_size: size_of::<PJRT_Client_Destroy_Args>(),
            extension_start: ptr::null_mut(),
            client: create_args.client,
        };
        check_ok(api, unsafe { client_destroy(&mut destroy_args) });
        assert!(destroy_args.client.is_null());
    }

    #[test]
    fn surfaces_pjrt_errors_through_official_error_api() {
        let api = unsafe { &*GetPjrtApi() };
        let client_devices = api
            .PJRT_Client_Devices
            .expect("PJRT_Client_Devices must be exported");
        let mut args = PJRT_Client_Devices_Args {
            struct_size: size_of::<PJRT_Client_Devices_Args>(),
            extension_start: ptr::null_mut(),
            client: ptr::null_mut(),
            devices: ptr::null(),
            num_devices: 0,
        };
        let error = unsafe { client_devices(&mut args) };
        assert!(!error.is_null());

        let mut code_args = PJRT_Error_GetCode_Args {
            struct_size: size_of::<PJRT_Error_GetCode_Args>(),
            extension_start: ptr::null_mut(),
            error,
            code: PJRT_Error_Code::PJRT_Error_Code_UNKNOWN,
        };
        check_ok(api, unsafe {
            api.PJRT_Error_GetCode.expect("error get code must exist")(&mut code_args)
        });
        assert_eq!(
            code_args.code,
            PJRT_Error_Code::PJRT_Error_Code_INVALID_ARGUMENT
        );

        unsafe {
            api.PJRT_Error_Destroy.expect("error destroy must exist")(
                &mut PJRT_Error_Destroy_Args {
                    struct_size: size_of::<PJRT_Error_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    error,
                },
            );
        }
    }

    #[test]
    fn device_abstraction_surfaces_board_metadata_through_pjrt_objects() {
        let device = Device::from_probe(
            7,
            3,
            PathBuf::from("/dev/tenstorrent/3"),
            ProbeInfo {
                tensix_enabled_col_mask: 0x0fff,
                gddr_enabled_mask: 0x7f,
            },
        )
        .expect("device");
        let client = PJRT_Client::new_with_devices(vec![device]);

        let description = &client.device_descriptions[0];
        assert_eq!(description.id, 7);
        assert_eq!(description.device_kind.as_bytes(), b"Tenstorrent p100");
        let description_debug = std::str::from_utf8(description.debug_string.as_bytes())
            .expect("device debug string should be utf-8");
        assert!(
            description_debug.contains("board=p100"),
            "expected board marker in {description_debug}"
        );
        assert!(
            description_debug.contains("workers=118"),
            "expected worker count in {description_debug}"
        );
        assert!(
            description_debug.contains("cq=14,2/14,3"),
            "expected cq cores in {description_debug}"
        );
        assert!(
            description_debug.contains("path=/dev/tenstorrent/3"),
            "expected path marker in {description_debug}"
        );

        let memory = &client.memories[0];
        assert_eq!(memory.kind.as_bytes(), b"dram");
        let memory_debug = std::str::from_utf8(memory.debug_string.as_bytes())
            .expect("memory debug string should be utf-8");
        assert!(memory_debug.contains("dram_banks=7"));
        assert!(memory_debug.contains("harvested=[7]"));
        assert!(memory_debug.contains("tiles=21"));

        let device = &client.devices[0];
        assert_eq!(device.id, 7);
        assert_eq!(device.local_hardware_id, 3);
    }

    #[cfg(libtt_mlir_frontend)]
    #[derive(Clone, Copy, Debug)]
    enum BinaryOpKind {
        Add,
        Subtract,
        Multiply,
        Divide,
        Power,
    }

    #[cfg(libtt_mlir_frontend)]
    fn assert_binary_op(op: &executable::Op, expected: BinaryOpKind) {
        match (expected, op) {
            (
                BinaryOpKind::Add,
                executable::Op::Add {
                    input_ids,
                    output_id,
                },
            )
            | (
                BinaryOpKind::Subtract,
                executable::Op::Subtract {
                    input_ids,
                    output_id,
                },
            )
            | (
                BinaryOpKind::Multiply,
                executable::Op::Multiply {
                    input_ids,
                    output_id,
                },
            )
            | (
                BinaryOpKind::Divide,
                executable::Op::Divide {
                    input_ids,
                    output_id,
                },
            )
            | (
                BinaryOpKind::Power,
                executable::Op::Power {
                    input_ids,
                    output_id,
                },
            ) => {
                assert_eq!(*input_ids, [0, 1]);
                assert_eq!(*output_id, 2);
            }
            _ => panic!("expected {expected:?} op"),
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[derive(Clone, Copy, Debug)]
    enum UnaryOpKind {
        Cosine,
        Sine,
        Rsqrt,
        Negate,
        Exponential,
    }

    #[cfg(libtt_mlir_frontend)]
    fn assert_unary_op(op: &executable::Op, expected: UnaryOpKind) {
        match (expected, op) {
            (
                UnaryOpKind::Cosine,
                executable::Op::Cosine {
                    input_id,
                    output_id,
                },
            )
            | (
                UnaryOpKind::Sine,
                executable::Op::Sine {
                    input_id,
                    output_id,
                },
            )
            | (
                UnaryOpKind::Rsqrt,
                executable::Op::Rsqrt {
                    input_id,
                    output_id,
                },
            )
            | (
                UnaryOpKind::Negate,
                executable::Op::Negate {
                    input_id,
                    output_id,
                },
            )
            | (
                UnaryOpKind::Exponential,
                executable::Op::Exponential {
                    input_id,
                    output_id,
                },
            ) => {
                assert_eq!(*input_id, 0);
                assert_eq!(*output_id, 1);
            }
            _ => panic!("expected {expected:?} op"),
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_simple_binary_ops() {
        struct Case {
            name: &'static str,
            op_name: &'static str,
            ty: &'static str,
            expected: BinaryOpKind,
        }

        let cases = [
            Case {
                name: "add",
                op_name: "add",
                ty: "bf16",
                expected: BinaryOpKind::Add,
            },
            Case {
                name: "multiply",
                op_name: "multiply",
                ty: "bf16",
                expected: BinaryOpKind::Multiply,
            },
            Case {
                name: "subtract",
                op_name: "subtract",
                ty: "bf16",
                expected: BinaryOpKind::Subtract,
            },
            Case {
                name: "divide",
                op_name: "divide",
                ty: "bf16",
                expected: BinaryOpKind::Divide,
            },
            Case {
                name: "power",
                op_name: "power",
                ty: "f32",
                expected: BinaryOpKind::Power,
            },
        ];

        for case in cases {
            let code = format!(
                r#"module {{
  func.func public @main(%arg0: tensor<2x2x{ty}>, %arg1: tensor<2x2x{ty}>) -> tensor<2x2x{ty}> {{
    %0 = stablehlo.{op_name} %arg0, %arg1 : tensor<2x2x{ty}>
    return %0 : tensor<2x2x{ty}>
  }}
}}
"#,
                ty = case.ty,
                op_name = case.op_name
            );

            with_compiled_mlir_executable(&code, |executable| {
                assert_eq!(executable.values.len(), 3, "{} values", case.name);
                assert_eq!(executable.output_ids, vec![2], "{} outputs", case.name);
                assert_eq!(executable.ops.len(), 3, "{} ops", case.name);
                assert_binary_op(&executable.ops[2], case.expected);
            });
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_multi_op_stablehlo_add_chain() {
        with_compiled_mlir_executable(
            r#"module {
  func.func @main(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>, %arg2: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    %1 = "stablehlo.add"(%0, %arg2) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    return %1 : tensor<2x2xbf16>
  }
}
"#,
            |executable| {
                assert_eq!(executable.values.len(), 5);
                assert_eq!(executable.ops.len(), 5);
                assert_eq!(executable.output_ids, vec![4]);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_maximum_with_broadcast_constant() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<32x32xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![3]);
                assert_eq!(executable.ops.len(), 4);
                let executable::Op::Constant {
                    packed_value,
                    output_id,
                } = &executable.ops[1]
                else {
                    panic!("scalar constant should lower to Constant");
                };
                assert_eq!(*output_id, 1);
                assert_eq!(*packed_value, 0x3f80_3f80);
                let executable::Op::Constant {
                    packed_value,
                    output_id,
                } = &executable.ops[2]
                else {
                    panic!("broadcasted constant should lower to Constant");
                };
                assert_eq!(*output_id, 2);
                assert_eq!(*packed_value, 0x3f80_3f80);
                assert!(matches!(executable.ops[3], executable::Op::Max { .. }));
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_small_integer_splat_constant() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main() -> tensor<2x2xi1> {
    %0 = stablehlo.constant dense<true> : tensor<2x2xi1>
    return %0 : tensor<2x2xi1>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![0]);
                assert_eq!(executable.ops.len(), 1);
                let executable::Op::Constant {
                    packed_value,
                    output_id,
                } = &executable.ops[0]
                else {
                    panic!("predicate splat constant should lower to Constant");
                };
                assert_eq!(*output_id, 0);
                assert_eq!(*packed_value, 1);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_integer_compare_and_select() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %1 = stablehlo.compare LT, %arg0, %0, SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %c_0 = stablehlo.constant dense<288> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %3 = stablehlo.add %arg0, %2 : tensor<2xi32>
    %4 = stablehlo.select %1, %3, %arg0 : tensor<2xi1>, tensor<2xi32>
    return %4 : tensor<2xi32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![7]);
                assert_eq!(executable.ops.len(), 8);
                let executable::Op::Constant { output_id, .. } = &executable.ops[1] else {
                    panic!("scalar constant should lower to Constant");
                };
                assert_eq!(*output_id, 1);
                let executable::Op::Constant { output_id, .. } = &executable.ops[2] else {
                    panic!("broadcasted constant should lower to Constant");
                };
                assert_eq!(*output_id, 2);
                let executable::Op::Compare {
                    input_ids,
                    output_id,
                    direction,
                } = &executable.ops[3]
                else {
                    panic!("compare should lower to Compare");
                };
                assert_eq!(*input_ids, [0, 2]);
                assert_eq!(*output_id, 3);
                assert_eq!(*direction, executable::CompareDirection::Lt);
                let executable::Op::Constant { output_id, .. } = &executable.ops[4] else {
                    panic!("second scalar constant should lower to Constant");
                };
                assert_eq!(*output_id, 4);
                let executable::Op::Constant { output_id, .. } = &executable.ops[5] else {
                    panic!("second broadcasted constant should lower to Constant");
                };
                assert_eq!(*output_id, 5);
                let executable::Op::Add {
                    input_ids,
                    output_id,
                } = &executable.ops[6]
                else {
                    panic!("add should lower to Add");
                };
                assert_eq!(*input_ids, [0, 5]);
                assert_eq!(*output_id, 6);
                let executable::Op::Select {
                    input_ids,
                    output_id,
                } = &executable.ops[7]
                else {
                    panic!("select should lower to Select");
                };
                assert_eq!(*input_ids, [3, 6, 0]);
                assert_eq!(*output_id, 7);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_nonconstant_broadcast_in_dim() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2xi32>) -> tensor<2x1xi32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![1]);
                assert_eq!(executable.ops.len(), 2);
                let executable::Op::BroadcastInDim {
                    input_id,
                    output_id,
                    broadcast_dimensions,
                } = &executable.ops[1]
                else {
                    panic!("nonconstant broadcast should lower to BroadcastInDim");
                };
                assert_eq!(*input_id, 0);
                assert_eq!(*output_id, 1);
                assert_eq!(*broadcast_dimensions, vec![0]);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_gather() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<4x8xbf16>, %arg1: tensor<2x1xi32>) -> tensor<2x8xbf16> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<4x8xbf16>, tensor<2x1xi32>) -> tensor<2x8xbf16>
    return %0 : tensor<2x8xbf16>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![2]);
                assert_eq!(executable.ops.len(), 3);
                let executable::Op::Gather {
                    input_ids,
                    output_id,
                    dimension_numbers,
                    slice_sizes,
                    indices_are_sorted,
                } = &executable.ops[2]
                else {
                    panic!("gather should lower to Gather");
                };
                assert_eq!(*input_ids, [0, 1]);
                assert_eq!(*output_id, 2);
                assert_eq!(dimension_numbers.offset_dims, vec![1]);
                assert_eq!(dimension_numbers.collapsed_slice_dims, vec![0]);
                assert!(dimension_numbers.operand_batching_dims.is_empty());
                assert!(dimension_numbers.start_indices_batching_dims.is_empty());
                assert_eq!(dimension_numbers.start_index_map, vec![0]);
                assert_eq!(dimension_numbers.index_vector_dim, 1);
                assert_eq!(*slice_sizes, vec![1, 8]);
                assert!(!indices_are_sorted);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_iota() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main() -> tensor<2x3xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![0]);
                assert_eq!(executable.ops.len(), 1);
                let executable::Op::Iota {
                    output_id,
                    iota_dimension,
                } = &executable.ops[0]
                else {
                    panic!("iota should lower to Iota");
                };
                assert_eq!(*output_id, 0);
                assert_eq!(*iota_dimension, 1);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_concatenate() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2x4xbf16>, %arg1: tensor<3x4xbf16>) -> tensor<5x4xbf16> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2x4xbf16>, tensor<3x4xbf16>) -> tensor<5x4xbf16>
    return %0 : tensor<5x4xbf16>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![2]);
                assert_eq!(executable.ops.len(), 3);
                let executable::Op::Concatenate {
                    input_ids,
                    output_id,
                    dimension,
                } = &executable.ops[2]
                else {
                    panic!("concatenate should lower to Concatenate");
                };
                assert_eq!(input_ids.as_slice(), [0, 1]);
                assert_eq!(*output_id, 2);
                assert_eq!(*dimension, 0);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_simple_unary_ops() {
        struct Case {
            name: &'static str,
            op_name: &'static str,
            expected: UnaryOpKind,
        }

        let cases = [
            Case {
                name: "cosine",
                op_name: "cosine",
                expected: UnaryOpKind::Cosine,
            },
            Case {
                name: "sine",
                op_name: "sine",
                expected: UnaryOpKind::Sine,
            },
            Case {
                name: "rsqrt",
                op_name: "rsqrt",
                expected: UnaryOpKind::Rsqrt,
            },
            Case {
                name: "negate",
                op_name: "negate",
                expected: UnaryOpKind::Negate,
            },
            Case {
                name: "exponential",
                op_name: "exponential",
                expected: UnaryOpKind::Exponential,
            },
        ];

        for case in cases {
            let code = format!(
                r#"module {{
  func.func public @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {{
    %0 = stablehlo.{op_name} %arg0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }}
}}
"#,
                op_name = case.op_name
            );

            with_compiled_mlir_executable(&code, |executable| {
                assert_eq!(executable.output_ids, vec![1], "{} outputs", case.name);
                assert_eq!(executable.ops.len(), 2, "{} ops", case.name);
                assert_unary_op(&executable.ops[1], case.expected);
            });
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_reshape() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![1]);
                assert_eq!(executable.ops.len(), 2);
                assert_eq!(executable.values[1].dims, vec![3, 2]);
                let executable::Op::Reshape {
                    input_id,
                    output_id,
                } = &executable.ops[1]
                else {
                    panic!("reshape should lower to Reshape");
                };
                assert_eq!(*input_id, 0);
                assert_eq!(*output_id, 1);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_slice() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<4x4xf32>) -> tensor<2x2xf32> {
    %0 = "stablehlo.slice"(%arg0) {
      start_indices = array<i64: 0, 1>,
      limit_indices = array<i64: 4, 3>,
      strides = array<i64: 2, 1>
    } : (tensor<4x4xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![1]);
                assert_eq!(executable.ops.len(), 2);
                assert_eq!(executable.values[1].dims, vec![2, 2]);
                let executable::Op::Slice {
                    input_id,
                    output_id,
                    start_indices,
                    limit_indices,
                    strides,
                } = &executable.ops[1]
                else {
                    panic!("slice should lower to Slice");
                };
                assert_eq!(*input_id, 0);
                assert_eq!(*output_id, 1);
                assert_eq!(start_indices, &vec![0, 1]);
                assert_eq!(limit_indices, &vec![4, 3]);
                assert_eq!(strides, &vec![2, 1]);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_transpose() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![1]);
                assert_eq!(executable.ops.len(), 2);
                assert_eq!(executable.values[1].dims, vec![4, 2, 3]);
                let executable::Op::Transpose {
                    input_id,
                    output_id,
                    permutation,
                } = &executable.ops[1]
                else {
                    panic!("transpose should lower to Transpose");
                };
                assert_eq!(*input_id, 0);
                assert_eq!(*output_id, 1);
                assert_eq!(permutation, &vec![2, 0, 1]);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_custom_call() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.custom_call @foo(%arg0) {
      has_side_effect = false
    } : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![1]);
                assert_eq!(executable.ops.len(), 2);
                let executable::Op::CustomCall {
                    input_ids,
                    output_id,
                    call_target_name,
                    has_side_effect,
                } = &executable.ops[1]
                else {
                    panic!("custom_call should lower to CustomCall");
                };
                assert_eq!(input_ids, &vec![0]);
                assert_eq!(*output_id, 1);
                assert_eq!(call_target_name, "foo");
                assert!(!has_side_effect);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_batched_dot_general() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2x3x4xf32>, %arg1: tensor<5x3x4xf32>) -> tensor<3x2x5xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1,
      batching_dims = [1] x [1],
      contracting_dims = [2] x [2]
      : (tensor<2x3x4xf32>, tensor<5x3x4xf32>) -> tensor<3x2x5xf32>
    return %0 : tensor<3x2x5xf32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![2]);
                assert_eq!(executable.ops.len(), 3);
                let executable::Op::Matmul {
                    input_ids,
                    output_id,
                    dimension_numbers,
                } = &executable.ops[2]
                else {
                    panic!("dot_general should lower to Matmul");
                };
                assert_eq!(*input_ids, [0, 1]);
                assert_eq!(*output_id, 2);
                assert_eq!(dimension_numbers.lhs_batching_dimensions, vec![1]);
                assert_eq!(dimension_numbers.rhs_batching_dimensions, vec![1]);
                assert_eq!(dimension_numbers.lhs_contracting_dimensions, vec![2]);
                assert_eq!(dimension_numbers.rhs_contracting_dimensions, vec![2]);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_convert() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2x2xf32>) -> tensor<2x2xbf16>
    return %0 : tensor<2x2xbf16>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![1]);
                assert_eq!(executable.ops.len(), 2);
                let executable::Op::Convert {
                    input_id,
                    output_id,
                } = &executable.ops[1]
                else {
                    panic!("convert should lower to Convert");
                };
                assert_eq!(*input_id, 0);
                assert_eq!(*output_id, 1);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_convert_of_scalar_constant() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main() -> tensor<bf16> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %cst : (tensor<f32>) -> tensor<bf16>
    return %0 : tensor<bf16>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![1]);
                assert_eq!(executable.ops.len(), 2);
                let executable::Op::Constant {
                    packed_value,
                    output_id,
                } = &executable.ops[0]
                else {
                    panic!("scalar constant should lower to Constant");
                };
                assert_eq!(*output_id, 0);
                assert_eq!(*packed_value, 0x3f80_0000);
                let executable::Op::Convert {
                    input_id,
                    output_id,
                } = &executable.ops[1]
                else {
                    panic!("convert should lower to Convert");
                };
                assert_eq!(*input_id, 0);
                assert_eq!(*output_id, 1);
            },
        );
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_reduce() {
        with_compiled_mlir_executable(
            r#"module {
  func.func public @main(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = "stablehlo.reduce"(%arg0, %cst) ({
      ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
        %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
        stablehlo.return %1 : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
"#,
            |executable| {
                assert_eq!(executable.output_ids, vec![2]);
                assert_eq!(executable.ops.len(), 3);
                let executable::Op::Constant { output_id, .. } = &executable.ops[1] else {
                    panic!("init constant should lower to Constant");
                };
                assert_eq!(*output_id, 1);
                let executable::Op::Reduce {
                    input_ids,
                    init_value_ids,
                    output_id,
                    dimensions,
                    reducer,
                } = &executable.ops[2]
                else {
                    panic!("reduce should lower to Reduce");
                };
                assert_eq!(input_ids, &vec![0]);
                assert_eq!(init_value_ids, &vec![1]);
                assert_eq!(*output_id, 2);
                assert_eq!(dimensions, &vec![1]);
                assert_eq!(*reducer, executable::ReduceReducer::Add);
            },
        );
    }
}
