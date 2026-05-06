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
const TILED_BUFFER_DIM: usize = 32;

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

fn round_up_to_tiled_dim(dim: usize) -> Result<usize, *mut PJRT_Error> {
    dim.max(1)
        .checked_add(TILED_BUFFER_DIM - 1)
        .map(|value| value / TILED_BUFFER_DIM * TILED_BUFFER_DIM)
        .ok_or_else(|| resource_exhausted("shape dimension overflow"))
}

fn tiled_allocation_shape(shape: &[usize]) -> Result<Vec<usize>, *mut PJRT_Error> {
    match shape.len() {
        0 => Ok(vec![TILED_BUFFER_DIM, TILED_BUFFER_DIM]),
        1 => Ok(vec![TILED_BUFFER_DIM, round_up_to_tiled_dim(shape[0])?]),
        _ => Ok(shape.to_vec()),
    }
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
    padded[..data.len()].copy_from_slice(data);
    Ok(Some(padded))
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
    let mut data = read_buffer_bytes(buffer)?;
    if data.len() == byte_size {
        return Ok(data);
    }
    if buffer.dims.len() < 2 && data.len() >= byte_size {
        data.truncate(byte_size);
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

                if analysis.outputs.len() != 1 {
                    return Err(unimplemented(format!(
                        "TT executable must contain exactly one output, got {}",
                        analysis.outputs.len()
                    )));
                }
                let output = analysis
                    .outputs
                    .first()
                    .expect("analysis output length was checked");
                return Ok(make_executable_metadata(
                    EXECUTABLE_NAME,
                    output.dims.clone(),
                    output.element_type,
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
    dims: Vec<i64>,
    output_type: PJRT_Buffer_Type,
    executable: Option<executable::Executable>,
) -> ExecutableMetadata {
    let output_memory_kinds = vec![cstring_lossy("dram")];
    let output_memory_kind_ptrs = output_memory_kinds
        .iter()
        .map(|kind| kind.as_ptr())
        .collect::<Vec<_>>();
    let output_memory_kind_sizes = output_memory_kinds
        .iter()
        .map(|kind| kind.as_bytes().len())
        .collect::<Vec<_>>();
    let output_dim_sizes = vec![dims.len()];
    let fingerprint = executable_fingerprint_string(name, &dims, output_type);
    ExecutableMetadata {
        name: cstring_lossy(name),
        fingerprint,
        num_outputs: 1,
        output_types: vec![output_type],
        output_dims: dims,
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
fn executable_fingerprint_string(
    name: &str,
    dims: &[i64],
    output_type: PJRT_Buffer_Type,
) -> CString {
    let dims = dims
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join("x");
    cstring_lossy(&format!(
        "tt:executable_v1:name={name}:dims={dims}:type={}:v1",
        output_type as u32
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

fn bf16_eltwise_input<'a>(
    values: &'a [Option<PJRT_Buffer>],
    plan: &'a executable::Executable,
    value_id: u32,
    field: &str,
) -> Result<kernels::binary_eltwise::Bf16EltwiseInput<'a>, *mut PJRT_Error> {
    let index = value_id as usize;
    if let Some(buffer) = values.get(index).and_then(|value| value.as_ref()) {
        let Some(dram_buffer) = buffer.dram_buffer.as_ref() else {
            return Err(failed_precondition(format!(
                "TT executable {field} buffer has no device allocation"
            )));
        };
        return Ok(kernels::binary_eltwise::Bf16EltwiseInput::Dram(dram_buffer));
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
                if desc.element_type != PJRT_Buffer_Type::PJRT_Buffer_Type_BF16 {
                    return Err(unimplemented(format!(
                        "{field} constant value id {value_id} has type {:?}; bf16 eltwise constants currently require bf16",
                        desc.element_type
                    )));
                }
                return Ok(kernels::binary_eltwise::Bf16EltwiseInput::Constant(
                    *packed_value,
                ));
            }
        }
    }
    Err(invalid_argument(format!(
        "{field} value id {value_id} is not available"
    )))
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
    if expected.element_type != PJRT_Buffer_Type::PJRT_Buffer_Type_BF16 {
        return Err(unimplemented(format!(
            "TT executable {op} currently only supports bf16 outputs"
        )));
    }
    if expected.dims != expected_dims {
        return Err(invalid_argument(format!(
            "TT executable {op} output shape mismatch: expected {:?}, got {:?}",
            expected.dims, expected_dims
        )));
    }
    values[output_index] = Some(PJRT_Buffer {
        buffer_type: PJRT_Buffer_Type::PJRT_Buffer_Type_BF16,
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
    if lhs_desc.element_type != PJRT_Buffer_Type::PJRT_Buffer_Type_BF16
        || rhs_desc.element_type != PJRT_Buffer_Type::PJRT_Buffer_Type_BF16
    {
        return Err(unimplemented(format!(
            "TT executable {op_name} currently only supports bf16 buffers"
        )));
    }
    if lhs_desc.dims != rhs_desc.dims {
        return Err(invalid_argument(format!(
            "TT executable {op_name} input shapes must match"
        )));
    }

    let output_dims = lhs_desc.dims.clone();
    let shape = dims_i64_to_usize(&output_dims)?;
    let lhs_input = bf16_eltwise_input(values, plan, input_ids[0], &lhs_field)?;
    let rhs_input = bf16_eltwise_input(values, plan, input_ids[1], &rhs_field)?;
    let output_name = format!("pjrt_{op_name}");
    let output_dram = kernels::binary_eltwise::eltwise_bf16(
        device,
        op,
        lhs_input,
        rhs_input,
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

fn execute_executable_v1(
    executable: &PJRT_LoadedExecutable,
    execute_device: *mut PJRT_Device,
    target_device: &mut PJRT_Device,
    inputs: &[*mut PJRT_Buffer],
) -> Result<PJRT_Buffer, *mut PJRT_Error> {
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
            executable::Op::Multiply { .. } => {
                return Err(unimplemented(
                    "TT executable multiply execution is not currently supported",
                ));
            }
            executable::Op::Divide { .. } => {
                return Err(unimplemented(
                    "TT executable divide execution is not currently supported",
                ));
            }
            executable::Op::Power { .. } => {
                return Err(unimplemented(
                    "TT executable power execution is not currently supported",
                ));
            }
            executable::Op::Concatenate { .. } => {
                return Err(unimplemented(
                    "TT executable concatenate execution is not currently supported",
                ));
            }
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
            executable::Op::Compare { .. } => {
                return Err(unimplemented(
                    "TT executable compare execution is not currently supported",
                ));
            }
            executable::Op::Select { .. } => {
                return Err(unimplemented(
                    "TT executable select execution is not currently supported",
                ));
            }
            executable::Op::BroadcastInDim { .. } => {
                return Err(unimplemented(
                    "TT executable broadcast_in_dim execution is not currently supported",
                ));
            }
            executable::Op::Gather { .. } => {
                return Err(unimplemented(
                    "TT executable gather execution is not currently supported",
                ));
            }
            executable::Op::Iota { .. } => {
                return Err(unimplemented(
                    "TT executable iota execution is not currently supported",
                ));
            }
        }
    }

    let output = device_buffer_for_value(&values, plan.output_ids[0], "output")?;
    Ok(output.clone())
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
    if args.argument_lists.is_null() || args.output_lists.is_null() {
        return invalid_argument("argument_lists and output_lists must not be null");
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

    let device_args = unsafe { *args.argument_lists };
    if device_args.is_null() {
        return invalid_argument("argument_lists[0] must not be null");
    }
    let input_ptrs = if args.num_args == 0 {
        &[][..]
    } else {
        unsafe { slice::from_raw_parts(device_args, args.num_args) }
    };
    let output_buffer =
        match execute_executable_v1(executable, execute_device, target_device, input_ptrs) {
            Ok(output) => output,
            Err(err) => return err,
        };

    let device_outputs = unsafe { *args.output_lists };
    if device_outputs.is_null() {
        return invalid_argument("output_lists[0] must not be null");
    }
    let output_ptr = Box::into_raw(Box::new(output_buffer));
    unsafe {
        *device_outputs.add(0) = output_ptr;
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
    let allocation_shape = match tiled_allocation_shape(&shape) {
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

    #[test]
    fn tiled_allocation_shape_pads_scalar_and_vector_buffers() {
        assert_eq!(
            tiled_allocation_shape(&[]).expect("scalar shape should pad"),
            vec![32, 32]
        );
        assert_eq!(
            tiled_allocation_shape(&[5]).expect("vector shape should pad"),
            vec![32, 32]
        );
        assert_eq!(
            tiled_allocation_shape(&[128]).expect("aligned vector shape should pad rank"),
            vec![32, 128]
        );
        assert_eq!(
            tiled_allocation_shape(&[32, 64]).expect("rank-2 shape should be preserved"),
            vec![32, 64]
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
    #[test]
    fn pjrt_compile_uses_mlir_frontend_for_generic_stablehlo_add() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func @main(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    return %0 : tensor<2x2xbf16>
  }
}
"#
        .to_vec();
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
        assert_eq!(executable.values.len(), 3);
        assert_eq!(executable.ops.len(), 3);
        assert_eq!(executable.output_ids, vec![2]);

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_multi_op_stablehlo_add_chain() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func @main(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>, %arg2: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    %1 = "stablehlo.add"(%0, %arg2) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    return %1 : tensor<2x2xbf16>
  }
}
"#
        .to_vec();
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
        assert_eq!(executable.values.len(), 5);
        assert_eq!(executable.ops.len(), 5);
        assert_eq!(executable.output_ids, vec![4]);

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_maximum_with_broadcast_constant() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func public @main(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<32x32xbf16>
    %1 = stablehlo.maximum %arg0, %0 : tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
"#
        .to_vec();
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
        assert_eq!(executable.output_ids, vec![3]);
        assert_eq!(executable.ops.len(), 3);
        let executable::Op::Constant {
            packed_value,
            output_id,
        } = &executable.ops[1]
        else {
            panic!("broadcasted constant should lower to Constant");
        };
        assert_eq!(*output_id, 2);
        assert_eq!(*packed_value, 0x3f80_3f80);
        assert!(matches!(executable.ops[2], executable::Op::Max { .. }));

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_integer_compare_and_select() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
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
"#
        .to_vec();
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
        assert_eq!(executable.output_ids, vec![7]);
        assert_eq!(executable.ops.len(), 6);
        let executable::Op::Constant { output_id, .. } = &executable.ops[1] else {
            panic!("broadcasted constant should lower to Constant");
        };
        assert_eq!(*output_id, 2);
        let executable::Op::Compare {
            input_ids,
            output_id,
            direction,
        } = &executable.ops[2]
        else {
            panic!("compare should lower to Compare");
        };
        assert_eq!(*input_ids, [0, 2]);
        assert_eq!(*output_id, 3);
        assert_eq!(*direction, executable::CompareDirection::Lt);
        let executable::Op::Constant { output_id, .. } = &executable.ops[3] else {
            panic!("second broadcasted constant should lower to Constant");
        };
        assert_eq!(*output_id, 5);
        let executable::Op::Add {
            input_ids,
            output_id,
        } = &executable.ops[4]
        else {
            panic!("add should lower to Add");
        };
        assert_eq!(*input_ids, [0, 5]);
        assert_eq!(*output_id, 6);
        let executable::Op::Select {
            input_ids,
            output_id,
        } = &executable.ops[5]
        else {
            panic!("select should lower to Select");
        };
        assert_eq!(*input_ids, [3, 6, 0]);
        assert_eq!(*output_id, 7);

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_nonconstant_broadcast_in_dim() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func public @main(%arg0: tensor<2xi32>) -> tensor<2x1xi32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<2xi32>) -> tensor<2x1xi32>
    return %0 : tensor<2x1xi32>
  }
}
"#
        .to_vec();
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

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_gather() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func public @main(%arg0: tensor<4x8xbf16>, %arg1: tensor<2x1xi32>) -> tensor<2x8xbf16> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 8>}> : (tensor<4x8xbf16>, tensor<2x1xi32>) -> tensor<2x8xbf16>
    return %0 : tensor<2x8xbf16>
  }
}
"#
        .to_vec();
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

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_iota() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func public @main() -> tensor<2x3xi32> {
    %0 = stablehlo.iota dim = 1 : tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}
"#
        .to_vec();
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

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_multiply() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func public @main(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x2xbf16>
    return %0 : tensor<2x2xbf16>
  }
}
"#
        .to_vec();
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
        assert_eq!(executable.output_ids, vec![2]);
        assert_eq!(executable.ops.len(), 3);
        let executable::Op::Multiply {
            input_ids,
            output_id,
        } = &executable.ops[2]
        else {
            panic!("multiply should lower to Multiply");
        };
        assert_eq!(*input_ids, [0, 1]);
        assert_eq!(*output_id, 2);

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_divide() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func public @main(%arg0: tensor<2x2xbf16>, %arg1: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<2x2xbf16>
    return %0 : tensor<2x2xbf16>
  }
}
"#
        .to_vec();
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
        assert_eq!(executable.output_ids, vec![2]);
        assert_eq!(executable.ops.len(), 3);
        let executable::Op::Divide {
            input_ids,
            output_id,
        } = &executable.ops[2]
        else {
            panic!("divide should lower to Divide");
        };
        assert_eq!(*input_ids, [0, 1]);
        assert_eq!(*output_id, 2);

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_power() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func public @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.power %arg0, %arg1 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
"#
        .to_vec();
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
        assert_eq!(executable.output_ids, vec![2]);
        assert_eq!(executable.ops.len(), 3);
        let executable::Op::Power {
            input_ids,
            output_id,
        } = &executable.ops[2]
        else {
            panic!("power should lower to Power");
        };
        assert_eq!(*input_ids, [0, 1]);
        assert_eq!(*output_id, 2);

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }

    #[cfg(libtt_mlir_frontend)]
    #[test]
    fn pjrt_compile_lowers_concatenate() {
        let api = unsafe { &*GetPjrtApi() };
        let client = Box::into_raw(Box::new(PJRT_Client::new_with_devices(Vec::new())));
        let mut format = b"mlir".to_vec();
        let mut code = br#"module {
  func.func public @main(%arg0: tensor<2x4xbf16>, %arg1: tensor<3x4xbf16>) -> tensor<5x4xbf16> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<2x4xbf16>, tensor<3x4xbf16>) -> tensor<5x4xbf16>
    return %0 : tensor<5x4xbf16>
  }
}
"#
        .to_vec();
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
        assert_eq!(*input_ids, vec![0, 1]);
        assert_eq!(*output_id, 2);
        assert_eq!(*dimension, 0);

        unsafe {
            drop(Box::from_raw(get_executable_args.executable));
            drop(Box::from_raw(compile_args.executable));
            drop(Box::from_raw(client));
        }
    }
}
