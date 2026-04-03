#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::{CString, c_char, c_void};
use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::ptr;

const PJRT_API_MAJOR: i32 = 0;
const PJRT_API_MINOR: i32 = 96;
const PJRT_API_UNUSED_TAIL_SLOTS: usize = 35;

type PjrtOpaqueFn = Option<unsafe extern "C" fn()>;
type PjrtResultFn<Args> = Option<unsafe extern "C" fn(args: *mut Args) -> *mut PJRT_Error>;
type PjrtVoidFn<Args> = Option<unsafe extern "C" fn(args: *mut Args)>;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PJRT_Error_Code {
    PJRT_Error_Code_OK = 0,
    PJRT_Error_Code_CANCELLED = 1,
    PJRT_Error_Code_UNKNOWN = 2,
    PJRT_Error_Code_INVALID_ARGUMENT = 3,
    PJRT_Error_Code_DEADLINE_EXCEEDED = 4,
    PJRT_Error_Code_NOT_FOUND = 5,
    PJRT_Error_Code_ALREADY_EXISTS = 6,
    PJRT_Error_Code_PERMISSION_DENIED = 7,
    PJRT_Error_Code_RESOURCE_EXHAUSTED = 8,
    PJRT_Error_Code_FAILED_PRECONDITION = 9,
    PJRT_Error_Code_ABORTED = 10,
    PJRT_Error_Code_OUT_OF_RANGE = 11,
    PJRT_Error_Code_UNIMPLEMENTED = 12,
    PJRT_Error_Code_INTERNAL = 13,
    PJRT_Error_Code_UNAVAILABLE = 14,
    PJRT_Error_Code_DATA_LOSS = 15,
    PJRT_Error_Code_UNAUTHENTICATED = 16,
}

#[repr(C)]
pub struct PJRT_Extension_Base {
    _private: [u8; 0],
}

#[repr(C)]
pub struct PJRT_NamedValue {
    _private: [u8; 0],
}

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
}

#[repr(C)]
pub struct PJRT_Device {
    id: i32,
    local_hardware_id: i32,
    description: *mut PJRT_DeviceDescription,
    addressable: bool,
    default_memory: *mut PJRT_Memory,
}

#[repr(C)]
pub struct PJRT_Client {
    platform_name: CString,
    platform_version: CString,
    topology: PJRT_TopologyDescription,
    device_descriptions: Vec<Box<PJRT_DeviceDescription>>,
    memories: Vec<Box<PJRT_Memory>>,
    devices: Vec<Box<PJRT_Device>>,
    device_ptrs: Vec<*mut PJRT_Device>,
    addressable_device_ptrs: Vec<*mut PJRT_Device>,
}

#[repr(C)]
pub struct PJRT_Api_Version {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub major_version: i32,
    pub minor_version: i32,
}

#[repr(C)]
pub struct PJRT_Error_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub error: *mut PJRT_Error,
}

#[repr(C)]
pub struct PJRT_Error_Message_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub error: *const PJRT_Error,
    pub message: *const c_char,
    pub message_size: usize,
}

#[repr(C)]
pub struct PJRT_Error_GetCode_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub error: *const PJRT_Error,
    pub code: PJRT_Error_Code,
}

#[repr(C)]
pub struct PJRT_Plugin_Initialize_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
}

#[repr(C)]
pub struct PJRT_Plugin_Attributes_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub attributes: *const PJRT_NamedValue,
    pub num_attributes: usize,
}

#[repr(C)]
pub struct PJRT_Client_Create_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub create_options: *const PJRT_NamedValue,
    pub num_options: usize,
    pub kv_get_callback: PjrtOpaqueFn,
    pub kv_get_user_arg: *mut c_void,
    pub kv_put_callback: PjrtOpaqueFn,
    pub kv_put_user_arg: *mut c_void,
    pub client: *mut PJRT_Client,
    pub kv_try_get_callback: PjrtOpaqueFn,
    pub kv_try_get_user_arg: *mut c_void,
}

#[repr(C)]
pub struct PJRT_Client_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
}

#[repr(C)]
pub struct PJRT_Client_PlatformName_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub platform_name: *const c_char,
    pub platform_name_size: usize,
}

#[repr(C)]
pub struct PJRT_Client_ProcessIndex_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub process_index: i32,
}

#[repr(C)]
pub struct PJRT_Client_PlatformVersion_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub platform_version: *const c_char,
    pub platform_version_size: usize,
}

#[repr(C)]
pub struct PJRT_Client_TopologyDescription_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub topology: *mut PJRT_TopologyDescription,
}

#[repr(C)]
pub struct PJRT_Client_Devices_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub devices: *const *mut PJRT_Device,
    pub num_devices: usize,
}

#[repr(C)]
pub struct PJRT_Client_AddressableDevices_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub addressable_devices: *const *mut PJRT_Device,
    pub num_addressable_devices: usize,
}

#[repr(C)]
pub struct PJRT_Client_LookupDevice_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub id: i32,
    pub device: *mut PJRT_Device,
}

#[repr(C)]
pub struct PJRT_Client_LookupAddressableDevice_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub local_hardware_id: i32,
    pub addressable_device: *mut PJRT_Device,
}

#[repr(C)]
pub struct PJRT_Client_AddressableMemories_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub addressable_memories: *const *mut c_void,
    pub num_addressable_memories: usize,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_Id_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub id: i32,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_ProcessIndex_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub process_index: i32,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_Attributes_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub num_attributes: usize,
    pub attributes: *const PJRT_NamedValue,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_Kind_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub device_kind: *const c_char,
    pub device_kind_size: usize,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_DebugString_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub debug_string: *const c_char,
    pub debug_string_size: usize,
}

#[repr(C)]
pub struct PJRT_DeviceDescription_ToString_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device_description: *mut PJRT_DeviceDescription,
    pub to_string: *const c_char,
    pub to_string_size: usize,
}

#[repr(C)]
pub struct PJRT_Device_GetDescription_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device: *mut PJRT_Device,
    pub device_description: *mut PJRT_DeviceDescription,
}

#[repr(C)]
pub struct PJRT_Device_IsAddressable_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device: *mut PJRT_Device,
    pub is_addressable: bool,
}

#[repr(C)]
pub struct PJRT_Device_LocalHardwareId_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device: *mut PJRT_Device,
    pub local_hardware_id: i32,
}

#[repr(C)]
pub struct PJRT_Device_AddressableMemories_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device: *mut PJRT_Device,
    pub memories: *const *mut PJRT_Memory,
    pub num_memories: usize,
}

#[repr(C)]
pub struct PJRT_Device_DefaultMemory_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub device: *mut PJRT_Device,
    pub default_memory: *mut PJRT_Memory,
}

#[repr(C)]
pub struct PJRT_Memory_Id_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub memory: *mut PJRT_Memory,
    pub id: i32,
}

#[repr(C)]
pub struct PJRT_Memory_Kind_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub memory: *mut PJRT_Memory,
    pub kind: *const c_char,
    pub kind_size: usize,
}

#[repr(C)]
pub struct PJRT_Memory_DebugString_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub memory: *mut PJRT_Memory,
    pub debug_string: *const c_char,
    pub debug_string_size: usize,
}

#[repr(C)]
pub struct PJRT_Memory_ToString_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub memory: *mut PJRT_Memory,
    pub to_string: *const c_char,
    pub to_string_size: usize,
}

#[repr(C)]
pub struct PJRT_Memory_AddressableByDevices_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub memory: *mut PJRT_Memory,
    pub devices: *const *mut PJRT_Device,
    pub num_devices: usize,
}

#[repr(C)]
pub struct PJRT_TopologyDescription_PlatformName_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub topology: *const PJRT_TopologyDescription,
    pub platform_name: *const c_char,
    pub platform_name_size: usize,
}

#[repr(C)]
pub struct PJRT_TopologyDescription_PlatformVersion_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub topology: *mut PJRT_TopologyDescription,
    pub platform_version: *const c_char,
    pub platform_version_size: usize,
}

#[repr(C)]
pub struct PJRT_TopologyDescription_GetDeviceDescriptions_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub topology: *const PJRT_TopologyDescription,
    pub descriptions: *const *mut PJRT_DeviceDescription,
    pub num_descriptions: usize,
}

#[repr(C)]
pub struct PJRT_TopologyDescription_Attributes_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub topology: *mut PJRT_TopologyDescription,
    pub attributes: *const PJRT_NamedValue,
    pub num_attributes: usize,
}

#[repr(C)]
pub struct PJRT_Api {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub pjrt_api_version: PJRT_Api_Version,
    pub PJRT_Error_Destroy: PjrtVoidFn<PJRT_Error_Destroy_Args>,
    pub PJRT_Error_Message: PjrtVoidFn<PJRT_Error_Message_Args>,
    pub PJRT_Error_GetCode: PjrtResultFn<PJRT_Error_GetCode_Args>,
    pub PJRT_Plugin_Initialize: PjrtResultFn<PJRT_Plugin_Initialize_Args>,
    pub PJRT_Plugin_Attributes: PjrtResultFn<PJRT_Plugin_Attributes_Args>,
    unused_events: [PjrtOpaqueFn; 5],
    pub PJRT_Client_Create: PjrtResultFn<PJRT_Client_Create_Args>,
    pub PJRT_Client_Destroy: PjrtResultFn<PJRT_Client_Destroy_Args>,
    pub PJRT_Client_PlatformName: PjrtResultFn<PJRT_Client_PlatformName_Args>,
    pub PJRT_Client_ProcessIndex: PjrtResultFn<PJRT_Client_ProcessIndex_Args>,
    pub PJRT_Client_PlatformVersion: PjrtResultFn<PJRT_Client_PlatformVersion_Args>,
    pub PJRT_Client_Devices: PjrtResultFn<PJRT_Client_Devices_Args>,
    pub PJRT_Client_AddressableDevices: PjrtResultFn<PJRT_Client_AddressableDevices_Args>,
    pub PJRT_Client_LookupDevice: PjrtResultFn<PJRT_Client_LookupDevice_Args>,
    pub PJRT_Client_LookupAddressableDevice:
        PjrtResultFn<PJRT_Client_LookupAddressableDevice_Args>,
    pub PJRT_Client_AddressableMemories: PjrtResultFn<PJRT_Client_AddressableMemories_Args>,
    unused_client_rest: [PjrtOpaqueFn; 3],
    pub PJRT_DeviceDescription_Id: PjrtResultFn<PJRT_DeviceDescription_Id_Args>,
    pub PJRT_DeviceDescription_ProcessIndex: PjrtResultFn<PJRT_DeviceDescription_ProcessIndex_Args>,
    pub PJRT_DeviceDescription_Attributes: PjrtResultFn<PJRT_DeviceDescription_Attributes_Args>,
    pub PJRT_DeviceDescription_Kind: PjrtResultFn<PJRT_DeviceDescription_Kind_Args>,
    pub PJRT_DeviceDescription_DebugString: PjrtResultFn<PJRT_DeviceDescription_DebugString_Args>,
    pub PJRT_DeviceDescription_ToString: PjrtResultFn<PJRT_DeviceDescription_ToString_Args>,
    pub PJRT_Device_GetDescription: PjrtResultFn<PJRT_Device_GetDescription_Args>,
    pub PJRT_Device_IsAddressable: PjrtResultFn<PJRT_Device_IsAddressable_Args>,
    pub PJRT_Device_LocalHardwareId: PjrtResultFn<PJRT_Device_LocalHardwareId_Args>,
    pub PJRT_Device_AddressableMemories: PjrtResultFn<PJRT_Device_AddressableMemories_Args>,
    pub PJRT_Device_DefaultMemory: PjrtResultFn<PJRT_Device_DefaultMemory_Args>,
    unused_device_memory_stats: [PjrtOpaqueFn; 1],
    pub PJRT_Memory_Id: PjrtResultFn<PJRT_Memory_Id_Args>,
    pub PJRT_Memory_Kind: PjrtResultFn<PJRT_Memory_Kind_Args>,
    pub PJRT_Memory_DebugString: PjrtResultFn<PJRT_Memory_DebugString_Args>,
    pub PJRT_Memory_ToString: PjrtResultFn<PJRT_Memory_ToString_Args>,
    pub PJRT_Memory_AddressableByDevices: PjrtResultFn<PJRT_Memory_AddressableByDevices_Args>,
    unused_before_topology: [PjrtOpaqueFn; 42],
    unused_topology_create_destroy: [PjrtOpaqueFn; 2],
    pub PJRT_TopologyDescription_PlatformName:
        PjrtResultFn<PJRT_TopologyDescription_PlatformName_Args>,
    pub PJRT_TopologyDescription_PlatformVersion:
        PjrtResultFn<PJRT_TopologyDescription_PlatformVersion_Args>,
    pub PJRT_TopologyDescription_GetDeviceDescriptions:
        PjrtResultFn<PJRT_TopologyDescription_GetDeviceDescriptions_Args>,
    unused_topology_serialize: [PjrtOpaqueFn; 1],
    pub PJRT_TopologyDescription_Attributes:
        PjrtResultFn<PJRT_TopologyDescription_Attributes_Args>,
    unused_before_client_topology: [PjrtOpaqueFn; 6],
    pub PJRT_Client_TopologyDescription: PjrtResultFn<PJRT_Client_TopologyDescription_Args>,
    unused_tail: [PjrtOpaqueFn; PJRT_API_UNUSED_TAIL_SLOTS],
}

// The API table is immutable process-global data.
unsafe impl Sync for PJRT_Api {}

impl PJRT_Client {
    fn new() -> Self {
        let discovered = discover_devices();
        let mut device_descriptions = Vec::with_capacity(discovered.len());

        for (id, _) in discovered.iter().enumerate() {
            device_descriptions.push(Box::new(PJRT_DeviceDescription {
                id: id as i32,
                process_index: 0,
                device_kind: cstring_lossy("Tenstorrent"),
                debug_string: cstring_lossy(format!("Tenstorrent device {id}")),
                to_string: cstring_lossy(format!("tt:{id}")),
            }));
        }

        let mut memories = Vec::with_capacity(discovered.len());
        for (index, _) in discovered.iter().enumerate() {
            memories.push(Box::new(PJRT_Memory {
                id: index as i32,
                kind: cstring_lossy("device"),
                debug_string: cstring_lossy(format!("Tenstorrent memory {index}")),
                to_string: cstring_lossy(format!("tt:memory:{index}")),
            }));
        }

        let mut memory_ptrs = Vec::with_capacity(memories.len());
        for memory in &mut memories {
            memory_ptrs.push(&mut **memory as *mut PJRT_Memory);
        }

        let mut devices = Vec::with_capacity(discovered.len());
        for (index, _) in discovered.iter().enumerate() {
            let description = &mut *device_descriptions[index] as *mut PJRT_DeviceDescription;
            let default_memory = memory_ptrs[index];
            devices.push(Box::new(PJRT_Device {
                id: index as i32,
                local_hardware_id: index as i32,
                description,
                addressable: true,
                default_memory,
            }));
        }

        let mut device_ptrs = Vec::with_capacity(devices.len());
        for device in &mut devices {
            device_ptrs.push(&mut **device as *mut PJRT_Device);
        }
        let addressable_device_ptrs = device_ptrs.clone();

        let topology = PJRT_TopologyDescription {
            platform_name: cstring_lossy("tt"),
            platform_version: cstring_lossy(format!("libtt {}", env!("CARGO_PKG_VERSION"))),
            device_description_ptrs: device_descriptions
                .iter_mut()
                .map(|description| &mut **description as *mut PJRT_DeviceDescription)
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

unsafe fn checked_mut<'a, T>(ptr: *mut T, name: &str) -> Result<&'a mut T, *mut PJRT_Error> {
    // SAFETY: caller guarantees `ptr` originates from the C ABI.
    unsafe { ptr.as_mut() }.ok_or_else(|| invalid_argument(format!("{name} must not be null")))
}

unsafe fn checked_ref<'a, T>(ptr: *const T, name: &str) -> Result<&'a T, *mut PJRT_Error> {
    // SAFETY: caller guarantees `ptr` originates from the C ABI.
    unsafe { ptr.as_ref() }.ok_or_else(|| invalid_argument(format!("{name} must not be null")))
}

fn discover_devices() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(entries) = fs::read_dir(Path::new("/dev/tenstorrent")) {
        for entry in entries.flatten() {
            paths.push(entry.path());
        }
    } else if let Ok(entries) = fs::read_dir(Path::new("/dev")) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with("tenstorrent") {
                paths.push(path);
            }
        }
    }

    paths.sort();
    paths.dedup();

    paths
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
pub unsafe extern "C" fn TT_Error_GetCode(
    args: *mut PJRT_Error_GetCode_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    args.code = unsafe { args.error.as_ref() }
        .map(|error| error.code)
        .unwrap_or(PJRT_Error_Code::PJRT_Error_Code_OK);
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
pub unsafe extern "C" fn TT_Client_Destroy(
    args: *mut PJRT_Client_Destroy_Args,
) -> *mut PJRT_Error {
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
pub unsafe extern "C" fn TT_Client_Devices(
    args: *mut PJRT_Client_Devices_Args,
) -> *mut PJRT_Error {
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
    let Ok(_args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    unimplemented("PJRT_Client_AddressableMemories is not implemented")
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
    let Ok(_args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    unimplemented("PJRT_Device_AddressableMemories is not implemented")
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
    args.default_memory = device.default_memory;
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
    let Ok(_args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    unimplemented("PJRT_Memory_AddressableByDevices is not implemented")
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

static PJRT_API: PJRT_Api = PJRT_Api {
    struct_size: size_of::<PJRT_Api>(),
    extension_start: ptr::null_mut(),
    pjrt_api_version: PJRT_Api_Version {
        struct_size: size_of::<PJRT_Api_Version>(),
        extension_start: ptr::null_mut(),
        major_version: PJRT_API_MAJOR,
        minor_version: PJRT_API_MINOR,
    },
    PJRT_Error_Destroy: Some(TT_Error_Destroy),
    PJRT_Error_Message: Some(TT_Error_Message),
    PJRT_Error_GetCode: Some(TT_Error_GetCode),
    PJRT_Plugin_Initialize: Some(TT_Plugin_Initialize),
    PJRT_Plugin_Attributes: Some(TT_Plugin_Attributes),
    unused_events: [None; 5],
    PJRT_Client_Create: Some(TT_Client_Create),
    PJRT_Client_Destroy: Some(TT_Client_Destroy),
    PJRT_Client_PlatformName: Some(TT_Client_PlatformName),
    PJRT_Client_ProcessIndex: Some(TT_Client_ProcessIndex),
    PJRT_Client_PlatformVersion: Some(TT_Client_PlatformVersion),
    PJRT_Client_Devices: Some(TT_Client_Devices),
    PJRT_Client_AddressableDevices: Some(TT_Client_AddressableDevices),
    PJRT_Client_LookupDevice: Some(TT_Client_LookupDevice),
    PJRT_Client_LookupAddressableDevice: Some(TT_Client_LookupAddressableDevice),
    PJRT_Client_AddressableMemories: Some(TT_Client_AddressableMemories),
    unused_client_rest: [None; 3],
    PJRT_DeviceDescription_Id: Some(TT_DeviceDescription_Id),
    PJRT_DeviceDescription_ProcessIndex: Some(TT_DeviceDescription_ProcessIndex),
    PJRT_DeviceDescription_Attributes: Some(TT_DeviceDescription_Attributes),
    PJRT_DeviceDescription_Kind: Some(TT_DeviceDescription_Kind),
    PJRT_DeviceDescription_DebugString: Some(TT_DeviceDescription_DebugString),
    PJRT_DeviceDescription_ToString: Some(TT_DeviceDescription_ToString),
    PJRT_Device_GetDescription: Some(TT_Device_GetDescription),
    PJRT_Device_IsAddressable: Some(TT_Device_IsAddressable),
    PJRT_Device_LocalHardwareId: Some(TT_Device_LocalHardwareId),
    PJRT_Device_AddressableMemories: Some(TT_Device_AddressableMemories),
    PJRT_Device_DefaultMemory: Some(TT_Device_DefaultMemory),
    unused_device_memory_stats: [None; 1],
    PJRT_Memory_Id: Some(TT_Memory_Id),
    PJRT_Memory_Kind: Some(TT_Memory_Kind),
    PJRT_Memory_DebugString: Some(TT_Memory_DebugString),
    PJRT_Memory_ToString: Some(TT_Memory_ToString),
    PJRT_Memory_AddressableByDevices: Some(TT_Memory_AddressableByDevices),
    unused_before_topology: [None; 42],
    unused_topology_create_destroy: [None; 2],
    PJRT_TopologyDescription_PlatformName: Some(TT_TopologyDescription_PlatformName),
    PJRT_TopologyDescription_PlatformVersion: Some(TT_TopologyDescription_PlatformVersion),
    PJRT_TopologyDescription_GetDeviceDescriptions: Some(
        TT_TopologyDescription_GetDeviceDescriptions,
    ),
    unused_topology_serialize: [None; 1],
    PJRT_TopologyDescription_Attributes: Some(TT_TopologyDescription_Attributes),
    unused_before_client_topology: [None; 6],
    PJRT_Client_TopologyDescription: Some(TT_Client_TopologyDescription),
    unused_tail: [None; PJRT_API_UNUSED_TAIL_SLOTS],
};

#[unsafe(no_mangle)]
pub extern "C" fn GetPjrtApi() -> *const PJRT_Api {
    &PJRT_API
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_ok(api: &PJRT_Api, error: *mut PJRT_Error) {
        if error.is_null() {
            return;
        }

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
        panic!("unexpected PJRT error {:?}: {detail}", code_args.code);
    }

    #[test]
    fn get_pjrt_api_exposes_minimal_client_and_device_interface() {
        let api = unsafe { &*GetPjrtApi() };
        assert_eq!(api.pjrt_api_version.major_version, PJRT_API_MAJOR);
        assert_eq!(api.pjrt_api_version.minor_version, PJRT_API_MINOR);

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
            let devices = unsafe { std::slice::from_raw_parts(devices_args.devices, devices_args.num_devices) };
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
            check_ok(api, unsafe { device_get_description(&mut get_description_args) });
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
                std::slice::from_raw_parts(kind_args.device_kind.cast::<u8>(), kind_args.device_kind_size)
            };
            assert_eq!(kind, b"Tenstorrent");
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
        check_ok(api, unsafe { api.PJRT_Error_GetCode.expect("error get code must exist")(&mut code_args) });
        assert_eq!(code_args.code, PJRT_Error_Code::PJRT_Error_Code_INVALID_ARGUMENT);

        unsafe {
            api.PJRT_Error_Destroy
                .expect("error destroy must exist")(&mut PJRT_Error_Destroy_Args {
                    struct_size: size_of::<PJRT_Error_Destroy_Args>(),
                    extension_start: ptr::null_mut(),
                    error,
                });
        }
    }
}
