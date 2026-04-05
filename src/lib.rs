#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

pub mod device;
pub mod dram;
mod linux;

use device::Device;
use dram::{DType, DramBuffer};
use std::ffi::{CString, c_char, c_void};
use std::io;
use std::mem::size_of;
use std::ptr;
use std::slice;
use std::sync::OnceLock;

const PJRT_API_MAJOR: i32 = 0;
const PJRT_API_MINOR: i32 = 96;
const PJRT_API_UNUSED_TAIL_SLOTS: usize = 35;
const PJRT_Buffer_Type_INVALID: i32 = 0;
const PJRT_Buffer_Type_S8: i32 = 2;
const PJRT_Buffer_Type_S32: i32 = 4;
const PJRT_Buffer_Type_U8: i32 = 6;
const PJRT_Buffer_Type_U16: i32 = 7;
const PJRT_Buffer_Type_U32: i32 = 8;
const PJRT_Buffer_Type_F16: i32 = 10;
const PJRT_Buffer_Type_F32: i32 = 11;
const PJRT_Buffer_Type_BF16: i32 = 13;
const PJRT_HostBufferSemantics_kImmutableOnlyDuringCall: i32 = 0;
const PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes: i32 = 1;
const PJRT_HostBufferSemantics_kImmutableZeroCopy: i32 = 2;
const PJRT_HostBufferSemantics_kMutableZeroCopy: i32 = 3;

type PjrtOpaqueFn = Option<unsafe extern "C" fn()>;
type PjrtResultFn<Args> = Option<unsafe extern "C" fn(args: *mut Args) -> *mut PJRT_Error>;
type PjrtVoidFn<Args> = Option<unsafe extern "C" fn(args: *mut Args)>;
type PjrtEventOnReadyCallback =
    Option<unsafe extern "C" fn(error: *mut PJRT_Error, user_arg: *mut c_void)>;

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
}

#[repr(C)]
pub struct PJRT_Event {
    error: Option<(PJRT_Error_Code, String)>,
}

#[repr(C)]
pub struct PJRT_Buffer_MemoryLayout {
    _private: [u8; 0],
}

#[repr(C)]
pub struct PJRT_Buffer {
    buffer_type: i32,
    dims: Vec<i64>,
    device: *mut PJRT_Device,
    memory: *mut PJRT_Memory,
    local_hardware_id: usize,
    dram_buffer: Option<DramBuffer>,
    deleted: bool,
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
    memory_ptrs: Vec<*mut PJRT_Memory>,
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
pub struct PJRT_Event_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub event: *mut PJRT_Event,
}

#[repr(C)]
pub struct PJRT_Event_IsReady_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub event: *mut PJRT_Event,
    pub is_ready: bool,
}

#[repr(C)]
pub struct PJRT_Event_Error_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub event: *mut PJRT_Event,
}

#[repr(C)]
pub struct PJRT_Event_Await_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub event: *mut PJRT_Event,
}

#[repr(C)]
pub struct PJRT_Event_OnReady_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub event: *mut PJRT_Event,
    pub callback: PjrtEventOnReadyCallback,
    pub user_arg: *mut c_void,
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
pub struct PJRT_Client_Compile_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub program: *const c_void,
    pub compile_options: *const c_char,
    pub compile_options_size: usize,
    pub executable: *mut c_void,
}

#[repr(C)]
pub struct PJRT_Client_DefaultDeviceAssignment_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub num_replicas: i32,
    pub num_partitions: i32,
    pub default_assignment_size: usize,
    pub default_assignment: *mut i32,
}

#[repr(C)]
pub struct PJRT_Client_BufferFromHostBuffer_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub client: *mut PJRT_Client,
    pub data: *const c_void,
    pub type_: i32,
    pub dims: *const i64,
    pub num_dims: usize,
    pub byte_strides: *const i64,
    pub num_byte_strides: usize,
    pub host_buffer_semantics: i32,
    pub device: *mut PJRT_Device,
    pub memory: *mut PJRT_Memory,
    pub device_layout: *mut PJRT_Buffer_MemoryLayout,
    pub done_with_host_buffer: *mut PJRT_Event,
    pub buffer: *mut PJRT_Buffer,
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
pub struct PJRT_Buffer_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
}

#[repr(C)]
pub struct PJRT_Buffer_ElementType_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub type_: i32,
}

#[repr(C)]
pub struct PJRT_Buffer_Dimensions_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub dims: *const i64,
    pub num_dims: usize,
}

#[repr(C)]
pub struct PJRT_Buffer_UnpaddedDimensions_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub dims: *const i64,
    pub num_dims: usize,
}

#[repr(C)]
pub struct PJRT_Buffer_DynamicDimensionIndices_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub dynamic_dimension_indices: *const bool,
    pub num_dynamic_dimension_indices: usize,
}

#[repr(C)]
pub struct PJRT_Buffer_GetMemoryLayout_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub layout: *mut PJRT_Buffer_MemoryLayout,
}

#[repr(C)]
pub struct PJRT_Buffer_OnDeviceSizeInBytes_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub on_device_size_in_bytes: usize,
}

#[repr(C)]
pub struct PJRT_Buffer_Device_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub device: *mut PJRT_Device,
}

#[repr(C)]
pub struct PJRT_Buffer_Memory_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub memory: *mut PJRT_Memory,
}

#[repr(C)]
pub struct PJRT_Buffer_Delete_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
}

#[repr(C)]
pub struct PJRT_Buffer_IsDeleted_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub is_deleted: bool,
}

#[repr(C)]
pub struct PJRT_Buffer_CopyToDevice_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub dst_device: *mut PJRT_Device,
    pub dst_buffer: *mut PJRT_Buffer,
}

#[repr(C)]
pub struct PJRT_Buffer_ToHostBuffer_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub src: *mut PJRT_Buffer,
    pub host_layout: *mut PJRT_Buffer_MemoryLayout,
    pub dst: *mut c_void,
    pub dst_size: usize,
    pub event: *mut PJRT_Event,
}

#[repr(C)]
pub struct PJRT_Buffer_IsOnCpu_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub is_on_cpu: bool,
}

#[repr(C)]
pub struct PJRT_Buffer_ReadyEvent_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub event: *mut PJRT_Event,
}

#[repr(C)]
pub struct PJRT_Buffer_UnsafePointer_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub buffer_pointer: usize,
}

#[repr(C)]
pub struct PJRT_Buffer_IncreaseExternalReferenceCount_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
}

#[repr(C)]
pub struct PJRT_Buffer_DecreaseExternalReferenceCount_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
}

#[repr(C)]
pub struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
    pub struct_size: usize,
    pub extension_start: *mut PJRT_Extension_Base,
    pub buffer: *mut PJRT_Buffer,
    pub device_memory_ptr: *mut c_void,
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
    pub PJRT_Event_Destroy: PjrtResultFn<PJRT_Event_Destroy_Args>,
    pub PJRT_Event_IsReady: PjrtResultFn<PJRT_Event_IsReady_Args>,
    pub PJRT_Event_Error: PjrtResultFn<PJRT_Event_Error_Args>,
    pub PJRT_Event_Await: PjrtResultFn<PJRT_Event_Await_Args>,
    pub PJRT_Event_OnReady: PjrtResultFn<PJRT_Event_OnReady_Args>,
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
    pub PJRT_Client_Compile: PjrtResultFn<PJRT_Client_Compile_Args>,
    pub PJRT_Client_DefaultDeviceAssignment:
        PjrtResultFn<PJRT_Client_DefaultDeviceAssignment_Args>,
    pub PJRT_Client_BufferFromHostBuffer: PjrtResultFn<PJRT_Client_BufferFromHostBuffer_Args>,
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
    unused_executable: [PjrtOpaqueFn; 10],
    unused_loaded_executable: [PjrtOpaqueFn; 8],
    pub PJRT_Buffer_Destroy: PjrtResultFn<PJRT_Buffer_Destroy_Args>,
    pub PJRT_Buffer_ElementType: PjrtResultFn<PJRT_Buffer_ElementType_Args>,
    pub PJRT_Buffer_Dimensions: PjrtResultFn<PJRT_Buffer_Dimensions_Args>,
    pub PJRT_Buffer_UnpaddedDimensions: PjrtResultFn<PJRT_Buffer_UnpaddedDimensions_Args>,
    pub PJRT_Buffer_DynamicDimensionIndices:
        PjrtResultFn<PJRT_Buffer_DynamicDimensionIndices_Args>,
    pub PJRT_Buffer_GetMemoryLayout: PjrtResultFn<PJRT_Buffer_GetMemoryLayout_Args>,
    pub PJRT_Buffer_OnDeviceSizeInBytes: PjrtResultFn<PJRT_Buffer_OnDeviceSizeInBytes_Args>,
    pub PJRT_Buffer_Device: PjrtResultFn<PJRT_Buffer_Device_Args>,
    pub PJRT_Buffer_Memory: PjrtResultFn<PJRT_Buffer_Memory_Args>,
    pub PJRT_Buffer_Delete: PjrtResultFn<PJRT_Buffer_Delete_Args>,
    pub PJRT_Buffer_IsDeleted: PjrtResultFn<PJRT_Buffer_IsDeleted_Args>,
    pub PJRT_Buffer_CopyToDevice: PjrtResultFn<PJRT_Buffer_CopyToDevice_Args>,
    pub PJRT_Buffer_ToHostBuffer: PjrtResultFn<PJRT_Buffer_ToHostBuffer_Args>,
    pub PJRT_Buffer_IsOnCpu: PjrtResultFn<PJRT_Buffer_IsOnCpu_Args>,
    pub PJRT_Buffer_ReadyEvent: PjrtResultFn<PJRT_Buffer_ReadyEvent_Args>,
    pub PJRT_Buffer_UnsafePointer: PjrtResultFn<PJRT_Buffer_UnsafePointer_Args>,
    pub PJRT_Buffer_IncreaseExternalReferenceCount:
        PjrtResultFn<PJRT_Buffer_IncreaseExternalReferenceCount_Args>,
    pub PJRT_Buffer_DecreaseExternalReferenceCount:
        PjrtResultFn<PJRT_Buffer_DecreaseExternalReferenceCount_Args>,
    pub PJRT_Buffer_OpaqueDeviceMemoryDataPointer:
        PjrtResultFn<PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args>,
    unused_copy_to_device_stream: [PjrtOpaqueFn; 5],
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
        Self::new_with_devices(Device::discover())
    }

    fn new_with_devices(discovered: Vec<Device>) -> Self {
        let mut device_descriptions = Vec::with_capacity(discovered.len());

        for info in &discovered {
            device_descriptions.push(Box::new(PJRT_DeviceDescription {
                id: info.id as i32,
                process_index: 0,
                device_kind: cstring_lossy(info.device_kind()),
                debug_string: cstring_lossy(info.device_debug_string()),
                to_string: cstring_lossy(info.device_to_string()),
            }));
        }

        let mut memories = Vec::with_capacity(discovered.len());
        for info in &discovered {
            memories.push(Box::new(PJRT_Memory {
                id: info.id as i32,
                kind: cstring_lossy("dram"),
                debug_string: cstring_lossy(info.memory_debug_string()),
                to_string: cstring_lossy(info.memory_to_string()),
                device_ptrs: Vec::with_capacity(1),
            }));
        }

        let mut memory_ptrs = Vec::with_capacity(memories.len());
        for memory in &mut memories {
            memory_ptrs.push(&mut **memory as *mut PJRT_Memory);
        }

        let mut devices = Vec::with_capacity(discovered.len());
        for info in &discovered {
            let index = info.id;
            let description = &mut *device_descriptions[index] as *mut PJRT_DeviceDescription;
            let default_memory = memory_ptrs[index];
            devices.push(Box::new(PJRT_Device {
                id: info.id as i32,
                local_hardware_id: info.local_hardware_id as i32,
                description,
                addressable: true,
                default_memory,
                memory_ptrs: vec![default_memory],
            }));
        }

        let mut device_ptrs = Vec::with_capacity(devices.len());
        for device in &mut devices {
            device_ptrs.push(&mut **device as *mut PJRT_Device);
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

fn lib_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LIBTT_LOG")
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    })
}

fn lib_log(message: impl std::fmt::Display) {
    if lib_log_enabled() {
        eprintln!("[libtt] {message}");
    }
}

fn io_error(err: io::Error) -> *mut PJRT_Error {
    let code = match err.kind() {
        io::ErrorKind::InvalidInput => PJRT_Error_Code::PJRT_Error_Code_INVALID_ARGUMENT,
        io::ErrorKind::OutOfMemory => PJRT_Error_Code::PJRT_Error_Code_RESOURCE_EXHAUSTED,
        _ => PJRT_Error_Code::PJRT_Error_Code_INTERNAL,
    };
    pjrt_error(err.to_string(), code)
}

fn ready_event() -> *mut PJRT_Event {
    Box::into_raw(Box::new(PJRT_Event { error: None }))
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

fn pjrt_buffer_type_to_dtype(buffer_type: i32) -> Result<DType, *mut PJRT_Error> {
    match buffer_type {
        PJRT_Buffer_Type_S8 => Ok(DType::Int8),
        PJRT_Buffer_Type_S32 => Ok(DType::Int32),
        PJRT_Buffer_Type_U8 => Ok(DType::UInt8),
        PJRT_Buffer_Type_U16 => Ok(DType::UInt16),
        PJRT_Buffer_Type_U32 => Ok(DType::UInt32),
        PJRT_Buffer_Type_F16 => Ok(DType::Float16),
        PJRT_Buffer_Type_F32 => Ok(DType::Float32),
        PJRT_Buffer_Type_BF16 => Ok(DType::Float16B),
        PJRT_Buffer_Type_INVALID => Err(invalid_argument("invalid PJRT buffer type")),
        _ => Err(unimplemented(format!(
            "unsupported PJRT buffer type {buffer_type}"
        ))),
    }
}

fn dtype_to_pjrt_buffer_type(dtype: DType) -> i32 {
    match dtype {
        DType::Int8 => PJRT_Buffer_Type_S8,
        DType::Int32 => PJRT_Buffer_Type_S32,
        DType::UInt8 => PJRT_Buffer_Type_U8,
        DType::UInt16 => PJRT_Buffer_Type_U16,
        DType::UInt32 => PJRT_Buffer_Type_U32,
        DType::Float16 => PJRT_Buffer_Type_F16,
        DType::Float32 => PJRT_Buffer_Type_F32,
        DType::Float16B => PJRT_Buffer_Type_BF16,
    }
}

fn dims_i64_to_usize(dims: &[i64]) -> Result<Vec<usize>, *mut PJRT_Error> {
    dims.iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| invalid_argument("shape dimensions must be >= 0"))
        })
        .collect()
}

fn checked_dims(ptr: *const i64, len: usize) -> Result<&'static [i64], *mut PJRT_Error> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(invalid_argument("dims must not be null when num_dims > 0"));
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
    let strides = checked_dims(byte_strides, num_byte_strides)?;
    let mut expected = dtype.bytes_per_element();
    for (&dim, &stride) in dims.iter().rev().zip(strides.iter().rev()) {
        let stride = usize::try_from(stride)
            .map_err(|_| invalid_argument("byte strides must be >= 0"))?;
        if stride != expected {
            return Err(unimplemented("only dense row-major host buffers are supported"));
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
pub unsafe extern "C" fn TT_Event_IsReady(
    args: *mut PJRT_Event_IsReady_Args,
) -> *mut PJRT_Error {
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
pub unsafe extern "C" fn TT_Event_OnReady(
    args: *mut PJRT_Event_OnReady_Args,
) -> *mut PJRT_Error {
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
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    args.addressable_memories = if client.memory_ptrs.is_empty() {
        ptr::null()
    } else {
        client.memory_ptrs.as_ptr().cast::<*mut c_void>()
    };
    args.num_addressable_memories = client.memory_ptrs.len();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Client_Compile(
    args: *mut PJRT_Client_Compile_Args,
) -> *mut PJRT_Error {
    if args.is_null() {
        return invalid_argument("args must not be null");
    }
    unimplemented("PJRT_Client_Compile is not implemented")
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
        })
        .ok_or_else(|| invalid_argument("default device assignment is too large"));
    let Ok(required) = required else {
        return invalid_argument("default device assignment is too large");
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
pub unsafe extern "C" fn TT_Client_BufferFromHostBuffer(
    args: *mut PJRT_Client_BufferFromHostBuffer_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    lib_log("pjrt buffer_from_host_buffer entered");
    let Ok(client) = (unsafe { checked_ref(args.client, "client") }) else {
        return invalid_argument("client must not be null");
    };
    if !args.device_layout.is_null() {
        return unimplemented("custom device layouts are not supported");
    }
    match args.host_buffer_semantics {
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall
        | PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes
        | PJRT_HostBufferSemantics_kImmutableZeroCopy
        | PJRT_HostBufferSemantics_kMutableZeroCopy => {}
        _ => return invalid_argument("unknown host buffer semantics"),
    }

    let dtype = match pjrt_buffer_type_to_dtype(args.type_) {
        Ok(dtype) => dtype,
        Err(err) => return err,
    };
    let dims_i64 = match checked_dims(args.dims, args.num_dims) {
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
            Ok(memory) => memory.device_ptrs.first().copied().unwrap_or(ptr::null_mut()),
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
    let target_memory = if !args.memory.is_null() {
        args.memory
    } else {
        match unsafe { checked_ref(target_device, "device") } {
            Ok(device) => device.default_memory,
            Err(err) => return err,
        }
    };
    let local_hardware_id = match unsafe { checked_ref(target_device, "device") } {
        Ok(device) => device.local_hardware_id as usize,
        Err(err) => return err,
    };
    lib_log(format!(
        "pjrt buffer_from_host_buffer type={} dims={:?} local_hardware_id={}",
        args.type_, dims_i64, local_hardware_id
    ));

    let data = if byte_size == 0 {
        &[]
    } else {
        // SAFETY: caller owns `data` for `byte_size` bytes during the call.
        unsafe { slice::from_raw_parts(args.data.cast::<u8>(), byte_size) }
    };
    let mut device = match Device::open(local_hardware_id) {
        Ok(device) => device,
        Err(err) => return io_error(err),
    };
    lib_log("pjrt buffer_from_host_buffer device opened");
    let dram_buffer = match device.alloc_write(data, dtype, &shape, "pjrt") {
        Ok(buffer) => buffer,
        Err(err) => return io_error(err),
    };
    lib_log(format!(
        "pjrt buffer_from_host_buffer allocated addr=0x{:x} tiles={}",
        dram_buffer.addr, dram_buffer.num_tiles
    ));

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
pub unsafe extern "C" fn TT_Buffer_Destroy(
    args: *mut PJRT_Buffer_Destroy_Args,
) -> *mut PJRT_Error {
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
    args.dims = if buffer.dims.is_empty() {
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
    args.dynamic_dimension_indices = ptr::null();
    args.num_dynamic_dimension_indices = 0;
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_GetMemoryLayout(
    args: *mut PJRT_Buffer_GetMemoryLayout_Args,
) -> *mut PJRT_Error {
    if args.is_null() {
        return invalid_argument("args must not be null");
    }
    unimplemented("PJRT_Buffer_GetMemoryLayout is not implemented")
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
        return invalid_argument("buffer has been deleted");
    };
    args.on_device_size_in_bytes = dram_buffer.size();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_Device(
    args: *mut PJRT_Buffer_Device_Args,
) -> *mut PJRT_Error {
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
pub unsafe extern "C" fn TT_Buffer_Memory(
    args: *mut PJRT_Buffer_Memory_Args,
) -> *mut PJRT_Error {
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
pub unsafe extern "C" fn TT_Buffer_Delete(
    args: *mut PJRT_Buffer_Delete_Args,
) -> *mut PJRT_Error {
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
pub unsafe extern "C" fn TT_Buffer_CopyToDevice(
    args: *mut PJRT_Buffer_CopyToDevice_Args,
) -> *mut PJRT_Error {
    if args.is_null() {
        return invalid_argument("args must not be null");
    }
    unimplemented("PJRT_Buffer_CopyToDevice is not implemented")
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_ToHostBuffer(
    args: *mut PJRT_Buffer_ToHostBuffer_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    lib_log("pjrt buffer_to_host_buffer entered");
    let Ok(buffer) = (unsafe { checked_ref(args.src, "src") }) else {
        return invalid_argument("src must not be null");
    };
    if !args.host_layout.is_null() {
        return unimplemented("custom host layouts are not supported");
    }
    let Some(dram_buffer) = buffer.dram_buffer.as_ref() else {
        return invalid_argument("buffer has been deleted");
    };
    let mut device = match Device::open(buffer.local_hardware_id) {
        Ok(device) => device,
        Err(err) => return io_error(err),
    };
    let data = match device.dram_read(dram_buffer) {
        Ok(data) => data,
        Err(err) => return io_error(err),
    };
    if args.dst.is_null() {
        args.dst_size = data.len();
        args.event = ready_event();
        return ptr::null_mut();
    }
    if args.dst_size < data.len() {
        return invalid_argument("dst buffer is too small");
    }
    // SAFETY: caller owns `dst` for at least `data.len()` bytes.
    unsafe {
        ptr::copy_nonoverlapping(data.as_ptr(), args.dst.cast::<u8>(), data.len());
    }
    args.dst_size = data.len();
    args.event = ready_event();
    ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn TT_Buffer_IsOnCpu(
    args: *mut PJRT_Buffer_IsOnCpu_Args,
) -> *mut PJRT_Error {
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
pub unsafe extern "C" fn TT_Buffer_UnsafePointer(
    args: *mut PJRT_Buffer_UnsafePointer_Args,
) -> *mut PJRT_Error {
    if args.is_null() {
        return invalid_argument("args must not be null");
    }
    unimplemented("PJRT_Buffer_UnsafePointer is not implemented")
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
pub unsafe extern "C" fn TT_Buffer_OpaqueDeviceMemoryDataPointer(
    args: *mut PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
) -> *mut PJRT_Error {
    let Ok(args) = (unsafe { checked_mut(args, "args") }) else {
        return invalid_argument("args must not be null");
    };
    if args.buffer.is_null() {
        return invalid_argument("buffer must not be null");
    }
    args.device_memory_ptr = ptr::null_mut();
    unimplemented("PJRT_Buffer_OpaqueDeviceMemoryDataPointer is not implemented")
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
    PJRT_Event_Destroy: Some(TT_Event_Destroy),
    PJRT_Event_IsReady: Some(TT_Event_IsReady),
    PJRT_Event_Error: Some(TT_Event_Error),
    PJRT_Event_Await: Some(TT_Event_Await),
    PJRT_Event_OnReady: Some(TT_Event_OnReady),
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
    PJRT_Client_Compile: Some(TT_Client_Compile),
    PJRT_Client_DefaultDeviceAssignment: Some(TT_Client_DefaultDeviceAssignment),
    PJRT_Client_BufferFromHostBuffer: Some(TT_Client_BufferFromHostBuffer),
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
    unused_executable: [None; 10],
    unused_loaded_executable: [None; 8],
    PJRT_Buffer_Destroy: Some(TT_Buffer_Destroy),
    PJRT_Buffer_ElementType: Some(TT_Buffer_ElementType),
    PJRT_Buffer_Dimensions: Some(TT_Buffer_Dimensions),
    PJRT_Buffer_UnpaddedDimensions: Some(TT_Buffer_UnpaddedDimensions),
    PJRT_Buffer_DynamicDimensionIndices: Some(TT_Buffer_DynamicDimensionIndices),
    PJRT_Buffer_GetMemoryLayout: Some(TT_Buffer_GetMemoryLayout),
    PJRT_Buffer_OnDeviceSizeInBytes: Some(TT_Buffer_OnDeviceSizeInBytes),
    PJRT_Buffer_Device: Some(TT_Buffer_Device),
    PJRT_Buffer_Memory: Some(TT_Buffer_Memory),
    PJRT_Buffer_Delete: Some(TT_Buffer_Delete),
    PJRT_Buffer_IsDeleted: Some(TT_Buffer_IsDeleted),
    PJRT_Buffer_CopyToDevice: Some(TT_Buffer_CopyToDevice),
    PJRT_Buffer_ToHostBuffer: Some(TT_Buffer_ToHostBuffer),
    PJRT_Buffer_IsOnCpu: Some(TT_Buffer_IsOnCpu),
    PJRT_Buffer_ReadyEvent: Some(TT_Buffer_ReadyEvent),
    PJRT_Buffer_UnsafePointer: Some(TT_Buffer_UnsafePointer),
    PJRT_Buffer_IncreaseExternalReferenceCount: Some(
        TT_Buffer_IncreaseExternalReferenceCount,
    ),
    PJRT_Buffer_DecreaseExternalReferenceCount: Some(
        TT_Buffer_DecreaseExternalReferenceCount,
    ),
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer: Some(
        TT_Buffer_OpaqueDeviceMemoryDataPointer,
    ),
    unused_copy_to_device_stream: [None; 5],
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
    use crate::device::{Device, ProbeInfo};
    use std::path::PathBuf;

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

    #[test]
    fn device_abstraction_surfaces_board_metadata_through_pjrt_objects() {
        let device = Device::from_probe(
            0,
            3,
            PathBuf::from("/dev/tenstorrent/3"),
            Some(ProbeInfo {
                tensix_enabled_col_mask: 0x0fff,
                gddr_enabled_mask: 0x7f,
            }),
        );
        let client = PJRT_Client::new_with_devices(vec![device]);

        let description = &client.device_descriptions[0];
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
        assert_eq!(device.local_hardware_id, 3);
    }
}
