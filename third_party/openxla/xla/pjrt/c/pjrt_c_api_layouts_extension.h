/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_LAYOUTS_EXTENSION_VERSION 4

typedef struct PJRT_Layouts_MemoryLayout PJRT_Layouts_MemoryLayout;
typedef struct PJRT_Layouts_SerializedLayout PJRT_Layouts_SerializedLayout;

struct PJRT_Layouts_MemoryLayout_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Layouts_MemoryLayout* layout;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_MemoryLayout_Destroy_Args, layout);

typedef PJRT_Error* PJRT_Layouts_MemoryLayout_Destroy(
    PJRT_Layouts_MemoryLayout_Destroy_Args* args);

struct PJRT_Layouts_MemoryLayout_Serialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Layouts_MemoryLayout* layout;

  const char* serialized_bytes;
  size_t serialized_bytes_size;

  PJRT_Layouts_SerializedLayout* serialized_layout;

  void (*serialized_layout_deleter)(PJRT_Layouts_SerializedLayout* s_layout);
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_MemoryLayout_Serialize_Args,
                          serialized_layout_deleter);

typedef PJRT_Error* PJRT_Layouts_MemoryLayout_Serialize(
    PJRT_Layouts_MemoryLayout_Serialize_Args* args);

struct PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Layouts_MemoryLayout* layout;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args, layout);

typedef PJRT_Error* PJRT_Layouts_PJRT_Buffer_MemoryLayout(
    PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args* args);

struct PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Buffer_Type type;
  const int64_t* dims;
  size_t num_dims;
  PJRT_Layouts_MemoryLayout* layout;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args,
                          layout);

typedef PJRT_Error* PJRT_Layouts_PJRT_Client_GetDefaultLayout(
    PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args* args);

struct PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology_description;
  PJRT_Buffer_Type type;
  const int64_t* dims;
  size_t num_dims;
  PJRT_Layouts_MemoryLayout* layout;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args,
                          layout);

typedef PJRT_Error* PJRT_Layouts_PJRT_Topology_GetDefaultLayout(
    PJRT_Layouts_PJRT_Topology_GetDefaultLayout_Args* args);

struct PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;
  PJRT_Layouts_MemoryLayout** layouts;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args,
                          layouts);

typedef PJRT_Error* PJRT_Layouts_PJRT_Executable_GetOutputLayouts(
    PJRT_Layouts_PJRT_Executable_GetOutputLayouts_Args* args);

struct PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_parameters;
  PJRT_Layouts_MemoryLayout** layouts;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args,
                          layouts);

typedef PJRT_Error* PJRT_Layouts_PJRT_Executable_GetParameterLayouts(
    PJRT_Layouts_PJRT_Executable_GetParameterLayouts_Args* args);

typedef struct PJRT_Layouts_Extension {
  PJRT_Extension_Base base;

  PJRT_Layouts_MemoryLayout_Destroy* PJRT_Layouts_MemoryLayout_Destroy;
  PJRT_Layouts_MemoryLayout_Serialize* PJRT_Layouts_MemoryLayout_Serialize;
  PJRT_Layouts_PJRT_Client_GetDefaultLayout*
      PJRT_Layouts_PJRT_Client_GetDefaultLayout;
  PJRT_Layouts_PJRT_Buffer_MemoryLayout* PJRT_Layouts_PJRT_Buffer_MemoryLayout;
  PJRT_Layouts_PJRT_Topology_GetDefaultLayout*
      PJRT_Layouts_PJRT_Topology_GetDefaultLayout;
  PJRT_Layouts_PJRT_Executable_GetOutputLayouts*
      PJRT_Layouts_PJRT_Executable_GetOutputLayouts;
  PJRT_Layouts_PJRT_Executable_GetParameterLayouts*
      PJRT_Layouts_PJRT_Executable_GetParameterLayouts;
} PJRT_Layouts_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_Extension,
                          PJRT_Layouts_PJRT_Executable_GetParameterLayouts);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_
