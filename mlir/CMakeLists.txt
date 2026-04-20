cmake_minimum_required(VERSION 3.20)
project(libtt_mlir_frontend LANGUAGES C CXX)

if(DEFINED ENV{LIBTT_MLIR_PREFIX} AND NOT "$ENV{LIBTT_MLIR_PREFIX}" STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTT_MLIR_PREFIX}")
endif()
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/libtt-deps")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

find_path(STABLEHLO_INCLUDE_DIR
    NAMES stablehlo/dialect/StablehloOps.h
    HINTS ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include
)
if(STABLEHLO_INCLUDE_DIR)
    message(STATUS "Found StableHLO headers: ${STABLEHLO_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "StableHLO headers not found. Run scripts/setup_deps.sh first.")
endif()

foreach(lib
    StablehloSerialization
    StablehloPasses
    StablehloOptimizationPasses
    StablehloPassUtils
    StablehloTypeConversion
    StablehloOps
    VhloOps
    ChloOps
    StablehloBase
    StablehloRegister
    VhloTypes
    StablehloTypeInference
    StablehloAssemblyFormat
    StablehloBroadcastUtils
    StablehloBroadcastLowering
    Version)
    find_library(${lib}_LIB ${lib} HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib)
    if(${lib}_LIB)
        list(APPEND STABLEHLO_LIBS ${${lib}_LIB})
    endif()
endforeach()

add_library(tt_mlir_frontend SHARED tt_mlir_frontend.cc)
set_target_properties(tt_mlir_frontend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

target_include_directories(tt_mlir_frontend PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${STABLEHLO_INCLUDE_DIR}
)

target_link_libraries(tt_mlir_frontend PRIVATE
    ${STABLEHLO_LIBS}
    MLIRReconcileUnrealizedCasts
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRArithDialect
    MLIRComplexDialect
    MLIRQuantDialect
    MLIRShapeDialect
    MLIRTensorDialect
    MLIRDialect
    MLIRViewLikeInterface
    MLIRControlFlowInterfaces
    MLIRDestinationStyleOpInterface
    MLIRInferIntRangeInterface
    MLIRDataLayoutInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRTransformUtils
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRAsmParser
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRSupport
    LLVMSupport
)
