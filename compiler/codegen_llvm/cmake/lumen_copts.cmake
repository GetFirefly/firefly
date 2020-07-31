#-------------------------------------------------------------------------------
# C++ used within Lumen
#-------------------------------------------------------------------------------

set(LUMEN_CXX_STANDARD 17)

set(LUMEN_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND LUMEN_COMMON_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${CMAKE_CURRENT_BINARY_DIR}/lib
)

lumen_select_compiler_opts(LUMEN_DEFAULT_COPTS
  CLANG
    "-Wno-strict-prototypes"
    "-Wno-shadow-uncaptured-local"
    "-Wno-gnu-zero-variadic-macro-arguments"
    "-Wno-shadow-field-in-constructor"
    "-Wno-unreachable-code-return"
    "-Wno-unused-private-field"
    "-Wno-missing-variable-declarations"
    "-Wno-gnu-label-as-value"
    "-Wno-unused-local-typedef"
    "-Wno-gnu-zero-variadic-macro-arguments"
  CLANG_OR_GCC
    "-Wno-unused-parameter"
    "-Wno-undef"
    "-fno-rtti"
  MSVC_OR_CLANG_CL
    "/DWIN32_LEAN_AND_MEAN"
    "/EHsc"
)
set(LUMEN_DEFAULT_LINKOPTS "")
set(LUMEN_TEST_COPTS "")

if(${LUMEN_ENABLE_TRACING})
  list(APPEND LUMEN_DEFAULT_COPTS
    "-DGLOBAL_WTF_ENABLE=1"
  )
endif()

#-------------------------------------------------------------------------------
# Compiler: Clang/LLVM
#-------------------------------------------------------------------------------

# TODO: Clang/LLVM options.

#-------------------------------------------------------------------------------
# Compiler: GCC
#-------------------------------------------------------------------------------

# TODO: GCC options.

#-------------------------------------------------------------------------------
# Compiler: MSVC
#-------------------------------------------------------------------------------

# TODO: MSVC options.

#-------------------------------------------------------------------------------
# Third party: benchmark
#-------------------------------------------------------------------------------

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# Third party: gtest
#-------------------------------------------------------------------------------

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(GTEST_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/include/"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googlemock/include/"
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# Third party: llvm/mlir
#-------------------------------------------------------------------------------

set(LLVM_PREFIX $ENV{LLVM_PREFIX} CACHE PATH "The root directory of the LLVM installation to use")
if (NOT LLVM_PREFIX)
    message(FATAL_ERROR "Missing LLVM_PREFIX environment variable!")
endif()

find_package(LLVM REQUIRED CONFIG
    PATHS ${LLVM_PREFIX}
    NO_DEFAULT_PATH
)

find_package(MLIR REQUIRED CONFIG
    PATHS ${LLVM_PREFIX}
    NO_DEFAULT_PATH
)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

list(APPEND LUMEN_COMMON_INCLUDE_DIRS
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  ${CARGO_MANIFEST_DIR}
)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(MLIR_TABLEGEN_EXE mlir-tblgen)
set(LUMEN_TABLEGEN_EXE lumen-tblgen)

find_program(LLVM_CONFIG_PATH NAMES llvm-config PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
