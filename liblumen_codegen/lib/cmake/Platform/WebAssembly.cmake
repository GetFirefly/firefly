# Cmake toolchain description file for WebAssembly
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

set(WASM 1)

set(triple wasm32-unknown-unknown)

set(CMAKE_C_COMPILER_TARGET ${triple})
set(CMAKE_CXX_COMPILER_TARGET ${triple})
set(CMAKE_EXE_LINKER_FLAGS "--no-threads")

message(STATUS "Set target triple to ${triple}")

find_program(WASM_LD wasm-ld ${LLVM_BINARY_TOOLS_DIR} NO_DEFAULT_PATH)
set(CMAKE_C_LINK_EXECUTABLE ${WASM_LD} <FLAGS> <CMAKE_C_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")

set(CMAKE_SYSROOT ${SYSROOT})
set(CMAKE_STAGING_PREFIX ${SYSROOT})

message(STATUS "Set linker to  : ${WASM_LD}")
message(STATUS "Set sysroot to : ${SYSROOT}")

# Don't look in the sysroot for executables to run during the build
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Only look in the sysroot (not in the host paths) for the rest
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Some other hacks
set(CMAKE_C_COMPILER_WORKS ON)
set(CMAKE_CXX_COMPILER_WORKS ON)
