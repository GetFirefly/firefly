cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Load modules from root
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(LocateLLVMToolchain)
include(CMakePushCheckState)
include(CheckSymbolExists)
include(GNUInstallDirs)

project(codegen-build
    VERSION 0.1
)

cmake_policy(VERSION 3.10..3.15)

option(ENABLE_TESTS "Enable testing" OFF)

if (ENABLE_TESTS)
    include(EnableTesting)
endif()

if (ENV{VERBOSE})
    set(CMAKE_INSTALL_MESSAGE ALWAYS)
else()
    set(CMAKE_INSTALL_MESSAGE LAZY)
endif()

message(STATUS "CMake Version   : " "${CMAKE_SYSTEM_VERSION}")
message(STATUS "Compiler        : " "${CMAKE_C_COMPILER}"    )
message(STATUS "Target          : " "${CMAKE_C_COMPILER_TARGET}")
message(STATUS "Operating System: " "${CMAKE_SYSTEM}"        )
message(STATUS "LLVM Version    : " "${LLVM_PACKAGE_VERSION}")
message(STATUS "Path to LLVM    : " "${LLVM_DIR}"            )
message(STATUS "Tests Enabled   : " "${ENABLE_TESTS}"        )

add_subdirectory(CodeGen)