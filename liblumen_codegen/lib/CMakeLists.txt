cmake_policy(SET CMP0048 NEW)

if (POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

cmake_policy(VERSION 3.10..3.15)
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Suppress warnings from LLVM that are out of our control
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)

# Load modules from root
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(lumen-llvm
    VERSION 0.1
)

include(LocateLLVMToolchain)
include(CMakeParseArguments)
include(CMakePushCheckState)
include(CheckSymbolExists)
include(GNUInstallDirs)

option(LUMEN_ENABLE_TESTS "Generate and build targets for the Lumen unit tests" OFF)

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

# Set up LLVM
include(AddLLVM)
include(TableGen)

message(STATUS "LLVM Source Dir : " "${LLVM_SOURCE_DIR}")
message(STATUS "LLVM Binary Dir : " "${LLVM_BINARY_DIR}")
message(STATUS "LLVM Libary Dir : " "${LLVM_LIBRARY_DIR}")
message(STATUS "LLVM Includes   : " "${LLVM_MAIN_INCLUDE_DIR}")

include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}")
link_directories("${LLVM_LIBRARY_DIR}")

# Make sure outputs go to expected locations
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

# Set up vars for important source locations
set(LUMEN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LUMEN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Configuration information about MLIR
include(MLIR)

# Make sure global includes are set
include_directories(BEFORE
  ${LUMEN_BINARY_DIR}/include
  ${LUMEN_SOURCE_DIR}/include
  ${MLIR_BINARY_DIR}/include
  ${MLIR_SOURCE_DIR}/include
)

# We don't want builds stomping all over the source directory
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. "
"Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

# Set up install targets for the headers
install(DIRECTORY include/lumen
    DESTINATION include
    COMPONENT lumen-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
)

install(DIRECTORY include/eir
    DESTINATION include
    COMPONENT eir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/lumen
    DESTINATION include
    COMPONENT lumen-headers
    FILES_MATCHING
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "*.inc"
    PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/eir
    DESTINATION include
    COMPONENT eir-headers
    FILES_MATCHING
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "*.inc"
    PATTERN "*.h"
)

add_subdirectory(include)
add_subdirectory(lib)

if (ENABLE_TESTING)
    add_executable(lumen_test tests.cpp)
    target_include_directories(lumen_test PUBLIC include/)
    target_link_libraries(lumen_test Lumen)
    gtest_discover_tests(lumen_test)
endif()
