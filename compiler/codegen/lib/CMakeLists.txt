cmake_policy(SET CMP0048 NEW)

cmake_policy(VERSION 3.10..3.15)
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

if (POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if (POLICY CMP0079)
  cmake_policy(SET CMP0079 NEW)
endif()

# Suppress warnings from LLVM that are out of our control
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)

if (ENV{VERBOSE})
    set(CMAKE_INSTALL_MESSAGE ALWAYS)
else()
    set(CMAKE_INSTALL_MESSAGE LAZY)
endif()

# Load modules from root
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(lumen
    VERSION 0.1
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set up vars for important source locations
set(LUMEN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LUMEN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(LUMEN_IDE_FOLDER Lumen)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(LUMEN_BUILD_COMPILER "Builds the Lumen compiler." ON)
option(LUMEN_BUILD_TESTS "Builds Lumen unit tests." OFF)

message(STATUS "CMake Version   : " "${CMAKE_SYSTEM_VERSION}")
message(STATUS "Compiler        : " "${CMAKE_C_COMPILER}")
message(STATUS "Target          : " "${CMAKE_C_COMPILER_TARGET}")
message(STATUS "Operating System: " "${CMAKE_SYSTEM}")
message(STATUS "Build Compiler  : " "${LUMEN_BUILD_COMPILER}")
message(STATUS "Build Tests     : " "${LUMEN_BUILD_TESTS}")


include(lumen_macros)
include(lumen_copts)
include(lumen_cc_binary)
include(lumen_cc_library)
include(lumen_cc_test)
include(lumen_tablegen_library)
include(lumen_glob_lit_tests)
include(lumen_lit_test)
include(lumen_alwayslink)

string(JOIN " " CMAKE_CXX_FLAGS ${LUMEN_DEFAULT_COPTS} ${CMAKE_CXX_FLAGS})

set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

include(external_cc_library)

if (${LUMEN_BUILD_TESTS})
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
endif()

# We don't want builds stomping all over the source directory
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. "
"Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()


if (${LUMEN_BUILD_COMPILER})
  set_alwayslink_mlir_libs()
  add_subdirectory(lumen/compiler)
  if (${LUMEN_BUILD_TESTS})
    enable_testing(lumen)
    add_subdirectory(lumen/testing)
  endif()
  add_subdirectory(tools)
endif()

#lumen_complete_dylib_link_options()
lumen_complete_binary_link_options()
