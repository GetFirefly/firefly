option(USE_LLVM_CLANG "When true, uses the LLVM-provided toolchain as compiler" OFF)

message(STATUS "USE_LLVM_CLANG: ${USE_LLVM_CLANG}")

set(LLVM_PREFIX $ENV{LLVM_PREFIX} CACHE PATH "The root directory of the LLVM installation to use")

# Whether LLVM_PREFIX was given or not, try and find the LLVM package with what we have
if (NOT LLVM_PREFIX)
    message(FATAL_ERROR "Missing LLVM_PREFIX environment variable!")
else()
    find_package(LLVM REQUIRED CONFIG
        PATHS ${LLVM_PREFIX}
        NO_DEFAULT_PATH
    )
endif()

message(STATUS "Using LLVM found in      : " "${LLVM_DIR}")
message(STATUS "LLVM Version             : " "${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM Assertions Enabled? : " "${LLVM_ENABLE_ASSERTIONS}")
message(STATUS "LLVM Exception Handling? : " "${LLVM_ENABLE_EH}")
message(STATUS "LLVM Runtime Type Info?  : " "${LLVM_ENABLE_RTTI}")

list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})

# Set up LLVM
include(AddLLVM)
include(TableGen)
include(MLIR)

message(STATUS "LLVM Source Dir          : " "${LLVM_SOURCE_DIR}")
message(STATUS "LLVM Binary Dir          : " "${LLVM_BINARY_DIR}")
message(STATUS "LLVM Libary Dir          : " "${LLVM_LIBRARY_DIR}")
message(STATUS "LLVM Includes            : " "${LLVM_MAIN_INCLUDE_DIR}")
message(STATUS "LLVM Tools Directory     : " "${LLVM_TOOLS_BINARY_DIR}")

find_program(LLC NAMES llc PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(LLVM_LINK NAMES llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(LLVM_DWARFDUMP NAMES llvm-dwarfdump PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(LLVM_CONFIG_PATH NAMES llvm-config PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

message(STATUS "LLC                      : " "${LLC}")
message(STATUS "LLVM_LINK                : " "${LLVM_LINK}")
message(STATUS "LLVM_DWARFDUMP           : " "${LLVM_DWARFDUMP}")
message(STATUS "LLVM_CONFIG              : " "${LLVM_CONFIG_PATH}")

if (USE_LLVM_CLANG)
    set(CMAKE_C_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang)
    set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
    set(CMAKE_AR ${LLVM_TOOLS_BINARY_DIR}/llvm-ar)
    set(CMAKE_RANLIB ${LLVM_TOOLS_BINARY_DIR}/llvm-ranlib)
endif()

set(CMAKE_CXX_STANDARD_MAX 11)
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -E -x c++ -c /dev/null
    RESULT_VARIABLE cxx_supports_std_status
    OUTPUT_QUIET ERROR_QUIET
)
if (cxx_supports_std_status EQUAL 0)
    set(CMAKE_CXX_STANDARD_MAX 17)
else()
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -std=c++14 -E -x c++ -c /dev/null
        RESULT_VARIABLE cxx_supports_std_status
        OUTPUT_QUIET ERROR_QUIET
    )
    if (cxx_supports_std_status EQUAL 0)
        set(CMAKE_CXX_STANDARD_MAX 14)
    else()
        set(CMAKE_CXX_STANDARD_MAX 11)
    endif()
endif()
set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD_MAX})
message(STATUS "Maximum supported C++ standard is ${CMAKE_CXX_STANDARD_MAX}")
message(STATUS "Setting default C++ standard to ${CMAKE_CXX_STANDARD}")

if (ENV{VERBOSE})
    list(APPEND CMAKE_CXX_FLAGS -v)
endif()
