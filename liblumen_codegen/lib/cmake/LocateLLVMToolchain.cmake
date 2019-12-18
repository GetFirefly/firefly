option(USE_LLVM_CLANG "When true, uses the LLVM-provided toolchain as compiler" OFF)

message(STATUS "USE_LLVM_CLANG: ${USE_LLVM_CLANG}")

set(LLVM_PREFIX $ENV{LLVM_PREFIX} CACHE PATH "The root directory of the LLVM installation to use")

# Whether LLVM_PREFIX was given or not, try and find the LLVM package with what we have
if (NOT LLVM_PREFIX)
    find_package(LLVM REQUIRED CONFIG)
else()
    find_package(LLVM REQUIRED CONFIG
        PATHS ${LLVM_PREFIX}
    )
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVM in ${LLVM_DIR}")
message(STATUS "LLVM: Assertions Enabled?: " "${LLVM_ENABLE_ASSERTIONS}")
message(STATUS "LLVM: Exception Handling?: " "${LLVM_ENABLE_EH}")
message(STATUS "LLVM: Runtime Type Info? : " "${LLVM_ENABLE_RTTI}")

list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})

find_program(LLC llc ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(LLVM_LINK llvm-link ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(LLVM_DWARFDUMP llvm-dwarfdump ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
find_program(LLVM_CONFIG llvm-config ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

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
