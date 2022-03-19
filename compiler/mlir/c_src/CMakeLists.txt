cmake_minimum_required(VERSION 3.13.4)
project(standalone-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

find_program(MLIR_PDLL_EXE "mlir-pdll" ${LLVM_TOOLS_BINARY_DIR}
  NO_DEFAULT_PATH)

# Like tablegen, but for mlir-pdll
function(pdlgen ofn)
  cmake_parse_arguments(ARG "" "" "EXTRA_INCLUDES" ${ARGN})
  if(NOT MLIR_PDLL_EXE)
    message(FATAL_ERROR "MLIR_PDLL_EXE not set")
  endif()

  if(CMAKE_GENERATOR MATCHES "Ninja")
    file(RELATIVE_PATH ofn_rel
      ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
    set(additional_cmdline
      -o ${ofn_rel}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  else()
    set(additional_cmdline
      -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn})
  endif()

  file(GLOB local_tds "*.td" "*.pdll")
  file(GLOB_RECURSE global_tds
    "*.td"
    "*.pdll"
    "${LLVM_MAIN_INCLUDE_DIR}/llvm/*.td"
    )

  list(APPEND MLIR_PDLL_FLAGS "-x=cpp")

  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  get_directory_property(mlir_pdll_includes INCLUDE_DIRECTORIES)
  list(APPEND mlir_pdll_includes ${ARG_EXTRA_INCLUDES})
  # Filter out empty items before prepending each entry with -I
  list(REMOVE_ITEM mlir_pdll_includes "")
  list(TRANSFORM mlir_pdll_includes PREPEND -I)

  set(mlir_pdll_exe ${MLIR_PDLL_EXE})
  set(mlir_pdll_depends ${MLIR_PDLL_TARGET} ${mlir_pdll_exe})

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMAND ${mlir_pdll_exe} ${ARG_UNPARSED_ARGUMENTS} -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${mlir_pdll_includes}
    ${MLIR_PDLL_FLAGS}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    ${additional_cmdline}
    DEPENDS ${ARG_DEPENDS} ${mlir_pdll_depends}
      ${local_tds} ${global_tds}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    ${LLVM_TARGET_DEPENDS}
    COMMENT "Building PDL in ${ofn}..."
    )

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})
  set(MLIR_PDLL_OUTPUT ${MLIR_PDLL_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${ofn} PROPERTIES
    GENERATED 1)
endfunction()

# Creates a target for exporting pdlgen dependencies
function(add_public_pdlgen_target target)
  if(NOT MLIR_PDLL_OUTPUT)
    message(FATAL_ERROR "Requires pdlgen() definitions as MLIR_PDLL_OUTPUT")
  endif()
  add_custom_target(${target}
    DEPENDS ${MLIR_PDLL_OUTPUT})
  if(LLVM_COMMON_DEPENDS)
    add_dependencies(${target} ${LLVM_COMMON_DEPENDS})
  endif()
  set_target_properties(${target} PROPERTIES FOLDER "Tablegenning")
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} ${target})
endfunction()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
#add_subdirectory(test)
#add_subdirectory(cir-opt)
