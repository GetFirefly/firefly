add_subdirectory(Diagnostics)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Target)
add_subdirectory(Translation)

lumen_tablegen_library(
  NAME
    RustEncodingGen
  TD_FILE
    "Dialect/EIR/IR/EIRBase.td"
  OUTS
    -gen-rust-eir-encoding-defs term_encoding.rs
  TBLGEN
    LUMEN
)

add_custom_target(CopyRustEncodingGenOutput
  ALL
  cp term_encoding.rs "$ENV{OUT_DIR}/"
)
add_dependencies(CopyRustEncodingGenOutput
  lumen::compiler::Dialect::EIR::Tools
  lumen::compiler::RustEncodingGen)

# Get the LLVM libraries we need to link against
set(LUMEN_REQUIRED_LLVM_COMPONENTS_LIST
  all-targets
  core
  ipo
  bitreader
  bitwriter
  linker
  asmparser
  lto
  instrumentation
)

# Map LLVM components to library names
#
# We are using this because llvm_map_components_to_libnames is
# not correctly producing all of the dependencies for each component
#
# TODO: The following conditional ideally should work, but if the
# LLVM build did not build with LLVM_BUILD_LLVM_DYLIB, then llvm-config
# will complain about not finding it when called with --link-shared. We
# could fallback to using --link-static, but I don't have time for that
# just now.
#if (BUILD_SHARED_LIBS)
  set(LLVM_CONFIG_LIBNAMES_ARGV "--libnames --link-static")
#else()
  #set(LLVM_CONFIG_LIBNAMES_ARGV "--libnames --link-shared")
#endif()
foreach(component IN LISTS LUMEN_REQUIRED_LLVM_COMPONENTS_LIST)
  set(LLVM_CONFIG_LIBNAMES_ARGV "${LLVM_CONFIG_LIBNAMES_ARGV} ${component}")
endforeach()
message(STATUS "LLVM_CONFIG_LIBNAMES_ARGV '${LLVM_CONFIG_LIBNAMES_ARGV}'")
separate_arguments(LLVM_CONFIG_LIBNAMES_ARGS NATIVE_COMMAND "${LLVM_CONFIG_LIBNAMES_ARGV}")

execute_process(
  COMMAND ${LLVM_CONFIG_PATH} ${LLVM_CONFIG_LIBNAMES_ARGS}
  RESULT_VARIABLE LLVM_LIBNAMES_status
  OUTPUT_VARIABLE LLVM_LIBNAMES_output
  ERROR_VARIABLE LLVM_LIBNAMES_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (LLVM_LIBNAMES_status EQUAL 0)
  separate_arguments(LLVM_LIBNAMES NATIVE_COMMAND ${LLVM_LIBNAMES_output})
else()
  message(FATAL_ERROR "Unable to detect LLVM library names!\n${LLVM_LIBNAMES_error}")
endif()

# Strip the .a suffix from each library found
set(LIB_NAMES "")
foreach(lib IN LISTS LLVM_LIBNAMES)
  string(REGEX REPLACE "^lib([A-Za-z0-9_-]+)[.]a$" "\\1" lib_stripped "${lib}")
  list(APPEND LIB_NAMES "${lib_stripped}")
endforeach()

file(WRITE ${CMAKE_BINARY_DIR}/llvm_deps.txt "")
foreach(lib IN LISTS LIB_NAMES)
  file(APPEND ${CMAKE_BINARY_DIR}/llvm_deps.txt
       "${lib}\n")
endforeach()

set(_CARGO_LINK_LIB_PATH "")
if(CARGO_TARGET_DIR)
  message(STATUS "CARGO_TARGET_DIR: ${CARGO_TARGET_DIR}")
  set(_CARGO_LINK_LIB_PATH "-L${CARGO_TARGET_DIR}")
endif()

if (BUILD_SHARED_LIBS)
  lumen_cc_library(
    NAME
      LumenCodegen
    SRCS
      "codegen.cpp"
    PUBLIC
    SHARED
  )
  set(_LumenCodegen_LINK_LIBS_LIST
    lumen_compiler_Diagnostics_Diagnostics
    lumen_compiler_Dialect_EIR_IR_IR
    lumen_compiler_Dialect_EIR_Conversion_EIRToLLVM_EIRToLLVM
    lumen_compiler_Dialect_EIR_Transforms_Transforms
    lumen_compiler_Support_Support
    lumen_compiler_Target_Target
    lumen_compiler_Translation_Translation
    LLVMTarget
  )
  set(_LumenCodegen_LINK_LIBS "")
  if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    foreach(_LumenCodegen_LINK_LIB ${_LumenCodegen_LINK_LIBS_LIST})
      list(APPEND _LumenCodegen_LINK_LIBS "-Wl,-force_load")
      list(APPEND _LumenCodegen_LINK_LIBS "${_LumenCodegen_LINK_LIB}")
    endforeach(_LumenCodegen_LINK_LIB)
  else()
    set(_LumenCodegen_LINK_LIBS "-Wl,--whole-archive" ${_LumenCodegen_LINK_LIBS_LIST} "-Wl,--no-whole-archive")
  endif()

  list(REMOVE_ITEM LIB_NAMES "LLVMTarget")

  target_link_libraries(lumen_compiler_LumenCodegen
    PUBLIC
      ${_LumenCodegen_LINK_LIBS}
      ${LIB_NAMES}
    PRIVATE
      ${_CARGO_LINK_LIB_PATH}/deps
      "-lliblumen_llvm-fdcc608a8bf21908"
      "-lliblumen_term"
    )
else()
  message(FATAL_ERROR "building static!")
endif()
