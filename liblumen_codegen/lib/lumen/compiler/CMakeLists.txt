add_subdirectory(Diagnostics)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Target)
add_subdirectory(Translation)


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
set(LLVM_CONFIG_LIBNAMES_ARGV "--libnames --link-static")
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
