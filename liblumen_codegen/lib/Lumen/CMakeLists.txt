# Make sure we've provided all of the necessary include directories
include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}" "${MLIR_BINARY_DIR}/include")

# Provide the linker with the correct search path for LLVM libraries
link_directories("${LLVM_LIBRARY_DIR}")

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

# Append the MLIR libraries we need to link against
set(LIB_NAMES
  ${LIB_NAMES}
  MLIRAnalysis
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRDialect
  MLIREDSC
  MLIREDSCInterface
  MLIRExecutionEngine
  MLIRFxpMathOps
  MLIRLinalgAnalysis
  MLIRLinalgEDSC
  MLIRLinalgOps
  MLIRLinalgToLLVM
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRIR
  MLIRLLVMIR
  MLIRLoopOps
  MLIRLoopToStandard
  MLIRParser
  MLIRPass
  MLIRQuantOps
  MLIRQuantizerFxpMathConfig
  MLIRQuantizerSupport
  MLIRQuantizerTransforms
  MLIRSDBM
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIRTransformUtils
  MLIRTransforms
  MLIRTranslateClParser
  MLIRTranslation
  MLIRVectorOps
  MLIRVectorToLLVM
  MLIRVectorToLoops
)

# Strip undesirable dependencies which are returned by LLVM but not actually used
list(REMOVE_ITEM LIB_NAMES "LLVMTableGen")
list(REMOVE_ITEM LIB_NAMES "LLVMLineEditor")
list(REMOVE_ITEM LIB_NAMES "LLVMjit")

# Clean up the list
list(REMOVE_DUPLICATES LIB_NAMES)

# Define the libLumen target
file(GLOB globbed *.cpp EIR/*.cpp)

add_llvm_library(Lumen
  ${globbed}

  STATIC
  DISABLE_LLVM_LINK_LLVM_DYLIB
  
  DEPENDS
  OpsIncGen
)

file(WRITE ${CMAKE_BINARY_DIR}/Lumen_deps.txt "")
foreach(lib IN LISTS LIB_NAMES)
  file(APPEND ${CMAKE_BINARY_DIR}/Lumen_deps.txt
       "${lib}\n")
endforeach()

if(("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (MINGW) OR (HAIKU)
    OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "FreeBSD")
    OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "GNU")
    OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "OpenBSD")
    OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "Fuchsia")
    OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "DragonFly")
    OR ("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")) # FIXME: It should be "GNU ld for elf"
  configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_version_script.map.in
  ${CMAKE_CURRENT_BINARY_DIR}/simple_version_script.map)

  # GNU ld doesn't resolve symbols in the version script.
  set(LIB_NAMES -Wl,--whole-archive ${LIB_NAMES} -Wl,--no-whole-archive)
  if (NOT LLVM_LINKER_IS_SOLARISLD AND NOT MINGW)
    # Solaris ld does not accept global: *; so there is no way to version *all* global symbols
    set(LIB_NAMES -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/simple_version_script.map ${LIB_NAMES})
  endif()
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  set(LIB_NAMES -Wl,-all_load ${LIB_NAMES})
endif()

target_link_libraries(Lumen PRIVATE ${LIB_NAMES})
    
install(TARGETS Lumen
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
