# Make sure we've provided all of the necessary include directories
include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}" "${MLIR_BINARY_DIR}/include")

# Provide the linker with the correct search path for LLVM libraries
link_directories("${LLVM_LIBRARY_DIR}")

llvm_map_components_to_libnames(LIB_NAMES ${LLVM_CONFIG_COMPONENTS_LIST})

# Define the MLIR libraries we need to link against
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
  MLIRLinalgOps
  MLIRLinalgToLLVM
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
  MLIRVectorToLoops
  MLIRVectorToLLVM
)

list(REMOVE_ITEM LIB_NAMES "LLVMTableGen")
list(REMOVE_ITEM LIB_NAMES "LLVMjit")
list(REMOVE_DUPLICATES LIB_NAMES)

message(STATUS "LIB NAMES == ${LIB_NAMES}")

add_llvm_library(LumenLLVM
  STATIC
  SHARED
  DISABLE_LLVM_LINK_LLVM_DYLIB
  SONAME

  libllvm.cpp
)

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

target_link_libraries(LumenLLVM PRIVATE ${LIB_NAMES})

if (APPLE)
  set_property(TARGET LumenLLVM APPEND_STRING PROPERTY
              LINK_FLAGS
              " -compatibility_version 1 -current_version ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
endif()
