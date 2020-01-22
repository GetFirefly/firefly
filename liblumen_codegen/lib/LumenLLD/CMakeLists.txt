# Make sure we've provided all of the necessary include directories
include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}")

# Provide the linker with the correct search path for LLVM libraries
link_directories("${LLVM_LIBRARY_DIR}")

# Define the LLD libraries we need to link against
# NOTE: We link against LLD for the built-in linker
set(LIB_NAMES
  lldCommon
  lldCOFF
  lldDriver
  lldELF
  lldMinGW
  lldWasm
  lldMachO
  lldYAML
  lldReaderWriter
)

file(WRITE ${CMAKE_BINARY_DIR}/LumenLLD_deps.txt "")
foreach(lib IN LISTS LIB_NAMES)
  file(APPEND ${CMAKE_BINARY_DIR}/LumenLLD_deps.txt
       "${lib}\n")
endforeach()

# Define the libLumenLLD target
add_llvm_library(LumenLLD
  Linker.cpp

  STATIC
  
  LINK_COMPONENTS
  support
  
  LINK_LIBS
  ${LIB_NAMES}
)

install(TARGETS LumenLLD
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
