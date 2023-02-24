add_mlir_public_c_api_library(LLVMCAPI
  Archives.cpp
  Diagnostics.cpp
  ErrorHandling.cpp
  IR.cpp
  Passes.cpp
  raw_win32_handle_ostream.cpp
  Target.cpp
  ModuleWriter.cpp

  LINK_COMPONENTS
  AsmParser
  BitReader
  BitWriter
  Core
  Instrumentation
  IPO
  Linker
  LTO
  Support
  LLVMTarget
)

add_mlir_aggregate(LLVMDynamic
  SHARED
  PUBLIC_LIBS
  LLVMAsmParser
  LLVMBitReader
  LLVMBitWriter
  LLVMCore
  LLVMInstrumentation
  LLVMipo
  LLVMLinker
  LLVMLTO
  LLVMSupport
  LLVMTarget

  EMBED_LIBS
  LLVMCAPI
)

add_mlir_aggregate(LLVMStatic
  STATIC
  PUBLIC_LIBS
  LLVMAsmParser
  LLVMBitReader
  LLVMBitWriter
  LLVMCore
  LLVMInstrumentation
  LLVMipo
  LLVMLinker
  LLVMLTO
  LLVMSupport
  LLVMTarget

  EMBED_LIBS
  LLVMCAPI
)
