add_mlir_public_c_api_library(CIRCAPI
  Attributes.cpp
  Builder.cpp
  Dialects.cpp
  Diagnostics.cpp
  Module.cpp
  Operation.cpp
  Ops.cpp
  Options.cpp
  Passes.cpp
  SymbolTable.cpp
  Translate.cpp
  Types.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_COMPONENTS
  Core
  Support
  BitWriter

  LINK_LIBS PUBLIC
  CIR
  MLIRCAPIIR
  MLIRParser
  MLIRTargetLLVMIRExport
)

add_mlir_aggregate(CIRDynamic
  SHARED
  PUBLIC_LIBS
  CIR

  EMBED_LIBS
  MLIRCAPIIR
  MLIRCAPIControlFlow
  MLIRCAPILLVM
  MLIRCAPISCF
  MLIRCAPIFunc
  CIRCAPI
)

add_mlir_aggregate(CIRStatic
  STATIC
  PUBLIC_LIBS
  CIR

  EMBED_LIBS
  MLIRCAPIIR
  MLIRCAPIControlFlow
  MLIRCAPILLVM
  MLIRCAPISCF
  MLIRCAPIFunc
  CIRCAPI
)
