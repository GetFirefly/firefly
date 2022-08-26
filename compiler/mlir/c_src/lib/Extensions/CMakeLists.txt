add_mlir_dialect_library(MLIRFireflyExtensions
  Builder.cpp
  Diagnostics.cpp
  Helpers.cpp
  ModuleReader.cpp
  ModuleWriter.cpp
  Passes.cpp
  SymbolTable.cpp
  Target.cpp

  LINK_COMPONENTS
  Core
  Support
  BitWriter

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
  MLIRParser
  MLIRTargetLLVMIRExport
)
