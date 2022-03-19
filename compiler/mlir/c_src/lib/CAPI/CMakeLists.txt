add_mlir_public_c_api_library(CIRCAPI
  Dialects.cpp
  Types.cpp
  Attributes.cpp
  Ops.cpp
  Passes.cpp

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  CIR
)
