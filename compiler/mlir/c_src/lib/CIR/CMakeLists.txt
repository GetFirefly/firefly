get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_mlir_dialect_library(CIR
  Dialect.cpp
  Attributes.cpp
  ConvertCIRToLLVMPass.cpp
  Ops.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIR

  DEPENDS
  MLIRCIRIncGen
  MLIRCIRTypeInterfacesIncGen
  MLIRCIRAttributesIncGen
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  ${dialect_libs}
  ${translation_libs}

  MLIRAnalysis
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
)
