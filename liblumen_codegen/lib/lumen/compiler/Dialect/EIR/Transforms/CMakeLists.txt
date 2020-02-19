# add_subdirectory(test)

lumen_cc_library(
  NAME
    Transforms
  HDRS
    "Passes.h"
  SRCS
    "Passes.cpp"
  DEPS
    lumen::compiler::Dialect::EIR::Conversion
    lumen::compiler::Dialect::EIR::Conversion::EIRToStandard
    lumen::compiler::Dialect::EIR::IR
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
    MLIRTransforms
  ALWAYSLINK
  PUBLIC
  SHARED
)
