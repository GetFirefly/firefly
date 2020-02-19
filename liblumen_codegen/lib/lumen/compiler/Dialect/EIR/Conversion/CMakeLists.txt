add_subdirectory(EIRToStandard)

lumen_cc_library(
  NAME
    Conversion
  HDRS
    "TypeConverter.h"
  SRCS
    "TypeConverter.cpp"
  DEPS
    lumen::compiler::Dialect::EIR::IR
    MLIRIR
    MLIRStandardOps
    MLIRTransforms
  PUBLIC
  SHARED
)
