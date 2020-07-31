lumen_tablegen_library(
  NAME
    EIRDialectGen
  TD_FILE
    EIROps.td
  OUTS
    -gen-dialect-decls EIRDialect.h.inc
  EXTRA
    -dialect=eir
)

lumen_tablegen_library(
  NAME
    EIREnumsGen
  TD_FILE
    EIRBase.td
  OUTS
    -gen-enum-decls EIREnums.h.inc
    -gen-enum-defs EIREnums.cpp.inc
)

lumen_tablegen_library(
  NAME
    EIROpsGen
  TD_FILE
    EIROps.td
  OUTS
    -gen-op-decls EIROps.h.inc
    -gen-op-defs EIROps.cpp.inc
)

lumen_tablegen_library(
  NAME
    EIREncodingGen
  TD_FILE
    EIROps.td
  OUTS
    -gen-eir-encoding-defs EIREncoding.h.inc
  TBLGEN
    LUMEN
)

lumen_tablegen_library(
  NAME
    EIROpInterfaceGen
  TD_FILE
    EIRBase.td
  OUTS
    -gen-op-interface-decls EIROpInterface.h.inc
    -gen-op-interface-defs EIROpInterface.cpp.inc
)

lumen_cc_library(
  HDRS
    "EIRAttributes.h"
    "EIRDialect.h"
    "EIREnums.h"
    "EIROpInterface.h"
    "EIROps.h"
    "EIROps.h.inc"
    "EIRTraits.h"
    "EIRTypes.h"
  SRCS
    "EIRAttributes.cpp"
    "EIRDialect.cpp"
    "EIREnums.cpp"
    "EIROps.cpp"
    "EIROps.cpp.inc"
    "EIRTypes.cpp"
  DEPS
    lumen::EIR::IR::EIRDialectGen
    lumen::EIR::IR::EIREnumsGen
    lumen::EIR::IR::EIROpsGen
    lumen::EIR::IR::EIROpInterfaceGen
    lumen::EIR::IR::EIREncodingGen
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
  PUBLIC
)
