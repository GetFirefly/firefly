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
    EIRTypesGen
  TD_FILE
    EIRBase.td
  OUTS
    -gen-typedef-decls EIRTypes.h.inc
    -gen-typedef-defs EIRTypes.cpp.inc
    -gen-type-interface-decls EIRTypeInterfaces.h.inc
    -gen-type-interface-defs EIRTypeInterfaces.cpp.inc
  EXTRA
    -typedefs-dialect=eir
)

lumen_tablegen_library(
  NAME
    EIRAttributesGen
  TD_FILE
    EIRBase.td
  OUTS
    -gen-attrdef-decls EIRAttributes.h.inc
    -gen-attrdef-defs EIRAttributes.cpp.inc
    -gen-struct-attr-decls EIRStructs.h.inc
    -gen-struct-attr-defs EIRStructs.cpp.inc
    -gen-attr-interface-decls EIRAttrInterfaces.h.inc
    -gen-attr-interface-defs EIRAttrInterfaces.cpp.inc
  EXTRA
    -attrdefs-dialect=eir
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
    -gen-op-interface-decls EIROpInterfaces.h.inc
    -gen-op-interface-defs EIROpInterfaces.cpp.inc
)

lumen_cc_library(
  HDRS
    "EIRAttributes.h"
    "EIRDialect.h"
    "EIREnums.h"
    "EIROpInterfaces.h"
    "EIROps.h"
    "EIRTypes.h"
    "TargetPlatform.h"
  SRCS
    "EIRAttributes.cpp"
    "EIRDialect.cpp"
    "EIREnums.cpp"
    "EIROpInterfaces.cpp"
    "EIROps.cpp"
    "EIRTypes.cpp"
    "TargetPlatform.cpp"
    "CAPI.cpp"
  DEPS
    lumen::EIR::IR::EIRDialectGen
    lumen::EIR::IR::EIREnumsGen
    lumen::EIR::IR::EIROpsGen
    lumen::EIR::IR::EIRTypesGen
    lumen::EIR::IR::EIRAttributesGen
    lumen::EIR::IR::EIROpInterfaceGen
    lumen::EIR::IR::EIREncodingGen
    MLIRIR
    MLIRParser
    MLIRStandard
    MLIRSupport
    MLIRTransformUtils
  LLVM_COMPONENTS
    LLVMSupport
  PUBLIC
)
