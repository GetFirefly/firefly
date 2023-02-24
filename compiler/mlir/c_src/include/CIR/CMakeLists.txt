add_mlir_dialect(CIR cir)

set(LLVM_TARGET_DEFINITIONS TypeInterfaces.td)
mlir_tablegen(CIRTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(CIRTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRCIRTypeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(CIRAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CIRAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(CIREnums.h.inc -gen-enum-decls)
mlir_tablegen(CIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCIRAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CIR)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix CIR)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix CIR)
add_public_tablegen_target(MLIRCIRPassIncGen)

#set(LLVM_TARGET_DEFINITIONS Patterns.pdll)
#pdlgen(Patterns.cpp.inc)
#add_public_pdlgen_target(MLIRCIRPatternsIncGen)

add_mlir_doc(Dialect Dialect CIR/ -gen-dialect-doc)
add_mlir_doc(Ops Ops CIR/ -gen-op-doc)
add_mlir_doc(Passes Passes CIR/ -gen-pass-doc)
