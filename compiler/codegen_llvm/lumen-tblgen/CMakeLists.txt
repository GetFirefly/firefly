lumen_cc_binary(
  NAME
    lumen_tblgen
  OUT
    lumen-tblgen
  SRCS
    "lumen-tblgen.cpp"
  DEPS
    lumen::EIR::Tools::TableGen
    MLIRSupport
    MLIRTableGen
  LLVM_COMPONENTS
    LLVMSupport
    LLVMTableGen
  LINKOPTS
    "-lpthread"
)
add_executable(lumen-tblgen ALIAS lumen-tblgen_lumen_tblgen)
