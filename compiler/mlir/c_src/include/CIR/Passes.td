#ifndef CIR_PASSES_TD
#define CIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertCIRToLLVM : Pass<"convert-cir-to-llvm", "ModuleOp"> {
    let summary = "Convert CIR dialect to LLVM dialect";
    let description = [{
      This pass converts supported CIR ops to LLVM dialect instructions.
    }];
    let constructor = "mlir::cir::createConvertCIRToLLVMPass()";
    let dependentDialects = ["CIRDialect", "LLVM::LLVMDialect"];
    let options = [
      Option<"enableNanboxing", "enable-nanboxing", "bool",
             /*default=*/"false",
             "When enabled, terms are encoded using a nanboxed representation if the data layout supports it">,
    ];
}

#endif
