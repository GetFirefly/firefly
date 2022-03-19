#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace mlir;

using llvm::LLVMContext;

extern "C" LLVMModuleRef mlirTranslateModuleToLLVMIR(MlirModule module,
                                                     LLVMContextRef llvmContext,
                                                     MlirStringRef name) {
  LLVMContext *context = llvm::unwrap(llvmContext);
  auto llvmModPtr =
      translateModuleToLLVMIR(unwrap(module), *context, unwrap(name));
  if (!llvmModPtr)
    return nullptr;

  return llvm::wrap(llvmModPtr.release());
}
