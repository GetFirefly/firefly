#include "lumen/mlir/MLIR.h"
#include "lumen/llvm/Target.h"

#include "mlir/Target/LLVMIR.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"

#include "llvm-c/Core.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Target/TargetMachine.h"

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OpPassManager;
using ::mlir::OwningModuleRef;
using ::mlir::PassManager;
using ::llvm::LLVMContext;
using ::llvm::StringRef;
using ::llvm::TargetMachine;
using ::llvm::Triple;
using ::llvm::unwrap;

using namespace lumen;

extern "C" bool MLIRVerifyModule(MLIRModuleRef m) {
  ModuleOp *mod = unwrap(m);
  if (mlir::failed(mlir::verify(mod->getOperation())))
    return false;

  return true;
}

extern "C" {
  struct LowerResult {
    void *module;
    bool success;
  };
}

extern "C" LowerResult MLIRLowerModule(MLIRContextRef context,
                                       MLIRPassManagerRef passManager,
                                       MLIRModuleRef m) {
  MLIRContext *ctx = unwrap(context);
  PassManager *pm = unwrap(passManager);
  ModuleOp *mod = unwrap(m);
  LowerResult result;

  OwningModuleRef ownedMod(*mod);
  if (mlir::failed(pm->run(*ownedMod))) {
    MLIRModuleRef ptr = wrap(new ModuleOp(ownedMod.release()));
    
    result = {.module = (void *)(ptr), .success = false};
    return result;
  }

  MLIRModuleRef ptr = wrap(new ModuleOp(ownedMod.release()));
  result = {.module = (void *)(ptr), .success = true};
  return result;
}


extern "C" LowerResult MLIRLowerToLLVMIR(MLIRModuleRef m,
                                         LLVMContextRef context,
                                         LLVMTargetMachineRef tm,
                                         const char *sourceName,
                                         unsigned sourceNameLen) {
  LLVMContext *ctx = unwrap(context);
  ModuleOp *mod = unwrap(m);
  StringRef srcName(sourceName, sourceNameLen);
  StringRef modName;
  if (auto mn = mod->getName())
    modName = mn.getValue();
  else
    modName = StringRef("unknown");

  OwningModuleRef ownedMod(*mod);
  auto llvmModPtr = mlir::translateModuleToLLVMIR(*ownedMod, *ctx, modName);
  if (!llvmModPtr) {
    MLIRModuleRef ptr = wrap(new ModuleOp(ownedMod.release()));
    return {.module = (void *)(ptr), .success = false};
  }

  TargetMachine *targetMachine = unwrap(tm);
  Triple triple = targetMachine->getTargetTriple();
  llvmModPtr->setDataLayout(targetMachine->createDataLayout());
  llvmModPtr->setTargetTriple(triple.getTriple());
  llvmModPtr->setModuleIdentifier(modName);
  llvmModPtr->setSourceFileName(srcName);

  LLVMModuleRef ptr = wrap(llvmModPtr.release());
  return {.module = (void *)(ptr), .success = true};
}
