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

#include "llvm-c/Core.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OpPassManager;
using ::mlir::OwningModuleRef;
using ::mlir::PassManager;
using ::llvm::StringRef;
using ::llvm::TargetMachine;
using ::llvm::Triple;

using namespace lumen;

extern "C" MLIRModuleRef MLIRLowerModule(MLIRContextRef context,
                                         MLIRPassManagerRef passManager,
                                         LLVMTargetMachineRef tm,
                                         MLIRModuleRef m) {
  MLIRContext *ctx = unwrap(context);
  PassManager *pm = unwrap(passManager);
  ModuleOp *mod = unwrap(m);
  TargetMachine *targetMachine = unwrap(tm);

  OwningModuleRef ownedMod(*mod);
  if (mlir::failed(pm->run(*ownedMod))) {
    ownedMod->dump();
    llvm::outs() << "\n";
    return nullptr;
  }

  return wrap(new ModuleOp(ownedMod.release()));
}


extern "C" LLVMModuleRef MLIRLowerToLLVMIR(MLIRModuleRef m,
                                           LLVMTargetMachineRef tm,
                                           const char *sourceName,
                                           unsigned sourceNameLen) {
  ModuleOp *mod = unwrap(m);
  TargetMachine *targetMachine = unwrap(tm);
  Triple triple = targetMachine->getTargetTriple();

  auto modName = mod->getName();

  OwningModuleRef ownedMod(*mod);
  auto llvmModPtr = mlir::translateModuleToLLVMIR(*ownedMod);
  if (!llvmModPtr)
    return nullptr;

  llvmModPtr->setDataLayout(targetMachine->createDataLayout());
  llvmModPtr->setTargetTriple(triple.getTriple());
  llvmModPtr->setModuleIdentifier(modName.getValue());
  if (sourceName != nullptr)
    llvmModPtr->setSourceFileName(StringRef(sourceName, sourceNameLen));

  return wrap(llvmModPtr.release());
}
