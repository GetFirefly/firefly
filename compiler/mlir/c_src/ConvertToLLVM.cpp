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
    ownedMod->emitError("unable to lower to llvm dialect");
    return nullptr;
  }

  return wrap(new ModuleOp(ownedMod.release()));
}


extern "C" LLVMModuleRef MLIRLowerToLLVMIR(MLIRModuleRef m,
                                           const char *sourceName, OptLevel opt,
                                           SizeLevel size,
                                           LLVMTargetMachineRef tm) {
  ModuleOp *mod = unwrap(m);
  TargetMachine *targetMachine = unwrap(tm);
  auto targetTriple = targetMachine->getTargetTriple();
  CodeGenOptLevel optLevel = toLLVM(opt);
  unsigned sizeLevel = toLLVM(size);

  bool enableOpt = optLevel >= CodeGenOptLevel::None;
  if (sizeLevel > 0) {
    optLevel = CodeGenOptLevel::Default;
  }

  auto modName = mod->getName();

  OwningModuleRef ownedMod(*mod);
  auto llvmModPtr = mlir::translateModuleToLLVMIR(*ownedMod);
  if (!llvmModPtr) {
    llvm::errs() << "Failed to emit LLVM IR!\n";
    return nullptr;
  }

  llvmModPtr->setModuleIdentifier(modName.getValue());
  if (sourceName != nullptr) {
    llvmModPtr->setSourceFileName(StringRef(sourceName));
  }
  llvmModPtr->setDataLayout(targetMachine->createDataLayout());
  llvmModPtr->setTargetTriple(targetTriple.getTriple());

  // mlir::ExecutionEngine::setupTargetTriple(llvmModPtr.get());

  // L::outs() << L::format("Making optimizing transformer with %p",
  // targetMachine) << "\n";
  // Optionally run an optimization pipeline
  // auto optPipeline =
  // M::makeOptimizingTransformer(optLevel, sizeLevel, targetMachine);
  // if (auto err = optPipeline(llvmModPtr.get())) {
  // L::errs() << "Failed to optimize LLVM IR " << err << "\n";
  // return nullptr;
  //}

  return wrap(llvmModPtr.release());
}
