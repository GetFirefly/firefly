#include "eir/Lower.h"
#include "eir/Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Target/TargetMachine.h"

#include "llvm-c/Core.h"
#include "llvm/Support/CBindingWrapping.h"

using namespace eir;

namespace M = mlir;
namespace L = llvm;

MLIRModuleRef MLIRLowerModule(MLIRContextRef context, MLIRModuleRef m,
                              TargetDialect dialect,
                              LLVMLumenCodeGenOptLevel opt) {
  M::MLIRContext *ctx = unwrap(context);
  M::ModuleOp *mod = unwrap(m);
  L::CodeGenOpt::Level optLevel = fromRust(opt);

  M::PassManager pm(&*ctx);
  M::applyPassManagerCLOptions(pm);

  bool enableOpt = optLevel >= L::CodeGenOpt::Level::None;
  bool lowerToStandard = dialect >= TargetDialect::TargetStandard;
  bool lowerToLLVM = dialect >= TargetDialect::TargetLLVM;

  if (enableOpt) {
    // Perform high-level inlining
    pm.addPass(M::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    M::OpPassManager &optPM = pm.nest<M::FuncOp>();
    optPM.addPass(M::createCanonicalizerPass());
    optPM.addPass(M::createCSEPass());
  }

  if (lowerToStandard) {
    // Lower the EIR dialect, then clean up
    pm.addPass(eir::createLowerToStandardPass());

    M::OpPassManager &optPM = pm.nest<M::FuncOp>();
    optPM.addPass(M::createCanonicalizerPass());
    optPM.addPass(M::createCSEPass());

    // Add optimizations if enabled
    if (enableOpt) {
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createMemRefDataFlowOptPass());
    }
  }

  if (lowerToLLVM) {
    // Lower from Standard to LLVM dialect
    pm.addPass(eir::createLowerToLLVMPass());
  }

  M::OwningModuleRef ownedMod(*mod);
  if (M::failed(pm.run(*ownedMod))) {
    return nullptr;
  }

  return wrap(new M::ModuleOp(ownedMod.release()));
}

LLVMModuleRef MLIRLowerToLLVMIR(MLIRModuleRef m, LLVMLumenCodeGenOptLevel opt,
                                LLVMLumenCodeGenSizeLevel size,
                                LLVMTargetMachineRef tm) {
  M::ModuleOp *mod = unwrap(m);
  L::TargetMachine *targetMachine = unwrap(tm);
  L::CodeGenOpt::Level optLevel = fromRust(opt);
  unsigned sizeLevel = fromRust(size);

  bool enableOpt = optLevel >= L::CodeGenOpt::Level::None;
  if (sizeLevel > 0) {
    optLevel = L::CodeGenOpt::Level::Default;
  }

  M::OwningModuleRef ownedMod(*mod);
  auto llvmModPtr = M::translateModuleToLLVMIR(*ownedMod);
  if (!llvmModPtr) {
    L::errs() << "Failed to emit LLVM IR!\n";
    return nullptr;
  }

  M::ExecutionEngine::setupTargetTriple(llvmModPtr.get());

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
