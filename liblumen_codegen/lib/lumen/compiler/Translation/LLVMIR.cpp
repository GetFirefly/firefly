#include "lumen/compiler/Support/MLIR.h"
#include "lumen/compiler/Target/Target.h"
#include "lumen/compiler/Target/TargetInfo.h"
#include "lumen/compiler/Dialect/EIR/Transforms/Passes.h"
#include "lumen/compiler/Dialect/EIR/Conversion/EIRToStandard/ConvertEIRToStandard.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"

#include "llvm-c/Core.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Target/TargetMachine.h"


using namespace lumen;
using namespace lumen::eir;

using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::FuncOp;
using ::mlir::PassManager;
using ::mlir::OpPassManager;
using ::mlir::OwningModuleRef;
using ::llvm::TargetMachine;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef);

using CodeGenOptLevel = ::llvm::CodeGenOpt::Level;

TargetInfo::TargetInfo(TargetMachine *targetMachine) {
  auto triple = targetMachine->getTargetTriple();
  auto dataLayout = targetMachine->createDataLayout();
  archType = triple.getArch();
  pointerSizeInBits = dataLayout.getPointerSizeInBits(0);
}

extern "C" MLIRModuleRef
MLIRLowerModule(MLIRContextRef context, MLIRModuleRef m,
                TargetDialect dialect,
                OptLevel opt,
                LLVMTargetMachineRef tm) {
  MLIRContext *ctx = unwrap(context);
  ModuleOp *mod = unwrap(m);
  TargetMachine *targetMachine = unwrap(tm);
  CodeGenOptLevel optLevel = toLLVM(opt);

  PassManager pm(&*ctx);
  mlir::applyPassManagerCLOptions(pm);

  bool enableOpt = optLevel >= CodeGenOptLevel::None;
  bool lowerToStandard = dialect >= TargetDialect::TargetStandard;
  bool lowerToLLVM = dialect >= TargetDialect::TargetLLVM;

  if (enableOpt) {
    // Perform high-level inlining
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    OpPassManager &optPM = pm.nest<eir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (lowerToStandard) {
    TargetInfo targetInfo(targetMachine);
    buildEIRTransformPassPipeline(pm);

    OpPassManager &optPM = pm.nest<mlir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled
    if (enableOpt) {
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createMemRefDataFlowOptPass());
    }
  }

  if (lowerToLLVM) {
    // Lower from Standard to LLVM dialect
    pm.addPass(mlir::createLowerToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  OwningModuleRef ownedMod(*mod);
  if (mlir::failed(pm.run(*ownedMod))) {
    return nullptr;
  }
  ownedMod->dump();

  return wrap(new ModuleOp(ownedMod.release()));
}

extern "C" LLVMModuleRef
MLIRLowerToLLVMIR(MLIRModuleRef m,
                  OptLevel opt,
                  SizeLevel size,
                  LLVMTargetMachineRef tm) {
  ModuleOp *mod = unwrap(m);
  TargetMachine *targetMachine = unwrap(tm);
  CodeGenOptLevel optLevel = toLLVM(opt);
  unsigned sizeLevel = toLLVM(size);

  bool enableOpt = optLevel >= CodeGenOptLevel::None;
  if (sizeLevel > 0) {
    optLevel = CodeGenOptLevel::Default;
  }

  OwningModuleRef ownedMod(*mod);
  auto llvmModPtr = mlir::translateModuleToLLVMIR(*ownedMod);
  if (!llvmModPtr) {
    llvm::errs() << "Failed to emit LLVM IR!\n";
    return nullptr;
  }

  mlir::ExecutionEngine::setupTargetTriple(llvmModPtr.get());

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
