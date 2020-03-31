#include "lumen/mlir/MLIR.h"
#include "lumen/llvm/Target.h"
#include "lumen/compiler/Target/TargetInfo.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRDialect.h"
#include "lumen/compiler/Dialect/EIR/Transforms/Passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Module.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using ::lumen::OptLevel;
using ::lumen::CodeGenOptLevel;
using ::llvm::TargetMachine;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OpPassManager;
using ::mlir::OwningModuleRef;
using ::mlir::PassManager;

extern "C"
void MLIRRegisterDialects() {
  // Initializing the command-line options more than once is not allowed.
  // So check if they've already been initialized.
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  // Register the EIR dialect with MLIR
  mlir::registerDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerDialect<lumen::eir::EirDialect>();
}

extern "C" MLIRPassManagerRef MLIRCreatePassManager(MLIRContextRef context,
                                                    LLVMTargetMachineRef tm,
                                                    OptLevel opt,
                                                    bool enableTiming,
                                                    bool enableStatistics) {
  MLIRContext *ctx = unwrap(context);
  TargetMachine *targetMachine = unwrap(tm);
  CodeGenOptLevel optLevel = toLLVM(opt);

  auto pm = new PassManager(ctx);
  mlir::applyPassManagerCLOptions(*pm);
  if (enableTiming)
    pm->enableTiming();
  if (enableStatistics)
    pm->enableStatistics();

  bool enableOpt = optLevel >= CodeGenOptLevel::None;

  if (enableOpt) {
    // Perform high-level inlining
    // pm.addPass(mlir::createInlinerPass());

    OpPassManager &optPM = pm->nest<::lumen::eir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  lumen::eir::buildEIRTransformPassPipeline(*pm, targetMachine);

  // Add optimizations if enabled
  if (enableOpt) {
    OpPassManager &optPM = pm->nest<::mlir::FuncOp>();
    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  return wrap(pm);
}
