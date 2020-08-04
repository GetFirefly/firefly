#include "lumen/EIR/Conversion/Passes.h"

#include <memory>

#include "llvm/Target/TargetMachine.h"
#include "lumen/EIR/Conversion/ConvertEIRToLLVM.h"
#include "lumen/EIR/IR/EIROps.h"
#include "lumen/llvm/Target.h"
#include "lumen/mlir/MLIR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using ::llvm::TargetMachine;
using ::llvm::unwrap;
using ::lumen::CodeGenOptLevel;
using ::lumen::OptLevel;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OpPassManager;
using ::mlir::OwningModuleRef;
using ::mlir::PassManager;

namespace lumen {
namespace eir {

void buildEIRTransformPassPipeline(mlir::OpPassManager &passManager,
                                   llvm::TargetMachine *targetMachine) {
  passManager.addPass(createConvertEIRToLLVMPass(targetMachine));
  OpPassManager &optPM = passManager.nest<::mlir::LLVM::LLVMFuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
  // passManager.addPass(createGlobalInitializationPass());
  // TODO: run symbol DCE pass.
}

}  // namespace eir
}  // namespace lumen

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
  if (enableTiming) pm->enableTiming();
  if (enableStatistics) pm->enableStatistics();

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
    OpPassManager &optPM = pm->nest<::mlir::LLVM::LLVMFuncOp>();
    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  return wrap(pm);
}
