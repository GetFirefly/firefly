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
using ::lumen::SizeLevel;
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
}

}  // namespace eir
}  // namespace lumen

extern "C" MLIRPassManagerRef MLIRCreatePassManager(
    MLIRContextRef context, LLVMTargetMachineRef tm, OptLevel opt,
    SizeLevel sizeOpt, bool enableTiming, bool enableStatistics) {
  MLIRContext *ctx = unwrap(context);
  TargetMachine *targetMachine = unwrap(tm);
  CodeGenOptLevel optLevel = toLLVM(opt);
  unsigned sizeLevel = toLLVM(opt);

  auto pm = new PassManager(ctx);
  if (enableTiming) pm->enableTiming();
  if (enableStatistics) pm->enableStatistics();
  mlir::applyPassManagerCLOptions(*pm);

  bool enableOpt = optLevel > CodeGenOptLevel::None;

  OpPassManager &eirFuncOpt = pm->nest<::lumen::eir::FuncOp>();
  eirFuncOpt.addPass(mlir::createCanonicalizerPass());

  lumen::eir::buildEIRTransformPassPipeline(*pm, targetMachine);

  // Add optimizations if enabled
  if (enableOpt) {
    // When optimizing for size, avoid aggressive inlining
    if (sizeLevel == 0) {
      // pm->addPass(mlir::createInlinerPass());
    }

    OpPassManager &optPM = pm->nest<::mlir::LLVM::LLVMFuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    // Sparse conditional constant propagation
    // optPM.addPass(mlir::createSCCPPass());
    // Common sub-expression elimination
    // optPM.addPass(mlir::createCSEPass());
    // Remove dead/unreachable symbols
    // pm->addPass(mlir::createSymbolDCEPass());
  }

  return wrap(pm);
}
