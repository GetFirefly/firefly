#include "mlir/Transforms/Passes.h"

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

using ::llvm::TargetMachine;
using ::llvm::unwrap;
using ::lumen::CodeGenOptLevel;
using ::lumen::OptLevel;
using ::lumen::SizeLevel;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OpPassManager;
using ::mlir::OpPrintingFlags;
using ::mlir::OwningModuleRef;
using ::mlir::Pass;
using ::mlir::PassDisplayMode;
using ::mlir::PassManager;

extern "C" {
struct PassManagerOptions {
  OptLevel optLevel;
  SizeLevel sizeLevel;
  bool enableTiming;
  bool enableStatistics;
  bool printBeforePass;
  bool printAfterPass;
  bool printModuleScopeAlways;
  bool printAfterOnlyOnChange;
};
}

extern "C" MLIRPassManagerRef MLIRCreatePassManager(
    MLIRContextRef context, LLVMTargetMachineRef tm,
    PassManagerOptions *options) {
  MLIRContext *ctx = unwrap(context);
  TargetMachine *targetMachine = unwrap(tm);
  CodeGenOptLevel optLevel = toLLVM(options->optLevel);
  unsigned sizeLevel = toLLVM(options->sizeLevel);
  bool printBeforePass = options->printBeforePass;
  bool printAfterPass = options->printAfterPass;

  auto pm = new PassManager(ctx, /*verifyPasses=*/true);

  // Configure IR printing
  OpPrintingFlags printerFlags;
  printerFlags.enableDebugInfo(/*pretty=*/true);
  printerFlags.useLocalScope();
  pm->enableIRPrinting(
      /*shouldPrintBefore=*/[printBeforePass](
                                Pass *,
                                Operation *) { return printBeforePass; },
      /*shouldPrintAfter=*/
      [printAfterPass](Pass *, Operation *) { return printAfterPass; },
      /*printModuleScopeAlways=*/options->printModuleScopeAlways,
      /*printAfterOnlyOnChange=*/options->printAfterOnlyOnChange, llvm::errs(),
      printerFlags);

  // Configure Pass Timing
  if (options->enableTiming) {
    auto config =
        std::make_unique<PassManager::PassTimingConfig>(PassDisplayMode::List);
    pm->enableTiming(std::move(config));
  }
  if (options->enableStatistics) {
    pm->enableStatistics(PassDisplayMode::List);
  }

  // Allow command-line options to override the default configuration
  mlir::applyPassManagerCLOptions(*pm);

  // TODO: Hook driver into instrumentation
  // pm.addInstrumentation(...);

  // Convert EIR to LLVM dialect
  pm->addPass(::lumen::eir::createConvertEIRToLLVMPass(targetMachine));

  // Canonicalize
  pm->addNestedPass<::mlir::LLVM::LLVMFuncOp>(mlir::createCanonicalizerPass());

  // Add optimizations if enabled
  if (optLevel > CodeGenOptLevel::None) {
    // When optimizing for size, avoid aggressive inlining
    if (sizeLevel == 0) {
      // pm->addPass(mlir::createInlinerPass());
    }

    // Canonicalize generated LLVM dialect, and perform optimizations
    // OpPassManager &optPM = pm->nest<::mlir::LLVM::LLVMFuncOp>();
    // optPM.addPass(mlir::createCanonicalizerPass());
    // Sparse conditional constant propagation
    // optPM.addPass(mlir::createSCCPPass());
    // Common sub-expression elimination
    // optPM.addPass(mlir::createCSEPass());
    // Remove dead/unreachable symbols
    // pm->addPass(mlir::createSymbolDCEPass());
  }

  return wrap(pm);
}
