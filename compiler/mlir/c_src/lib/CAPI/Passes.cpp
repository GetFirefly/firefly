#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "CIR-c/Passes.h"
#include "CIR/Passes.h"

using namespace mlir;
using namespace mlir::cir;

#include "CIR/Passes.capi.cpp.inc"

void mlirContextRegisterLLVMDialectTranslation(MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);
  registerLLVMDialectTranslation(*context);
}

void mlirPassManagerEnableIRPrintingWithFlags(MlirPassManager pm,
                                              IrPrintingConfig *c) {
  IrPrintingConfig config = *c;
  OpPrintingFlags flags;
  if (config.enableDebugInfo) {
    flags.enableDebugInfo(config.enablePrettyDebugInfo);
  }
  if (config.printGenericForm) {
    flags.printGenericOpForm();
  }
  if (config.useLocalScope) {
    flags.useLocalScope();
  }
  auto printBeforePass = config.printBeforePass;
  auto printAfterPass = config.printAfterPass;
  auto shouldPrintBeforePass = [=](Pass *, Operation *) {
    return printBeforePass;
  };
  auto shouldPrintAfterPass = [=](Pass *, Operation *) {
    return printAfterPass;
  };
  unwrap(pm)->enableIRPrinting(
      shouldPrintBeforePass, shouldPrintAfterPass, config.printModuleScope,
      config.printOnlyAfterChange, config.printOnlyAfterFailure, llvm::errs(),
      flags);
}

void mlirPassManagerEnableStatistics(MlirPassManager pm) {
  unwrap(pm)->enableStatistics();
}

void mlirPassManagerEnableTiming(MlirPassManager pm) {
  unwrap(pm)->enableTiming();
}

void mlirPassManagerEnableCrashReproducerGeneration(MlirPassManager pm,
                                                    MlirStringRef outputFile,
                                                    bool genLocalReproducer) {
  unwrap(pm)->enableCrashReproducerGeneration(unwrap(outputFile),
                                              genLocalReproducer);
}

/// Necessary to ensure Drop is properly implemented for OwnedPass
void mlirPassDestroy(MlirPass pass) { delete unwrap(pass); }
