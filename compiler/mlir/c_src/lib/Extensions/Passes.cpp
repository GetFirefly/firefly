#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::llvm::Twine;

extern "C" void mlirContextRegisterLLVMDialectTranslation(MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);
  registerLLVMDialectTranslation(*context);
}

extern "C" {
struct IrPrintingConfig {
  bool printBeforePass;
  bool printAfterPass;
  bool printModuleScope;
  bool printOnlyAfterChange;
  bool printOnlyAfterFailure;
  bool enableDebugInfo;
  bool enablePrettyDebugInfo;
  bool printGenericForm;
  bool useLocalScope;
};
}

extern "C" void mlirPassManagerEnableIRPrintingWithFlags(MlirPassManager pm,
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

extern "C" void mlirPassManagerEnableStatistics(MlirPassManager pm) {
  unwrap(pm)->enableStatistics();
}

extern "C" void mlirPassManagerEnableTiming(MlirPassManager pm) {
  unwrap(pm)->enableTiming();
}

extern "C" void mlirPassManagerEnableCrashReproducerGeneration(
    MlirPassManager pm, MlirStringRef outputFile, bool genLocalReproducer) {
  unwrap(pm)->enableCrashReproducerGeneration(unwrap(outputFile),
                                              genLocalReproducer);
}

/// Necessary to ensure Drop is properly implemented for OwnedPass
extern "C" void mlirPassDestroy(MlirPass pass) { delete unwrap(pass); }
