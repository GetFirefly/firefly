#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Pass.h"

#include "CIR/Passes.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

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

MLIR_CAPI_EXPORTED void
mlirContextRegisterLLVMDialectTranslation(MlirContext ctx);

MLIR_CAPI_EXPORTED void
mlirPassManagerEnableIRPrintingWithFlags(MlirPassManager pm,
                                         IrPrintingConfig *c);
MLIR_CAPI_EXPORTED void mlirPassManagerEnableStatistics(MlirPassManager pm);

MLIR_CAPI_EXPORTED void mlirPassManagerEnableTiming(MlirPassManager pm);

MLIR_CAPI_EXPORTED void mlirPassManagerEnableCrashReproducerGeneration(
    MlirPassManager pm, MlirStringRef outputFile, bool genLocalReproducer);

MLIR_CAPI_EXPORTED void mlirPassDestroy(MlirPass pass);

#ifdef __cplusplus
}
#endif
