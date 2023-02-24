#pragma once

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FileLineColLoc {
  unsigned line;
  unsigned column;
  unsigned filenameLen;
  const char *filename;
};

MLIR_CAPI_EXPORTED FileLineColLoc
mlirDiagnosticGetFileLineCol(MlirDiagnostic d);

MLIR_CAPI_EXPORTED void mlirContextSetPrintOpOnDiagnostic(MlirContext context,
                                                          bool enable);
MLIR_CAPI_EXPORTED void
mlirContextSetPrintStackTraceOnDiagnostic(MlirContext context, bool enable);

#ifdef __cplusplus
}
#endif
