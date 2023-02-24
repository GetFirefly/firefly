#pragma once

#include "mlir-c/Support.h"

extern "C" {
MLIR_CAPI_EXPORTED void LLVMFireflyInstallFatalErrorHandler(void);
}
