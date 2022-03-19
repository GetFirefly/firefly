#include "CIR-c/Dialects.h"
#include "CIR/Dialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CIR, cir, mlir::cir::CIRDialect)
