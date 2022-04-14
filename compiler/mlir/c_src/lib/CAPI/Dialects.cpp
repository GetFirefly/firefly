#include "CIR-c/Dialects.h"
#include "CIR/Dialect.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CIR, cir, mlir::cir::CIRDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arithmetic, arith,
                                      mlir::arith::ArithmeticDialect)
