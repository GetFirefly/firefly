#include "CIR-c/Dialects.h"
#include "CIR/Dialect.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(CIR, cir, mlir::cir::CIRDialect)

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Arithmetic, arith,
                                      mlir::arith::ArithDialect)

MlirTypeID mlirDialectHandleGetTypeID(MlirDialect dialect) {
  return wrap(unwrap(dialect)->getTypeID());
}
