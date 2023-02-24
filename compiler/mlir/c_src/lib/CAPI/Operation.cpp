#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "CIR-c/Operation.h"

MlirStringRef mlirOperationGetDialectName(MlirOperation op) {
  return wrap(unwrap(op)->getDialect()->getNamespace());
}
