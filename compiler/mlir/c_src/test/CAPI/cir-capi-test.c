// RUN: cir-capi-test 2>&1 | FileCheck %s

#include <stdio.h>

#include "mlir-c/IR.h"

#include "CIR-c/Dialects.h"

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  // TODO: Create the dialect handles for the builtin dialects and avoid this.
  // This adds dozens of MB of binary size over just the standalone dialect.
  mlirRegisterAllDialects(ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__cir__(), ctx);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString("%0 = arith.constant 2 : i32\n"
                                          "%1 = standalone.foo %0 : i32\n"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = arith.constant 2 : i32
  // CHECK: cir.foo %[[C]] : i32
  mlirOperationDump(op);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
