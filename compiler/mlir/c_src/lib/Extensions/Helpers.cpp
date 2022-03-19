#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

extern "C" void mlirRegisterCommandLineOptions() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
}

extern "C" MlirStringRef mlirOperationGetDialectName(MlirOperation op) {
  return wrap(unwrap(op)->getDialect()->getNamespace());
}

extern "C" void mlirContextSetPrintOpOnDiagnostic(MlirContext context,
                                                  bool enable) {
  return unwrap(context)->printOpOnDiagnostic(enable);
}

extern "C" void mlirContextSetPrintStackTraceOnDiagnostic(MlirContext context,
                                                          bool enable) {
  return unwrap(context)->printStackTraceOnDiagnostic(enable);
}

extern "C" MlirStringRef mlirModuleGetName(MlirModule module) {
  auto name = unwrap(module).getName();
  return wrap(name.getValue());
}

extern "C" MlirModule mlirModuleClone(MlirModule module) {
  return MlirModule{unwrap(module).getOperation()->clone()};
}

extern "C" MlirTypeID mlirDialectHandleGetTypeID(MlirDialect dialect) {
  return wrap(unwrap(dialect)->getTypeID());
}

extern "C" bool mlirOperationIsAModule(MlirOperation op) {
  return isa<ModuleOp>(unwrap(op));
}
