#include "CIR/Dialect.h"

using namespace mlir;
using namespace mlir::cir;

#include "CIR/CIRDialect.cpp.inc"

void CIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOperations();
  registerInterfaces();
}
