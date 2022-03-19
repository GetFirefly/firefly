#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Pass/PassRegistry.h"

#include "CIR-c/Passes.h"
#include "CIR/Passes.h"

using namespace mlir;
using namespace mlir::cir;

#include "CIR/Passes.capi.cpp.inc"
