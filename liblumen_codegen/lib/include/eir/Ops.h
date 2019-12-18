#ifndef EIR_OPS_H_
#define EIR_OPS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

using llvm::ArrayRef;
using llvm::StringRef;

namespace eir {

/// All operations are declared in this auto-generated header
#define GET_OP_CLASSES
#include "eir/Ops.h.inc"

} // namespace eir

#endif // EIR_OPS_H_
