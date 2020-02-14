#ifndef EIR_IR_OPINTERFACE_H
#define EIR_IR_OPINTERFACE_H

#include "mlir/Analysis/CallInterfaces.h"
#include "mlir/IR/OpImplementation.h"

using ::mlir::CallInterfaceCallable;
using ::mlir::Operation;
using ::mlir::OpInterface;
using ::mlir::OpAsmSetValueNameFn;
using ::mlir::OpAsmSetValueNameFn;
using ::mlir::Region;
using ::mlir::Block;
using ::mlir::Type;
using ::mlir::Value;
using ::llvm::ArrayRef;
using ::llvm::StringRef;

namespace lumen {
namespace eir {
#include "lumen/compiler/Dialect/EIR/IR/EIROpInterface.h.inc"
}  // namespace eir
}  // namespace lumen

#endif
