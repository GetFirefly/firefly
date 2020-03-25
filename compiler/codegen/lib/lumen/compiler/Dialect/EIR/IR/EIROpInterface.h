#ifndef EIR_IR_OPINTERFACE_H
#define EIR_IR_OPINTERFACE_H

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/IR/OpImplementation.h"

using ::llvm::ArrayRef;
using ::llvm::StringRef;
using ::mlir::Block;
using ::mlir::CallInterfaceCallable;
using ::mlir::OpAsmSetValueNameFn;
using ::mlir::Operation;
using ::mlir::OpInterface;
using ::mlir::Region;
using ::mlir::Type;
using ::mlir::Value;

namespace lumen {
namespace eir {
#include "lumen/compiler/Dialect/EIR/IR/EIROpInterface.h.inc"
}  // namespace eir
}  // namespace lumen

#endif
