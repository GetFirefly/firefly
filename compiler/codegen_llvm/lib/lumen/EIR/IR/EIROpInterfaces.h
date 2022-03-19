#ifndef EIR_IR_OPINTERFACE_H
#define EIR_IR_OPINTERFACE_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"

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

#include "lumen/EIR/IR/EIROpInterfaces.h.inc"

#endif
