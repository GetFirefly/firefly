#ifndef EIR_DIALECT_H
#define EIR_DIALECT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dialect.h"

namespace lumen {
namespace eir {

#include "lumen/EIR/IR/EIRDialect.h.inc"

}  // namespace eir
}  // namespace lumen

#endif  // EIR_DIALECT_H
