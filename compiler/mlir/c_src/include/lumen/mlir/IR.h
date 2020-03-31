#ifndef LUMEN_MLIR_IR_H
#define LUMEN_MLIR_IR_H

#include "lumen/mlir/MLIR.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/CBindingWrapping.h"

namespace mlir {
class Builder;
class Location;
class FuncOp;
}  // namespace mlir

using ::mlir::Block;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;
using ::mlir::Attribute;
using ::mlir::Location;
using ::mlir::Region;

typedef struct MLIROpaqueBlock *MLIRBlockRef;
typedef struct MLIROpaqueValue *MLIRValueRef;
typedef struct MLIROpaqueLocation *MLIRLocationRef;
typedef struct MLIROpaqueAttribute *MLIRAttributeRef;
typedef struct MLIROpaqueBuilder *MLIRBuilderRef;
typedef struct MLIROpaqueFuncOp *MLIRFunctionOpRef;
typedef struct MLIROpaqueAttribute *MLIRAttributeRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::Builder, MLIRBuilderRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::Block, MLIRBlockRef);

inline Attribute unwrap(const void *P) {
  return Attribute::getFromOpaquePointer(P);
}

inline MLIRAttributeRef wrap(const Attribute &attr) {
  auto ptr = attr.getAsOpaquePointer();
  return reinterpret_cast<MLIRAttributeRef>(const_cast<void *>(ptr));
}

inline Value unwrap(MLIRValueRef v) { return Value::getFromOpaquePointer(v); }

inline MLIRValueRef wrap(const Value &val) {
  auto ptr = val.getAsOpaquePointer();
  return reinterpret_cast<MLIRValueRef>(const_cast<void *>(ptr));
}

inline Location unwrap(MLIRLocationRef l) {
  return Location::getFromOpaquePointer(l);
}

inline MLIRLocationRef wrap(const Location &loc) {
  auto ptr = loc.getAsOpaquePointer();
  return reinterpret_cast<MLIRLocationRef>(const_cast<void *>(ptr));
}

#endif
