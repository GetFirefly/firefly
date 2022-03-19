#pragma once

#include "CIR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "mlir/IR/Types.h"

#include "lumen/term/Encoding.h"

namespace mlir {
// class AsmPrinter;
// class DialectAsmParser;

namespace cir {
class CIRDialect;

namespace detail {
struct CIRFunTypeStorage;
} // namespace detail
} // namespace cir
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen
//===----------------------------------------------------------------------===//
#include "CIR/CIRTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "CIR/CIRTypes.h.inc"

namespace mlir {
namespace cir {
//===----------------------------------------------------------------------===//
// Printing and parsing.
//===----------------------------------------------------------------------===//

// namespace detail {
/// Parses a CIR dialect type.
// Type parseType(DialectAsmParser &parser);

/// Prints a CIR dialect type.
// void printType(Type type, AsmPrinter &printer);
//} // namespace detail

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Returns `true` if the given type is a CIR dialect type
bool isCIRType(Type type);

/// Returns `true` if the given type is a CIR dialect type and representable as
/// a term
bool isTermType(Type type);

/// Returns `true` if the given type is a pointer-sized value
bool isImmediateType(Type type);

/// Returns `true` if the given type is a valid pointee type for CIRBoxType
bool isBoxableType(Type type);

// Returns true if the given type is representable as a primitive numeric
// immediate
//
// NOTE: This returns false for bigints
bool isTypePrimitiveNumeric(Type ty);

} // namespace cir
} // namespace mlir
