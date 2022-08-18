#pragma once

#include "CIR-c/AtomRef.h"
#include "CIR-c/BigIntRef.h"
#include "CIR-c/BinaryEntrySpecifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlir {
class IntegerAttr;
namespace cir {
class CIRAtomType;
class CIRBigIntType;

//===----------------------------------------------------------------------===//
// Atom
//===----------------------------------------------------------------------===//
llvm::hash_code hash_value(const AtomRef &atom);

//===----------------------------------------------------------------------===//
// BigInt
//===----------------------------------------------------------------------===//
llvm::hash_code hash_value(const BigIntRef &bigint);

//===----------------------------------------------------------------------===//
// BinaryEntrySpecifier
//===----------------------------------------------------------------------===//
llvm::hash_code hash_value(const BinaryEntrySpecifier &spec);

} // namespace cir
} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen
//===----------------------------------------------------------------------===//

#include "CIR/CIREnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "CIR/CIRAttributes.h.inc"
