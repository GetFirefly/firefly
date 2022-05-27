#pragma once

#include "CIR-c/AtomRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlir {
class IntegerAttr;
namespace cir {
class CIRAtomType;

enum class Endianness { Big, Little, Native };

inline static uint32_t wrap(Endianness e) {
  switch (e) {
  case Endianness::Big:
    return 0;
  case Endianness::Little:
    return 1;
  default:
    return 2;
  }
}

inline static Endianness unwrap(uint32_t e) {
  switch (e) {
  case 0:
    return Endianness::Big;
  case 1:
    return Endianness::Little;
  default:
    return Endianness::Native;
  }
}

//===----------------------------------------------------------------------===//
// Atom
//===----------------------------------------------------------------------===//
llvm::hash_code hash_value(const AtomRef &atom);

} // namespace cir
} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen
//===----------------------------------------------------------------------===//

#include "CIR/CIREnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "CIR/CIRAttributes.h.inc"
