#ifndef EIR_ATTRIBUTES_H
#define EIR_ATTRIBUTES_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"

using ::llvm::ArrayRef;
using ::llvm::StringRef;
using ::mlir::ArrayAttr;
using ::mlir::Attribute;
using ::mlir::BoolAttr;
using ::mlir::FlatSymbolRefAttr;
using ::mlir::IntegerAttr;
using ::mlir::NamedAttribute;
using ::mlir::StringAttr;
using ::mlir::Type;
using ::mlir::TypeAttr;

namespace lumen {
namespace eir {
namespace detail {
struct APIntAttributeStorage;
}
}  // namespace eir
}  // namespace lumen

//===----------------------------------------------------------------------===//
// Tablegen Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "lumen/EIR/IR/EIRAttributes.h.inc"
#include "lumen/EIR/IR/EIRStructs.h.inc"

#endif  // EIR_ATTRIBUTES_H
