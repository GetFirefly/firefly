#ifndef EIR_TYPES_H
#define EIR_TYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIREnums.h"

#include <stdint.h>

using ::mlir::FunctionType;
using ::mlir::Type;
using ::mlir::TypeRange;

/// This enumeration represents the mapping between MLIR types and kinds
/// understood by the runtime for encoding/decoding.
namespace lumen {
namespace eir {
namespace TypeKind {
enum Kind : uint32_t {
#define EIR_TERM_KIND(Name, Val) Name = Val,
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/EIR/IR/EIREncoding.h.inc"
#undef EIR_TERM_KIND
#undef FIRST_EIR_TERM_KIND
};
}  // namespace TypeKind
}  // namespace eir
}  // namespace lumen

namespace lumen {
namespace eir {
//===----------------------------------------------------------------------===//
// Type Traits
//===----------------------------------------------------------------------===//

template <typename ConcreteType>
class NumberLike
    : public ::mlir::TypeTrait::TraitBase<ConcreteType, NumberLike> {};

template <typename ConcreteType>
class AtomLike : public ::mlir::TypeTrait::TraitBase<ConcreteType, AtomLike> {};

template <typename ConcreteType>
class ListLike : public ::mlir::TypeTrait::TraitBase<ConcreteType, ListLike> {};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Returns a value indicating whether the first type matches the type
// constraint expressed by the second type.
//
// Returns 0 for false, 1 for true, 2 for unknown
static unsigned isMatch(Type termType, Type matcher);

// Returns true if values of the two types can potentially compare as equal
// (strictness is determined by `strict` flag)
static bool canTypeEverBeEqual(Type a, Type b, bool strict);

}  // namespace eir
}  // namespace lumen

//===----------------------------------------------------------------------===//
// Tablegen Type and Type Interface Declarations
//===----------------------------------------------------------------------===//

#include "lumen/EIR/IR/EIRTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "lumen/EIR/IR/EIRTypes.h.inc"

#endif  // EIR_TYPES_H
