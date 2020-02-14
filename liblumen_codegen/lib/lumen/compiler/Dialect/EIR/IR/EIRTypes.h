#ifndef EIR_TYPES_H
#define EIR_TYPES_H

#include "lumen/compiler/Dialect/EIR/IR/EIREnums.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include <vector>

using ::mlir::Type;
using ::mlir::FunctionType;
using ::mlir::TypeStorage;
using ::mlir::MLIRContext;
using ::mlir::LogicalResult;
using ::mlir::Location;
using ::mlir::success;
using ::mlir::failure;
using ::llvm::ArrayRef;
using ::llvm::Optional;

/// This enumeration represents all of the types defined by the EIR dialect
namespace lumen {
namespace eir {

namespace detail {
struct OpaqueTermTypeStorage;
struct TupleTypeStorage;
struct BoxTypeStorage;
} // namespace detail

namespace TypeKind {
enum Kind {
#define EIR_TERM_KIND(Name, Val) Name,
#define FIRST_EIR_TERM_KIND(Name, Val) Name = mlir::Type::FIRST_EIR_TYPE,
#include "lumen/compiler/Dialect/EIR/IR/EIREncoding.h.inc"
};
} // namespace TypeKind

class OpaqueTermType : public Type {
public:
  using ImplType = detail::OpaqueTermTypeStorage;
  using Type::Type;

  unsigned getImplKind() const { return implKind; }

  bool isKind(unsigned kind) const { return kind == getImplKind(); }

  bool isOpaque() const { return isOpaque(getImplKind()); }

  bool isImmediate() const { return isImmediate(getImplKind()); }

  bool isBoxable() const { return isBoxable(getImplKind()); }

  bool isAtom() const { return isAtom(getImplKind()); }

  bool isNumber() const { return isNumber(getImplKind()); }

  bool isInteger() const { return isInteger(getImplKind()); }

  bool isFloat() const { return isFloat(getImplKind()); }

  bool isList() const { return isList(getImplKind()); }

  bool isNil() const { return isNil(getImplKind()); }

  bool isNonEmptyList() const { return isNonEmptyList(getImplKind()); }

  bool isBinary() const { return isBinary(getImplKind()); }

  bool isBox() const { return isBox(getImplKind()); }

  // Returns 0 for false, 1 for true, 2 for unknown
  unsigned isMatch(Type matcher) const {
    auto matcherBase = matcher.dyn_cast_or_null<OpaqueTermType>();
    if (!matcherBase)
      return 2;

    auto implKind = getImplKind();
    auto matcherImplKind = matcherBase.getImplKind();

    // Unresolvable statically
    if (!isOpaque(implKind) || !matcherBase.isOpaque(matcherImplKind))
      return 2;

    // Guaranteed to match
    if (implKind == matcherImplKind)
      return 1;

    // Generic matches
    if (matcherImplKind == TypeKind::Atom)
      return isAtom(implKind) ? 1 : 0;
    if (matcherImplKind == TypeKind::List)
      return isList(implKind) ? 1 : 0;
    if (matcherImplKind == TypeKind::Number)
      return isNumber(implKind) ? 1 : 0;
    if (matcherImplKind == TypeKind::Integer)
      return isInteger(implKind) ? 1 : 0;
    if (matcherImplKind == TypeKind::Float)
      return isFloat(implKind) ? 1 : 0;

    return 0;
  }

  static bool classof(Type type) {
    auto kind = type.getKind();
    return kind >= TypeKind::Term && kind <= TypeKind::Box;
  }

private:
  unsigned implKind;

  static bool isOpaque(unsigned implKind) {
    return implKind == TypeKind::Term;
  }

  static bool isImmediate(unsigned implKind) {
    return implKind == TypeKind::Atom ||
      implKind == TypeKind::Boolean ||
      implKind == TypeKind::Fixnum ||
      implKind == TypeKind::Float ||
      implKind == TypeKind::Nil ||
      implKind == TypeKind::Box;
  }

  static bool isBoxable(unsigned implKind) {
    return implKind == TypeKind::Float ||
      implKind == TypeKind::BigInt ||
      implKind == TypeKind::Cons ||
      implKind == TypeKind::Tuple ||
      implKind == TypeKind::Map ||
      implKind == TypeKind::Closure ||
      implKind == TypeKind::Binary ||
      implKind == TypeKind::HeapBin ||
      implKind == TypeKind::ProcBin;
  }

  static bool isAtom(unsigned implKind) {
    return implKind >= TypeKind::Atom && implKind <= TypeKind::Boolean;
  }

  static bool isNumber(unsigned implKind) {
    return implKind == TypeKind::Number ||
      implKind == TypeKind::Integer ||
      implKind == TypeKind::Fixnum ||
      implKind == TypeKind::BigInt ||
      implKind == TypeKind::Float;
  }

  static bool isInteger(unsigned implKind) {
    return implKind == TypeKind::Integer ||
      implKind == TypeKind::Fixnum ||
      implKind == TypeKind::BigInt;
  }

  static bool isFloat(unsigned implKind) {
    return implKind == TypeKind::Float;
  }

  static bool isList(unsigned implKind) {
    return implKind == TypeKind::List ||
      implKind == TypeKind::Nil ||
      implKind == TypeKind::Cons;
  }

  static bool isNil(unsigned implKind) {
    return implKind == TypeKind::Nil;
  }

  static bool isNonEmptyList(unsigned implKind) {
    return implKind == TypeKind::Cons;
  }

  static bool isBinary(unsigned implKind) {
    return implKind == TypeKind::Binary ||
      implKind == TypeKind::HeapBin ||
      implKind == TypeKind::ProcBin;
  }

  static bool isBox(unsigned implKind) {
    return implKind == TypeKind::Box;
  }
};

#define PrimitiveType(TYPE, KIND)                                              \
  class TYPE : public mlir::Type::TypeBase<TYPE, OpaqueTermType> {             \
  public:                                                                      \
    using Base::Base;                                                          \
    static TYPE get(mlir::MLIRContext *context) {                              \
      return Base::get(context, KIND);                                         \
    }                                                                          \
    static bool kindof(unsigned kind) { return kind == KIND; }                 \
  }

PrimitiveType(TermType, TypeKind::Term);
PrimitiveType(ListType, TypeKind::List);
PrimitiveType(NumberType, TypeKind::Number);
PrimitiveType(IntegerType, TypeKind::Integer);
PrimitiveType(FloatType, TypeKind::Float);
PrimitiveType(AtomType, TypeKind::Atom);
PrimitiveType(BooleanType, TypeKind::Boolean);
PrimitiveType(FixnumType, TypeKind::Fixnum);
PrimitiveType(BigIntType, TypeKind::BigInt);
PrimitiveType(NilType, TypeKind::Nil);
PrimitiveType(ConsType, TypeKind::Cons);
PrimitiveType(MapType, TypeKind::Map);
PrimitiveType(ClosureType, TypeKind::Closure);
PrimitiveType(BinaryType, TypeKind::Binary);
PrimitiveType(HeapBinType, TypeKind::HeapBin);
PrimitiveType(ProcBinType, TypeKind::ProcBin);

/// A dynamically/statically shaped vector of elements
class TupleType : public Type::TypeBase<TupleType, OpaqueTermType, detail::TupleTypeStorage> {
  public:
    using Base::Base;

    /// Support method to enable LLVM-style type casting.
    static bool kindof(unsigned kind) {
      return kind == TypeKind::Tuple;
    }

    static TupleType get(MLIRContext *context);
    static TupleType get(MLIRContext *context, unsigned arity);
    static TupleType get(MLIRContext *context, unsigned arity, Type elementType);
    static TupleType get(MLIRContext *context, ArrayRef<Type> elementTypes);
    static TupleType get(unsigned arity, Type elementType);
    static TupleType get(ArrayRef<Type> elementTypes);

    // Verifies construction invariants and issues errors/warnings.
    static LogicalResult verifyConstructionInvariants(Optional<Location> loc,
                                                      MLIRContext *context,
                                                      unsigned arity,
                                                      ArrayRef<Type> elementTypes);

    // Returns the size of the shaped type
    int64_t getArity() const;
    // Get the size in bytes needed to represent the tuple in memory
    int64_t getSizeInBytes() const;
    // Returns true if both the size and element types are known
    bool isFullyStatic() const;
    // Returns true if neither the size or element types are known
    bool isFullyDynamic() const;
    // Returns the element type for the given element
    Type getElementType(unsigned index) const;
};

/// A pointer to a heap-allocated term header
class BoxType
    : public Type::TypeBase<BoxType, OpaqueTermType, detail::BoxTypeStorage> {
 public:
  using Base::Base;

  /// Gets or creates a BoxType with the provided target object type.
  static BoxType get(OpaqueTermType boxedType);
  static BoxType get(MLIRContext *context, OpaqueTermType boxedType);

  /// Gets or creates a BoxType with the provided target object type.
  /// This emits an error at the specified location and returns null if the
  /// object type isn't supported.
  static BoxType getChecked(Type boxedType, mlir::Location location);

  /// Verifies construction of a type with the given object.
  static LogicalResult verifyConstructionInvariants(
      llvm::Optional<Location> loc, MLIRContext *context, Type boxedType) {
    if (!OpaqueTermType::classof(boxedType)) {
      if (loc) {
        emitError(*loc) << "invalid target type for a box: " << boxedType;
      }
      return failure();
    }
    return success();
  }

  OpaqueTermType getBoxedType() const;

  static bool kindof(unsigned kind) { return kind == TypeKind::Box; }
};


template <typename A, typename B>
bool inbounds(A v, B lb, B ub) {
  return v >= lb && v < ub;
}

bool isa_eir_type(Type t) {
  return inbounds(t.getKind(),
                  Type::Kind::FIRST_EIR_TYPE,
                  Type::Kind::LAST_EIR_TYPE);
}


bool isa_std_type(Type t) {
  return inbounds(t.getKind(),
                  Type::Kind::FIRST_STANDARD_TYPE,
                  Type::Kind::LAST_STANDARD_TYPE);
}

} // namespace eir
} // namespace lumen

#endif // EIR_TYPES_H
