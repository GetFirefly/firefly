#ifndef EIR_TYPES_H
#define EIR_TYPES_H

#include <vector>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "lumen/compiler/Dialect/EIR/IR/EIREnums.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

using ::llvm::ArrayRef;
using ::llvm::Optional;
using ::mlir::failure;
using ::mlir::FunctionType;
using ::mlir::Location;
using ::mlir::LogicalResult;
using ::mlir::MLIRContext;
using ::mlir::success;
using ::mlir::Type;
using ::mlir::TypeStorage;

/// This enumeration represents all of the types defined by the EIR dialect
namespace lumen {
namespace eir {

namespace detail {
struct OpaqueTermTypeStorage;
struct TupleTypeStorage;
struct BoxTypeStorage;
struct RefTypeStorage;
struct PtrTypeStorage;
struct ReceiveRefTypeStorage;
}  // namespace detail

namespace TypeKind {
enum Kind {
#define EIR_TERM_KIND(Name, Val) Name = mlir::Type::FIRST_EIR_TYPE + Val,
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/compiler/Dialect/EIR/IR/EIREncoding.h.inc"
#undef EIR_TERM_KIND
#undef FIRST_EIR_TERM_KIND
  Ref = mlir::Type::FIRST_EIR_TYPE + 19,
  Ptr = mlir::Type::FIRST_EIR_TYPE + 20,
  ReceiveRef = mlir::Type::LAST_EIR_TYPE,
};
}  // namespace TypeKind

class OpaqueTermType : public Type {
 public:
  using ImplType = detail::OpaqueTermTypeStorage;
  using Type::Type;

  unsigned getImplKind() const { return getKind(); }

  // Returns the raw integer value of the TypeKind::Kind variant
  // that matches the equivalent variant in Rust, given an enum
  // definition that is a 1:1 match of TypeKind::Kind
  unsigned getForeignKind() const {
    unsigned offset = mlir::Type::FIRST_EIR_TYPE;
    return getKind() - offset;
  }

  bool isKind(unsigned kind) const { return kind == getImplKind(); }

  bool isOpaque() const { return isOpaque(getImplKind()); }

  bool hasDynamicExtent() const { return hasDynamicExtent(getImplKind()); }

  bool isImmediate() const { return isImmediate(getImplKind()); }

  bool isBoxable() const { return isBoxable(getImplKind()); }

  bool isAtom() const { return isAtom(getImplKind()); }

  bool isBoolean() const { return isBoolean(getImplKind()); }

  bool isNumber() const { return isNumber(getImplKind()); }

  bool isFixnum() const { return isFixnum(getImplKind()); }

  bool isInteger() const { return isInteger(getImplKind()); }

  bool isFloat() const { return isFloat(getImplKind()); }

  bool isList() const { return isList(getImplKind()); }

  bool isNil() const { return isNil(getImplKind()); }

  bool isNonEmptyList() const { return isNonEmptyList(getImplKind()); }

  bool isTuple() const { return isTuple(getImplKind()); }

  bool isBinary() const { return isBinary(getImplKind()); }

  bool isClosure() const { return isClosure(getImplKind()); }

  bool isBox() const { return isBox(getImplKind()); }

  // Returns 0 for false, 1 for true, 2 for unknown
  unsigned isMatch(Type matcher) const {
    auto matcherBase = matcher.dyn_cast_or_null<OpaqueTermType>();
    if (!matcherBase) return 2;

    auto implKind = getImplKind();
    auto matcherImplKind = matcherBase.getImplKind();

    // Unresolvable statically
    if (!isOpaque(implKind) || !matcherBase.isOpaque(matcherImplKind)) return 2;

    // Guaranteed to match
    if (implKind == matcherImplKind) return 1;

    // Generic matches
    if (matcherImplKind == TypeKind::Atom) return isAtom(implKind) ? 1 : 0;
    if (matcherImplKind == TypeKind::List) return isList(implKind) ? 1 : 0;
    if (matcherImplKind == TypeKind::Number) return isNumber(implKind) ? 1 : 0;
    if (matcherImplKind == TypeKind::Integer)
      return isInteger(implKind) ? 1 : 0;
    if (matcherImplKind == TypeKind::Float) return isFloat(implKind) ? 1 : 0;

    return 0;
  }

  static bool isTypeKind(Type type, TypeKind::Kind kind) {
    if (!OpaqueTermType::classof(type)) {
      return false;
    }
    return type.cast<OpaqueTermType>().getImplKind() == kind;
  }

  static bool classof(Type type) {
    auto kind = type.getKind();
    // ReceiveRefs are not actually term types
    if (kind == TypeKind::ReceiveRef) return false;
#define EIR_TERM_KIND(Name, Val)                    \
  if (kind == (mlir::Type::FIRST_EIR_TYPE + Val)) { \
    return true;                                    \
  }
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/compiler/Dialect/EIR/IR/EIREncoding.h.inc"
#undef EIR_TERM_KIND
#undef FIRST_EIR_TERM_KIND
    return false;
  }

 private:
  static bool isOpaque(unsigned implKind) { return implKind == TypeKind::Term; }

  static bool hasDynamicExtent(unsigned implKind) {
    return implKind == TypeKind::Tuple || implKind == TypeKind::Binary ||
           implKind == TypeKind::HeapBin || implKind == TypeKind::ProcBin ||
           implKind == TypeKind::Closure;
  }

  static bool isImmediate(unsigned implKind) {
    return implKind == TypeKind::Atom || implKind == TypeKind::Boolean ||
           implKind == TypeKind::Fixnum || implKind == TypeKind::Float ||
           implKind == TypeKind::Nil || implKind == TypeKind::Box ||
           implKind == TypeKind::Term;
  }

  static bool isBoxable(unsigned implKind) {
    return implKind == TypeKind::Float || implKind == TypeKind::BigInt ||
           implKind == TypeKind::Cons || implKind == TypeKind::Tuple ||
           implKind == TypeKind::Map || implKind == TypeKind::Closure ||
           implKind == TypeKind::Binary || implKind == TypeKind::HeapBin ||
           implKind == TypeKind::ProcBin;
  }

  static bool isAtom(unsigned implKind) {
    return implKind >= TypeKind::Atom && implKind <= TypeKind::Boolean;
  }

  static bool isBoolean(unsigned implKind) {
    return implKind == TypeKind::Boolean;
  }

  static bool isNumber(unsigned implKind) {
    return implKind == TypeKind::Number || implKind == TypeKind::Integer ||
           implKind == TypeKind::Fixnum || implKind == TypeKind::BigInt ||
           implKind == TypeKind::Float;
  }

  static bool isFixnum(unsigned implKind) {
    return implKind == TypeKind::Fixnum;
  }

  static bool isInteger(unsigned implKind) {
    return implKind == TypeKind::Integer || implKind == TypeKind::Fixnum ||
           implKind == TypeKind::BigInt;
  }

  static bool isFloat(unsigned implKind) { return implKind == TypeKind::Float; }

  static bool isList(unsigned implKind) {
    return implKind == TypeKind::List || implKind == TypeKind::Nil ||
           implKind == TypeKind::Cons;
  }

  static bool isNil(unsigned implKind) { return implKind == TypeKind::Nil; }

  static bool isNonEmptyList(unsigned implKind) {
    return implKind == TypeKind::Cons;
  }

  static bool isTuple(unsigned implKind) { return implKind == TypeKind::Tuple; }

  static bool isBinary(unsigned implKind) {
    return implKind == TypeKind::Binary || implKind == TypeKind::HeapBin ||
           implKind == TypeKind::ProcBin;
  }

  static bool isClosure(unsigned implKind) {
    return implKind == TypeKind::Closure;
  }

  static bool isBox(unsigned implKind) { return implKind == TypeKind::Box; }
};

#define PrimitiveType(TYPE, KIND)                                  \
  class TYPE : public mlir::Type::TypeBase<TYPE, OpaqueTermType> { \
   public:                                                         \
    using Base::Base;                                              \
    static TYPE get(mlir::MLIRContext *context) {                  \
      return Base::get(context, KIND);                             \
    }                                                              \
    static bool kindof(unsigned kind) { return kind == KIND; }     \
  }

PrimitiveType(NoneType, TypeKind::None);
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
class TupleType : public Type::TypeBase<TupleType, OpaqueTermType,
                                        detail::TupleTypeStorage> {
 public:
  using Base::Base;

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == TypeKind::Tuple; }

  static TupleType get(MLIRContext *context);
  static TupleType get(MLIRContext *context, unsigned arity);
  static TupleType get(MLIRContext *context, unsigned arity, Type elementType);
  static TupleType get(MLIRContext *context, ArrayRef<Type> elementTypes);
  static TupleType get(unsigned arity, Type elementType);
  static TupleType get(ArrayRef<Type> elementTypes);

  // Verifies construction invariants and issues errors/warnings.
  static LogicalResult verifyConstructionInvariants(
      Location loc, unsigned arity, ArrayRef<Type> elementTypes);

  // Returns the size of the shaped type
  int64_t getArity() const;
  // Get the size in bytes needed to represent the tuple in memory
  int64_t getSizeInBytes() const;
  // Returns true if the dimensions of the tuple are known
  bool hasStaticShape() const;
  // Returns true if the dimensions of the tuple are unknown
  bool hasDynamicShape() const;
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
  static LogicalResult verifyConstructionInvariants(Location loc,
                                                    Type boxedType) {
    if (!OpaqueTermType::classof(boxedType)) {
      emitError(loc) << "invalid target type for a box: " << boxedType;
      return failure();
    }
    return success();
  }

  OpaqueTermType getBoxedType() const;

  static bool kindof(unsigned kind) { return kind == TypeKind::Box; }
};

/// A pointer to a term
class RefType : public Type::TypeBase<RefType, Type, detail::RefTypeStorage> {
 public:
  using Base::Base;

  /// Gets or creates a RefType with the provided target object type.
  static RefType get(OpaqueTermType innerType);
  static RefType get(MLIRContext *context, OpaqueTermType innerType);

  /// Gets or creates a RefType with the provided target object type.
  /// This emits an error at the specified location and returns null if the
  /// object type isn't supported.
  static RefType getChecked(Type innerType, mlir::Location location);

  OpaqueTermType getInnerType() const;

  static bool kindof(unsigned kind) { return kind == TypeKind::Ref; }
};

/// A raw pointer
class PtrType : public Type::TypeBase<PtrType, Type, detail::PtrTypeStorage> {
 public:
  using Base::Base;

  /// Gets or creates a PtrType with the provided target object type.
  static PtrType get(Type innerType);
  /// Gets or creates a PtrType with a default type of i8
  static PtrType get(MLIRContext *context);

  Type getInnerType() const;

  static bool kindof(unsigned kind) { return kind == TypeKind::Ptr; }
};

/// Used to represent the opaque handle for a receive construct
class ReceiveRefType : public Type::TypeBase<ReceiveRefType, Type> {
 public:
  using Base::Base;

  static ReceiveRefType get(mlir::MLIRContext *context);

  static bool kindof(unsigned kind) { return kind == TypeKind::ReceiveRef; }
};

template <typename A, typename B>
bool inbounds(A v, B lb, B ub) {
  return v >= lb && v <= ub;
}

}  // namespace eir
}  // namespace lumen

#endif  // EIR_TYPES_H
