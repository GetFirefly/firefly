#ifndef EIR_TYPES_H
#define EIR_TYPES_H

#include <vector>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIREnums.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

using ::llvm::ArrayRef;
using ::llvm::Optional;
using ::llvm::TypeSwitch;
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

#define EIR_TERM_KIND(Name, Val) class Name##Type;
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/EIR/IR/EIREncoding.h.inc"
#undef EIR_TERM_KIND
#undef FIRST_EIR_TERM_KIND
class RefType;
class PtrType;
class ReceiveRefType;

namespace detail {
struct TupleTypeStorage;
struct BoxTypeStorage;
struct RefTypeStorage;
struct PtrTypeStorage;
struct ReceiveRefTypeStorage;
}  // namespace detail

namespace TypeKind {
enum Kind {
#define EIR_TERM_KIND(Name, Val) Name = Val,
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/EIR/IR/EIREncoding.h.inc"
#undef EIR_TERM_KIND
#undef FIRST_EIR_TERM_KIND
  Ref = 19,
  Ptr = 20,
  ReceiveRef = 21,
};
}  // namespace TypeKind

class OpaqueTermType : public Type {
 public:
  using Type::Type;

  // Returns the raw integer value of the TypeKind::Kind variant
  // that matches the equivalent variant in Rust, given an enum
  // definition that is a 1:1 match of TypeKind::Kind
  Optional<unsigned> getTypeKind() const {
    return TypeSwitch<Type, Optional<unsigned>>(*this)
#define EIR_TERM_KIND(Name, Val) .Case<Name##Type>([&](Type) { return Val; })
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/EIR/IR/EIREncoding.h.inc"
#undef EIR_TERM_KIND
#undef FIRST_EIR_TERM_KIND
        .Default([](Type) { return llvm::None; });
  }

  bool hasDynamicExtent() const {
    return TypeSwitch<Type, bool>(*this)
        .Case<TupleType>([&](Type) { return true; })
        .Case<BinaryType>([&](Type) { return true; })
        .Case<HeapBinType>([&](Type) { return true; })
        .Case<ProcBinType>([&](Type) { return true; })
        .Case<ClosureType>([&](Type) { return true; })
        .Default([](Type) { return false; });
  }

  bool isImmediate() const {
    return TypeSwitch<Type, bool>(*this)
        .Case<AtomType>([&](Type) { return true; })
        .Case<BooleanType>([&](Type) { return true; })
        .Case<FixnumType>([&](Type) { return true; })
        .Case<FloatType>([&](Type) { return true; })
        .Case<NilType>([&](Type) { return true; })
        .Default([](Type) { return false; });
  }

  bool isBoxable() const {
    return TypeSwitch<Type, bool>(*this)
        .Case<FloatType>([&](Type) { return true; })
        .Case<BigIntType>([&](Type) { return true; })
        .Case<ConsType>([&](Type) { return true; })
        .Case<TupleType>([&](Type) { return true; })
        .Case<MapType>([&](Type) { return true; })
        .Case<ClosureType>([&](Type) { return true; })
        .Case<BinaryType>([&](Type) { return true; })
        .Case<HeapBinType>([&](Type) { return true; })
        .Case<ProcBinType>([&](Type) { return true; })
        .Default([](Type) { return false; });
  }

  bool isOpaque() { return isa<TermType>(); }

  bool isAtom() const { return isa<AtomType>() || isBoolean(); }

  bool isBoolean() const { return isa<BooleanType>(); }

  bool isNumber() const {
    return TypeSwitch<Type, bool>(*this)
        .Case<NumberType>([&](Type) { return true; })
        .Case<IntegerType>([&](Type) { return true; })
        .Case<FixnumType>([&](Type) { return true; })
        .Case<BigIntType>([&](Type) { return true; })
        .Case<FloatType>([&](Type) { return true; })
        .Default([](Type) { return false; });
  }

  bool isFixnum() const { return isa<FixnumType>(); }

  bool isInteger() const {
    return TypeSwitch<Type, bool>(*this)
        .Case<IntegerType>([&](Type) { return true; })
        .Case<FixnumType>([&](Type) { return true; })
        .Case<BigIntType>([&](Type) { return true; })
        .Default([](Type) { return false; });
  }

  bool isFloat() const { return isa<FloatType>(); }

  bool isCons() const { return isa<ConsType>(); }

  bool isList() const { return isa<ListType>() || isCons() || isNil(); }

  bool isNil() const { return isa<NilType>(); }

  bool isNonEmptyList() const { return isCons() || isList(); }

  bool isTuple() const { return isa<TupleType>(); }

  bool isBinary() const {
    return isa<BinaryType>() || isHeapBin() || isProcBin();
  }

  bool isHeapBin() const { return isa<HeapBinType>(); }

  bool isProcBin() const { return isa<ProcBinType>(); }

  bool isClosure() const { return isa<ClosureType>(); }

  bool isBox() const { return isa<BoxType>(); }

  // Returns 0 for false, 1 for true, 2 for unknown
  unsigned isMatch(Type matcher) {
    auto matcherBase = matcher.dyn_cast_or_null<OpaqueTermType>();
    if (!matcherBase) return 2;

    auto typeId = getTypeID();
    auto matcherTypeId = matcher.getTypeID();

    // Unresolvable statically
    if (!isOpaque() || !matcherBase.isOpaque()) return 2;

    // Guaranteed to match
    if (typeId == matcherTypeId) return 1;

    // Generic matches
    if (matcher.isa<AtomType>()) return isAtom() ? 1 : 0;
    if (matcher.isa<ListType>()) return isList() ? 1 : 0;
    if (matcher.isa<NumberType>()) return isNumber() ? 1 : 0;
    if (matcher.isa<IntegerType>()) return isInteger() ? 1 : 0;
    if (matcher.isa<BinaryType>()) return isBinary() ? 1 : 0;

    return 0;
  }

  static bool classof(Type type) {
    if (!llvm::isa<eirDialect>(type.getDialect())) return false;
    // ReceiveRefs are not actually term types
    if (type.isa<ReceiveRefType>()) return false;
    return true;
  }

  Optional<int64_t> getSizeInBytes() {
    if (isImmediate() || isBox()) {
      return 8;
    }
    return llvm::None;
  }
};

#define PrimitiveType(TYPE, KIND)                                              \
  class TYPE                                                                   \
      : public mlir::Type::TypeBase<TYPE, OpaqueTermType, mlir::TypeStorage> { \
   public:                                                                     \
    using Base::Base;                                                          \
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

  static TupleType get(MLIRContext *context);
  static TupleType get(MLIRContext *context, unsigned arity);
  static TupleType get(MLIRContext *context, unsigned arity, Type elementType);
  static TupleType get(MLIRContext *context, ArrayRef<Type> elementTypes);
  static TupleType get(unsigned arity, Type elementType);
  static TupleType get(ArrayRef<Type> elementTypes);

  // Verifies construction invariants and issues errors/warnings.
  static LogicalResult verifyConstructionInvariants(
      Location loc, ArrayRef<Type> elementTypes);

  // Returns the size of the shaped type
  size_t getArity() const;
  // Get the size in bytes needed to represent the tuple in memory
  size_t getSizeInBytes() const;
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

  OpaqueTermType getBoxedType() const;
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
};

/// Used to represent the raw stacktrace capture used in exception handling
class TraceRefType
    : public Type::TypeBase<TraceRefType, Type, mlir::TypeStorage> {
 public:
  using Base::Base;

  static TraceRefType get(mlir::MLIRContext *context);
};

/// Used to represent the opaque handle for a receive construct
class ReceiveRefType
    : public Type::TypeBase<ReceiveRefType, Type, mlir::TypeStorage> {
 public:
  using Base::Base;

  static ReceiveRefType get(mlir::MLIRContext *context);
};

template <typename A, typename B>
bool inbounds(A v, B lb, B ub) {
  return v >= lb && v <= ub;
}

}  // namespace eir
}  // namespace lumen

#endif  // EIR_TYPES_H
