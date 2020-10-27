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
using ::mlir::TypeRange;
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
class TraceRefType;

namespace detail {
struct TupleTypeStorage;
struct ClosureTypeKey;
struct ClosureTypeStorage;
struct BoxTypeStorage;
struct RefTypeStorage;
struct PtrTypeStorage;
}  // namespace detail

namespace TypeKind {
enum Kind {
#define EIR_TERM_KIND(Name, Val) Name = Val,
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/EIR/IR/EIREncoding.h.inc"
#undef EIR_TERM_KIND
#undef FIRST_EIR_TERM_KIND
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
        .Case<PidType>([&](Type) { return true; })
        .Case<BooleanType>([&](Type) { return true; })
        .Case<FixnumType>([&](Type) { return true; })
        .Case<FloatType>([&](Type) { return true; })
        .Case<NilType>([&](Type) { return true; })
        .Default([](Type) { return false; });
  }

  // NOTE: We default to 32 bits so that floats are boxable by default
  bool isBoxable(uint8_t pointerSizeInBits = 32) const {
    if (pointerSizeInBits == 64) {
      return TypeSwitch<Type, bool>(*this)
          .Case<PidType>([&](Type) { return true; })
          .Case<ReferenceType>([&](Type) { return true; })
          .Case<BigIntType>([&](Type) { return true; })
          .Case<ConsType>([&](Type) { return true; })
          .Case<TupleType>([&](Type) { return true; })
          .Case<MapType>([&](Type) { return true; })
          .Case<ClosureType>([&](Type) { return true; })
          .Case<BinaryType>([&](Type) { return true; })
          .Case<HeapBinType>([&](Type) { return true; })
          .Case<ProcBinType>([&](Type) { return true; })
          .Default([](Type) { return false; });
    } else {
      return TypeSwitch<Type, bool>(*this)
          .Case<PidType>([&](Type) { return true; })
          .Case<ReferenceType>([&](Type) { return true; })
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
  }

  bool isOpaque() { return isa<TermType>(); }

  bool isAtom() const { return isa<AtomType>() || isBoolean(); }

  bool isBoolean() const { return isa<BooleanType>(); }

  bool isPid() const { return isa<PidType>(); }

  bool isReference() const { return isa<ReferenceType>(); }

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

  bool isMap() const { return isa<MapType>(); }

  bool isBinary() const {
    return isa<BinaryType>() || isHeapBin() || isProcBin();
  }

  bool isHeapBin() const { return isa<HeapBinType>(); }

  bool isProcBin() const { return isa<ProcBinType>(); }

  bool isClosure() const { return isa<ClosureType>(); }

  bool isBox() const { return isa<BoxType>(); }

  // Returns 0 for false, 1 for true, 2 for unknown
  unsigned isMatch(Type matcher);

  bool canTypeEverBeEqual(Type other);

  static bool classof(Type type) {
    if (!llvm::isa<eirDialect>(type.getDialect())) return false;
    // The following are not actually term types
    if (type.isa<PtrType>()) return false;
    if (type.isa<RefType>()) return false;
    if (type.isa<ReceiveRefType>()) return false;
    if (type.isa<TraceRefType>()) return false;
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
PrimitiveType(BinaryType, TypeKind::Binary);
PrimitiveType(HeapBinType, TypeKind::HeapBin);
PrimitiveType(ProcBinType, TypeKind::ProcBin);
PrimitiveType(PidType, TypeKind::Pid);
PrimitiveType(ReferenceType, TypeKind::Reference);

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

/// A closure with a potentially dynamically-sized environment
class ClosureType : public Type::TypeBase<ClosureType, OpaqueTermType,
                                          detail::ClosureTypeStorage> {
 public:
  using Base::Base;

  static ClosureType get(MLIRContext *context);
  static ClosureType get(MLIRContext *context, size_t envLen);
  static ClosureType get(MLIRContext *context, TypeRange env);
  static ClosureType get(MLIRContext *context, FunctionType functionType);
  static ClosureType get(MLIRContext *context, FunctionType functionType,
                         size_t envLen);
  static ClosureType get(MLIRContext *context, FunctionType functionType,
                         TypeRange env);

  // Verifies construction invariants and issues errors/warnings.
  static LogicalResult verifyConstructionInvariants(
      Location loc, const detail::ClosureTypeKey &key);

  // Gets the type of the underlying function
  Optional<FunctionType> getCalleeType() const;
  // Returns the arity of the underlying function
  Optional<size_t> getArity() const;
  // Returns the size of the closure environment
  size_t getEnvLen() const;
  // Returns true if the dimensions of the closure environment are known
  bool hasStaticShape() const;
  // Returns true if the dimensions of the closure environment are unknown
  bool hasDynamicShape() const;
  // Returns the element type for the given element
  Type getEnvType(unsigned index) const;
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
  static PtrType get(MLIRContext *context, Type innerType);
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
