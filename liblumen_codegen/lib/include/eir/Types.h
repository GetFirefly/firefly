#ifndef EIR_TYPES_H
#define EIR_TYPES_H

#include "lumen/LLVM.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include <vector>

namespace L = llvm;
namespace M = mlir;

/// This enumeration represents all of the types defined by the EIR dialect
namespace eir {

namespace detail {
struct ShapedTypeStorage;
struct TupleTypeStorage;
struct ClosureTypeStorage;
struct BoxTypeStorage;
struct RefTypeStorage;
} // namespace detail

enum EirTypes {
  Term = M::Type::FIRST_EIR_TYPE,
  Atom,
  Boolean,
  Fixnum,
  BigInt,
  Float,
  FloatPacked,
  Nil,
  Cons,
  Tuple,
  Map,
  Closure,
  HeapBin,
  Box,
  Ref,
};

#define IntrinsicType(TYPE, KIND)                                              \
  class TYPE : public M::Type::TypeBase<TYPE, M::Type> {                       \
  public:                                                                      \
    using Base::Base;                                                          \
    static TYPE get(M::MLIRContext *context) {                                 \
      return Base::get(context, KIND);                                         \
    }                                                                          \
    static bool kindof(unsigned kind) { return kind == KIND; }                 \
  }

IntrinsicType(TermType, EirTypes::Term);
IntrinsicType(AtomType, EirTypes::Atom);
IntrinsicType(BooleanType, EirTypes::Boolean);
IntrinsicType(FixnumType, EirTypes::Fixnum);
IntrinsicType(BigIntType, EirTypes::BigInt);
IntrinsicType(FloatType, EirTypes::Float);
IntrinsicType(PackedFloatType, EirTypes::FloatPacked);
IntrinsicType(NilType, EirTypes::Nil);
IntrinsicType(ConsType, EirTypes::Cons);
IntrinsicType(MapType, EirTypes::Map);
IntrinsicType(HeapBinType, EirTypes::HeapBin);

// Shaped Types, i.e. types that are parameterized and have dynamic extent

using TypeList = std::vector<M::Type>;

struct Shape {
  Shape() : known(false) {}
  Shape(M::Type elementType, unsigned len)
      : known(true), len(len), elementTypes(len, elementType) {}
  Shape(const TypeList &ts) : known(true), len(ts.size()), elementTypes(ts) {}

  bool operator==(const Shape &shape) const {
    if (known) {
      return known == shape.known && elementTypes == elementTypes;
    }
    return known == shape.known;
  }

  L::hash_code hash_value() const {
    if (!known) {
      return L::hash_combine(0);
    }
    return L::hash_combine_range(elementTypes.begin(), elementTypes.end());
  }

  // Does this shape have a known size
  bool isKnown() const { return known; }

  // Returns a vector of element types, if the shape is of a known size
  TypeList getElementTypes() const {
    assert(known);
    return elementTypes;
  }

  // Returns the shape size, if known
  unsigned arity() const {
    assert(known);
    return len;
  }

  unsigned subclassData() const {
    if (!known) {
      return 0;
    } else {
      return len;
    }
  }

  // Returns the type of the element at the given index, if known
  //
  // NOTE: Panics if the given index is larger than the last known
  // element, when the size is known. If the size is not known, it
  // simply returns None.
  L::Optional<M::Type> getType(unsigned index) {
    if (!known) {
      return L::None;
    } else {
      assert(index < len);
      return L::Optional(elementTypes[index]);
    }
  }

private:
  bool known;
  unsigned len;
  TypeList elementTypes;
};

class DynamicallyShapedType : public M::Type {
public:
  using ImplType = detail::ShapedTypeStorage;
  using M::Type::Type;

  Shape getShape() const { return shape; }

  unsigned getNumElements() const { return getShape().arity(); }

  bool hasStaticShape() const { return getShape().isKnown(); }

  static bool classof(M::Type type) {
    return type.getKind() == EirTypes::Tuple ||
           type.getKind() == EirTypes::Closure;
  }

protected:
  Shape shape;
};

class TupleType : public M::Type::TypeBase<TupleType, DynamicallyShapedType,
                                           detail::TupleTypeStorage> {
public:
  using Base::Base;

  static TupleType get(M::MLIRContext *context, const Shape &shape);
  static bool kindof(unsigned kind) { return kind == EirTypes::Tuple; }

  int64_t getSizeInBytes() const;

  detail::TupleTypeStorage const *uniqueKey() const;

  static mlir::LogicalResult
  verifyConstructionInvariants(L::Optional<M::Location> loc,
                               M::MLIRContext *context, const Shape &shape);
};

class ClosureType : public M::Type::TypeBase<ClosureType, DynamicallyShapedType,
                                             detail::ClosureTypeStorage> {
public:
  using Base::Base;

  M::Type getFnType() const;

  static ClosureType get(M::MLIRContext *context, M::Type fnType);
  static ClosureType get(M::MLIRContext *context, M::Type fnType,
                         const Shape &shape);
  static bool kindof(unsigned kind) { return kind == EirTypes::Closure; }

  int64_t getSizeInBytes() const;

  detail::ClosureTypeStorage const *uniqueKey() const;

  static M::LogicalResult
  verifyConstructionInvariants(L::Optional<M::Location> loc,
                               M::MLIRContext *context, M::Type fnTy,
                               const Shape &shape);
};

class BoxType
    : public M::Type::TypeBase<BoxType, M::Type, detail::BoxTypeStorage> {
public:
  using Base::Base;

  M::Type getEleTy() const;

  static BoxType get(M::MLIRContext *context, M::Type eleTy);
  static bool kindof(unsigned kind) { return kind == EirTypes::Box; }

  static M::LogicalResult verifyConstructionInvariants(L::Optional<M::Location>,
                                                       M::MLIRContext *ctx,
                                                       M::Type eleTy);
};

class RefType
    : public M::Type::TypeBase<RefType, M::Type, detail::RefTypeStorage> {
public:
  using Base::Base;

  M::Type getEleTy() const;

  static RefType get(M::MLIRContext *context, M::Type eleTy);
  static bool kindof(unsigned kind) { return kind == EirTypes::Ref; }

  static M::LogicalResult
  verifyConstructionInvariants(L::Optional<M::Location> loc,
                               M::MLIRContext *context, M::Type eleTy);
};

template <typename A, typename B>
bool inbounds(A v, B lb, B ub) {
  return v >= lb && v < ub;
}

bool isa_eir_type(M::Type);
bool isa_std_type(M::Type t);
bool isa_eir_or_std_type(M::Type t);

} // namespace eir

#endif // EIR_TYPES_H
