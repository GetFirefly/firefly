#ifndef EIR_TYPES_H
#define EIR_TYPES_H

#include "lumen/LLVM.h"

#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include <vector>

namespace L = llvm;
namespace M = mlir;

/// This enumeration represents all of the types defined by the EIR dialect
namespace eir {

namespace detail {
struct TermBaseStorage;
struct ShapedTypeStorage;
struct TupleTypeStorage;
struct ClosureTypeStorage;
struct BoxTypeStorage;
struct RefTypeStorage;
} // namespace detail

namespace EirTypes {
enum EirTypes {
  // A generic type used for an unknown term value
  Term = M::Type::FIRST_EIR_TYPE,
  // A generic type used only for specifying a type match for any list type
  AnyList,
  // A generic type used only for specifying a type match for any number type
  AnyNumber,
  // A generic type used only for specifying a type match for any integer type
  AnyInteger,
  // A generic type used only for specifying a type match for any float type
  AnyFloat,
  // A generic type used only for specifying a type match for any bianry type
  AnyBinary,
  // Start of concrete types
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
  Binary,
  HeapBin,
  Box,
  Ref,
};
} // namespace EirTypes

class TermBase : public M::Type {
public:
  using ImplType = detail::TermBaseStorage;
  using M::Type::Type;

  unsigned getImplKind() const { return implKind; }

  bool isConcrete() const { return isConcrete(getImplKind()); }

  bool isImmediate() const { return isImmediate(getImplKind()); }

  bool isAtom() const { return isAtom(getImplKind()); }

  bool isNumber() const { return isNumber(getImplKind()); }

  bool isInteger() const { return isInteger(getImplKind()); }

  bool isFloat() const { return isFloat(getImplKind()); }

  bool isList() const { return isList(getImplKind()); }

  bool isNil() const { return isNil(getImplKind()); }

  bool isNonEmptyList() const { return isNonEmptyList(getImplKind()); }

  bool isBinary() const { return isBinary(getImplKind()); }

  // Returns 0 for false, 1 for true, 2 for unknown
  unsigned isMatch(M::Type matcher) const {
    auto matcherBase = matcher.dyn_cast_or_null<TermBase>();
    if (!matcherBase)
      return 2;

    auto implKind = getImplKind();
    auto matcherImplKind = matcherBase.getImplKind();

    // Unresolvable statically
    if (!isConcrete(implKind) || !matcherBase.isConcrete(matcherImplKind))
      return 2;

    // Guaranteed to match
    if (implKind == matcherImplKind)
      return 1;

    // Generic matches
    if (matcherImplKind == EirTypes::Atom)
      return isAtom(implKind) ? 1 : 0;
    if (matcherImplKind == EirTypes::AnyList)
      return isList(implKind) ? 1 : 0;
    if (matcherImplKind == EirTypes::AnyNumber)
      return isNumber(implKind) ? 1 : 0;
    if (matcherImplKind == EirTypes::AnyInteger)
      return isInteger(implKind) ? 1 : 0;
    if (matcherImplKind == EirTypes::AnyFloat)
      return isFloat(implKind) ? 1 : 0;
    if (matcherImplKind == EirTypes::AnyBinary)
      return isBinary(implKind) ? 1 : 0;

    return 0;
  }

  static bool classof(M::Type type) {
    auto kind = type.getKind();
    return kind >= EirTypes::Term && kind <= EirTypes::Ref;
  }

private:
  unsigned implKind;

  bool isConcrete(unsigned implKind) const {
    return implKind > EirTypes::AnyBinary;
  }

  bool isImmediate(unsigned implKind) const {
    return implKind == EirTypes::Atom ||
      implKind == EirTypes::Boolean ||
      implKind == EirTypes::Fixnum ||
      implKind == EirTypes::Float ||
      implKind == EirTypes::Nil ||
      implKind == EirTypes::Box ||
      implKind == EirTypes::Ref;
  }

  bool isAtom(unsigned implKind) const {
    return implKind >= EirTypes::Atom && implKind <= EirTypes::Boolean;
  }

  bool isNumber(unsigned implKind) const {
    return implKind >= EirTypes::Fixnum && implKind <= EirTypes::FloatPacked;
  }

  bool isInteger(unsigned implKind) const {
    return implKind >= EirTypes::Fixnum && implKind <= EirTypes::BigInt;
  }

  bool isFloat(unsigned implKind) const {
    return implKind >= EirTypes::Float && implKind <= EirTypes::FloatPacked;
  }

  bool isList(unsigned implKind) const {
    return implKind >= EirTypes::Nil && implKind <= EirTypes::Cons;
  }

  bool isNil(unsigned implKind) const {
    return implKind == EirTypes::Nil;
  }

  bool isNonEmptyList(unsigned implKind) const {
    return implKind == EirTypes::Cons;
  }

  bool isBinary(unsigned implKind) const {
    return implKind >= EirTypes::Binary && implKind <= EirTypes::HeapBin;
  }
};

#define IntrinsicType(TYPE, KIND)                                              \
  class TYPE : public M::Type::TypeBase<TYPE, TermBase> {                      \
  public:                                                                      \
    using Base::Base;                                                          \
    static TYPE get(M::MLIRContext *context) {                                 \
      return Base::get(context, KIND);                                         \
    }                                                                          \
    static bool kindof(unsigned kind) { return kind == KIND; }                 \
  }

IntrinsicType(TermType, EirTypes::Term);
IntrinsicType(AnyListType, EirTypes::AnyList);
IntrinsicType(AnyNumberType, EirTypes::AnyNumber);
IntrinsicType(AnyIntegerType, EirTypes::AnyInteger);
IntrinsicType(AnyFloatType, EirTypes::AnyFloat);
IntrinsicType(AnyBinaryType, EirTypes::AnyBinary);
IntrinsicType(AtomType, EirTypes::Atom);
IntrinsicType(BooleanType, EirTypes::Boolean);
IntrinsicType(FixnumType, EirTypes::Fixnum);
IntrinsicType(BigIntType, EirTypes::BigInt);
IntrinsicType(FloatType, EirTypes::Float);
IntrinsicType(PackedFloatType, EirTypes::FloatPacked);
IntrinsicType(NilType, EirTypes::Nil);
IntrinsicType(ConsType, EirTypes::Cons);
IntrinsicType(MapType, EirTypes::Map);
IntrinsicType(BinaryType, EirTypes::Binary);
IntrinsicType(HeapBinType, EirTypes::HeapBin);

// Shaped Types, i.e. types that are parameterized and have dynamic extent

using TypeList = std::vector<M::Type>;

struct Shape {
  Shape() : known(false) {}
  Shape(M::Type elementType, unsigned len)
      : known(true), len(len), elementTypes(len, elementType) {}
  Shape(const TypeList &ts) : known(true), len(ts.size()), elementTypes(ts) {}

  static Shape infer(L::ArrayRef<M::Value> elements) {
    auto size = elements.size();
    // If there are no elements, then the shape is unknown
    if (size == 0) {
      return Shape();
    }
    // Otherwise, construct a list of types from the elements
    std::vector<M::Type> types;
    types.reserve(size);
    for (auto &element : elements) {
      types.push_back(element.getType());
    }
    return Shape(types);
  }

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

class DynamicallyShapedType : public TermBase {
public:
  using ImplType = detail::ShapedTypeStorage;
  using TermBase::TermBase;

  Shape getShape() const { return shape; }

  unsigned getNumElements() const { return getShape().arity(); }

  bool hasStaticShape() const { return getShape().isKnown(); }

  static bool classof(M::Type type) {
    auto kind = type.getKind();
    return kind == EirTypes::Tuple || kind == EirTypes::Closure;
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
    : public M::Type::TypeBase<BoxType, TermBase, detail::BoxTypeStorage> {
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
    : public M::Type::TypeBase<RefType, TermBase, detail::RefTypeStorage> {
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
