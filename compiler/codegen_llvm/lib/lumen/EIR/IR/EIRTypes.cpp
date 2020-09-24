#include "lumen/EIR/IR/EIRTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIREnums.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Parser.h"

using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::mlir::TypeRange;
using ::mlir::LLVM::LLVMType;

//===----------------------------------------------------------------------===//
// Type Implementations
//===----------------------------------------------------------------------===//

namespace lumen {
namespace eir {

// Tuple<T>
namespace detail {
struct TupleTypeStorage final
    : public mlir::TypeStorage,
      public llvm::TrailingObjects<TupleTypeStorage, Type> {
  using KeyTy = TypeRange;

  TupleTypeStorage(unsigned arity) : arity(arity) {}

  /// Construction.
  static TupleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     TypeRange key) {
    // Allocate a new storage instance.
    auto byteSize = TupleTypeStorage::totalSizeToAlloc<Type>(key.size());
    auto rawMem = allocator.allocate(byteSize, alignof(TupleTypeStorage));
    auto result = ::new (rawMem) TupleTypeStorage(key.size());

    // Copy in the element types into the trailing storage.
    std::uninitialized_copy(key.begin(), key.end(),
                            result->getTrailingObjects<Type>());
    return result;
  }

  bool operator==(const KeyTy &key) const { return key == getTypes(); }

  /// Return the number of held types.
  unsigned size() const { return arity; }

  /// Return the held types.
  ArrayRef<Type> getTypes() const {
    return {getTrailingObjects<Type>(), size()};
  }

 private:
  unsigned arity;
};
} // namespace detail

TupleType TupleType::get(MLIRContext *context) {
  return Base::get(context, ArrayRef<Type>{});
}

TupleType TupleType::get(MLIRContext *context, ArrayRef<Type> elementTypes) {
  return Base::get(context, elementTypes);
}

TupleType TupleType::get(MLIRContext *context, unsigned arity) {
  return TupleType::get(context, arity, TermType::get(context));
}

TupleType TupleType::get(MLIRContext *context, unsigned arity,
                         Type elementType) {
  SmallVector<Type, 4> elementTypes;
  for (unsigned i = 0; i < arity; i++) {
    elementTypes.push_back(elementType);
  }
  return Base::get(context, elementTypes);
}

TupleType TupleType::get(unsigned arity, Type elementType) {
  return TupleType::get(elementType.getContext(), arity, elementType);
}

TupleType TupleType::get(ArrayRef<Type> elementTypes) {
  auto context = elementTypes.front().getContext();
  return Base::get(context, elementTypes);
}

LogicalResult TupleType::verifyConstructionInvariants(
    Location loc, ArrayRef<Type> elementTypes) {
  auto arity = elementTypes.size();
  if (arity < 1) {
    // If this is dynamically-shaped, then there is nothing to verify
    return success();
  }

  // Make sure elements are word-sized/immediates, and valid
  unsigned numElements = elementTypes.size();
  for (unsigned i = 0; i < numElements; i++) {
    Type elementType = elementTypes[i];
    if (auto termType = elementType.dyn_cast_or_null<OpaqueTermType>()) {
      if (termType.isOpaque() || termType.isImmediate() || termType.isBox())
        continue;
    }
    if (auto llvmType = elementType.dyn_cast_or_null<LLVMType>()) {
      if (llvmType.isIntegerTy()) continue;
    }
    // Allow an exception for TraceRef, since it will be replaced by the
    // InsertTraceConstructors pass
    if (elementType.isa<TraceRefType>())
      continue;

    return emitError(loc, "invalid tuple type element at index ") << i << ": " << elementType;
  }

  return success();
}

size_t TupleType::getArity() const { return getImpl()->size(); }
size_t TupleType::getSizeInBytes() const {
  auto arity = getArity();
  if (arity < 0) return -1;
  // Header word is always present, each element is one word
  return 8 + (arity * 8);
};
bool TupleType::hasStaticShape() const { return getArity() != 0; }
bool TupleType::hasDynamicShape() const { return getArity() == 0; }
Type TupleType::getElementType(unsigned index) const {
  return getImpl()->getTypes()[index];
}

// Closure
namespace detail {
struct ClosureTypeKey {
  ClosureTypeKey() : functionType(nullptr), envTypes(nullptr), envLen(llvm::None) {}
  ClosureTypeKey(FunctionType ft) : functionType(ft), envTypes(nullptr), envLen(llvm::None) {}
  ClosureTypeKey(size_t len) : functionType(nullptr), envTypes(nullptr), envLen(len) {}
  ClosureTypeKey(TypeRange env) : functionType(nullptr), envTypes(env), envLen(env.size()) {}
  ClosureTypeKey(FunctionType ft, size_t len) : functionType(ft), envTypes(nullptr), envLen(len) {}
  ClosureTypeKey(FunctionType ft, TypeRange env) : functionType(ft), envTypes(env), envLen(env.size()) {}
  ClosureTypeKey(Optional<FunctionType> ft, Optional<TypeRange> env, Optional<size_t> len)
    : functionType(ft), envTypes(env), envLen(len) {}
  ClosureTypeKey(const ClosureTypeKey &key)
    : functionType(key.functionType), envTypes(key.envTypes), envLen(key.envLen) {}

  bool operator==(const ClosureTypeKey &key) const {
    // Fully dynamic closure types are equivalent
    if (!hasStaticShape() && !key.hasStaticShape())
      return true;

    // Closure types with differing callee types are considered unique
    if (functionType.hasValue() && key.functionType.hasValue()) {
      if (functionType.getValue() != key.functionType.getValue())
        return false;
    } else if (functionType.hasValue() || key.functionType.hasValue()) {
      return false;
    }

    // Closure types with differently typed environents are considered unique
    if (envTypes.hasValue() && key.envTypes.hasValue()) {
      if (envTypes.getValue() != key.envTypes.getValue())
        return false;
    } else if (envTypes.hasValue() && !key.envTypes.hasValue()) {
      for (auto ty : envTypes.getValue())
        if (!ty.isa<TermType>())
          return false;
    } else if (key.envTypes.hasValue() && !envTypes.hasValue()) {
      for (auto ty : key.envTypes.getValue())
        if (!ty.isa<TermType>())
          return false;
    } else if (envLen.hasValue() && key.envLen.hasValue()) {
      if (envLen.getValue() != key.envLen.getValue())
        return false;
    } else if (envLen.hasValue() && !key.envLen.hasValue()) {
      return false;
    } else if (!envLen.hasValue() && key.envLen.hasValue()) {
      return false;
    }

    return true;
  }

  inline bool hasStaticShape() const { return envTypes.hasValue() || envLen.hasValue(); }

  Optional<FunctionType> functionType;
  Optional<TypeRange> envTypes;
  Optional<size_t> envLen;
};
  
struct ClosureTypeStorage final
    : public mlir::TypeStorage,
      public llvm::TrailingObjects<ClosureTypeStorage, Type> {
  using KeyTy = ClosureTypeKey;

  ClosureTypeStorage(Optional<FunctionType> ft, Optional<TypeRange> env, Optional<size_t> len)
      : functionType(ft), envTypes(env), envLen(len) {}

  static KeyTy getKey(Optional<FunctionType> functionType, Optional<TypeRange> envTypes, Optional<size_t> envLen) {
    return KeyTy(functionType, envTypes, envLen);
  }

  /// Construction.
  static ClosureTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    // Allocate a new storage instance.
    size_t actualEnvLen;
    if (key.envTypes.hasValue())
      actualEnvLen = key.envTypes.getValue().size();
    else if (key.envLen.hasValue())
      actualEnvLen = key.envLen.getValue();
    else
      actualEnvLen = 0;

    auto byteSize = ClosureTypeStorage::totalSizeToAlloc<Type>(actualEnvLen);
    auto rawMem = allocator.allocate(byteSize, alignof(ClosureTypeStorage));
    ClosureTypeStorage *result =
      ::new (rawMem) ClosureTypeStorage(key.functionType, key.envTypes, key.envLen);

    if (key.envTypes.hasValue()) {
      auto env = key.envTypes.getValue();
      // Copy in the element types into the trailing storage.
      std::uninitialized_copy(env.begin(), env.end(),
                              result->getTrailingObjects<Type>());
    }

    return result;
  }

  bool operator==(const KeyTy &key) const {
    return key == getKey(functionType, envTypes, envLen);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    Type functionType = key.functionType.hasValue() ? Type(key.functionType.getValue()) : Type();
    TypeRange envTypes = key.envTypes.hasValue() ? TypeRange(key.envTypes.getValue()) : TypeRange();
    size_t envLen = key.envLen.hasValue() ? key.envLen.getValue() : 0;
    return llvm::hash_combine(mlir::hash_value(functionType), mlir::hash_value(envTypes), envLen);
  }

  Optional<FunctionType> getCalleeType() const { return functionType; }

  Optional<size_t> arity() const {
    if (!functionType.hasValue())
      return llvm::None;
    auto fnTy = functionType.getValue();
    return fnTy.getNumInputs();
  }
  inline size_t getEnvLen() const { return envLen.hasValue() ? envLen.getValue() : 0; }
  inline bool hasTypedEnv() const { return envTypes.hasValue(); }
  inline bool hasStaticShape() const { return envTypes.hasValue() || envLen.hasValue(); }
  TypeRange getEnvTypes() const { return TypeRange(ArrayRef<Type>{getTrailingObjects<Type>(), getEnvLen()}); }

  Type getEnvType(unsigned index) const {
    if (!hasTypedEnv())
      return nullptr;
    auto envTypes = getEnvTypes();
    if (envTypes.size() > index)
      return envTypes[index];
    return nullptr;
  }

 private:
  Optional<FunctionType> functionType;
  Optional<TypeRange> envTypes;
  Optional<size_t> envLen;
};
} // namespace detail

ClosureType ClosureType::get(MLIRContext *context) {
  return Base::get(context, detail::ClosureTypeStorage::KeyTy());
}

ClosureType ClosureType::get(MLIRContext *context, size_t envLen) {
  return Base::get(context, detail::ClosureTypeStorage::KeyTy(envLen));
}

ClosureType ClosureType::get(MLIRContext *context, TypeRange env) {
  return Base::get(context, detail::ClosureTypeStorage::KeyTy(env));
}

ClosureType ClosureType::get(MLIRContext *context, FunctionType functionType) {
  return Base::get(context, detail::ClosureTypeStorage::KeyTy(functionType));
}

ClosureType ClosureType::get(MLIRContext *context, FunctionType functionType, size_t envLen) {
  return Base::get(context, detail::ClosureTypeStorage::KeyTy(functionType, envLen));
}

ClosureType ClosureType::get(MLIRContext *context, FunctionType functionType, TypeRange env) {
  return Base::get(context, detail::ClosureTypeStorage::KeyTy(functionType, env));
}

LogicalResult ClosureType::verifyConstructionInvariants(
    Location loc, const detail::ClosureTypeStorage::KeyTy &key) {
  // TODO
  return success();
}

Optional<FunctionType> ClosureType::getCalleeType() const { return getImpl()->getCalleeType(); }

Optional<size_t> ClosureType::getArity() const { return getImpl()->arity(); }

size_t ClosureType::getEnvLen() const { return getImpl()->getEnvLen(); }

bool ClosureType::hasStaticShape() const { return getImpl()->hasStaticShape(); }

bool ClosureType::hasDynamicShape() const { return !hasStaticShape(); }

Type ClosureType::getEnvType(unsigned index) const {
  if (!getImpl()->hasTypedEnv())
    return TermType::get(getContext());
  Type result = getImpl()->getEnvType(index);
  if (result)
    return result;
  return NoneType::get(getContext());
}

// Box<T>
namespace detail {
struct BoxTypeStorage : public mlir::TypeStorage {
  using KeyTy = Type;

  BoxTypeStorage(Type boxedType)
      : boxedType(boxedType.cast<OpaqueTermType>()) {}

  /// The hash key used for uniquing.
  bool operator==(const KeyTy &key) const { return key == boxedType; }

  static BoxTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<BoxTypeStorage>()) BoxTypeStorage(key);
  }

  OpaqueTermType boxedType;
};
} // namespace detail

BoxType BoxType::get(OpaqueTermType boxedType) {
  return Base::get(boxedType.getContext(), boxedType);
}

BoxType BoxType::get(MLIRContext *context, OpaqueTermType boxedType) {
  return Base::get(context, boxedType);
}

BoxType BoxType::getChecked(Type type, Location location) {
  return Base::getChecked(location, type);
}

OpaqueTermType BoxType::getBoxedType() const { return getImpl()->boxedType; }

// Ref<T>

namespace detail {
struct RefTypeStorage : public mlir::TypeStorage {
  using KeyTy = Type;

  RefTypeStorage(Type innerType)
      : innerType(innerType.cast<OpaqueTermType>()) {}

  /// The hash key used for uniquing.
  bool operator==(const KeyTy &key) const { return key == innerType; }

  static RefTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<RefTypeStorage>()) RefTypeStorage(key);
  }

  OpaqueTermType innerType;
};
} // namespace detail

RefType RefType::get(OpaqueTermType innerType) {
  return Base::get(innerType.getContext(), innerType);
}

RefType RefType::get(MLIRContext *context, OpaqueTermType innerType) {
  return Base::get(context, innerType);
}

RefType RefType::getChecked(Type type, Location location) {
  return Base::getChecked(location, type);
}

OpaqueTermType RefType::getInnerType() const { return getImpl()->innerType; }

// Ptr<T>

namespace detail {
struct PtrTypeStorage : public mlir::TypeStorage {
  using KeyTy = Type;

  PtrTypeStorage(Type innerType) : innerType(innerType) {}

  /// The hash key used for uniquing.
  bool operator==(const KeyTy &key) const { return key == innerType; }

  static PtrTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<PtrTypeStorage>()) PtrTypeStorage(key);
  }

  Type innerType;
};
} // namespace detail

PtrType PtrType::get(Type innerType) {
  return Base::get(innerType.getContext(), innerType);
}

PtrType PtrType::get(MLIRContext *context) {
  return PtrType::get(context, mlir::IntegerType::get(8, context));
}

PtrType PtrType::get(MLIRContext *context, Type innerType) {
  return Base::get(context, innerType);
}

Type PtrType::getInnerType() const { return getImpl()->innerType; }

// TraceRef

TraceRefType TraceRefType::get(MLIRContext *context) {
  return Base::get(context);
}

// ReceiveRef

ReceiveRefType ReceiveRefType::get(MLIRContext *context) {
  return Base::get(context);
}


// OpaqueTermType

unsigned OpaqueTermType::isMatch(Type matcher) {
  auto matcherBase = matcher.dyn_cast_or_null<OpaqueTermType>();
  if (!matcherBase) return 2;

  auto typeId = getTypeID();
  auto matcherTypeId = matcher.getTypeID();

  // Unresolvable statically
  if (isOpaque() || matcherBase.isOpaque()) return 2;

  // Guaranteed to match
  if (typeId == matcherTypeId) return 1;

  // Handle boxed types if the matcher is a box type
  if (matcherBase.isBox() && isBox()) {
    auto expected = matcher.cast<BoxType>().getBoxedType();
    auto inner = cast<BoxType>().getBoxedType();
    return inner.isMatch(expected);
  }

  // If the matcher is not a box, but is a boxable type, handle
  // comparing types correctly (i.e. if this is a boxed type, then
  // compare the boxed type)
  if (matcherBase.isBoxable() && isBox()) {
    auto inner = cast<BoxType>().getBoxedType();
    return inner.isMatch(matcher);
  }

  // Generic matches
  if (matcher.isa<AtomType>()) return isAtom() ? 1 : 0;
  if (matcher.isa<BooleanType>())
    if (isBoolean())
      return 1;
    else if (isAtom())
      return 2;
    else
      return 0;
  if (matcher.isa<NumberType>()) return isNumber() ? 1 : 0;
  if (matcher.isa<IntegerType>()) return isInteger() ? 1 : 0;
  if (matcher.isa<FloatType>()) return isFloat() ? 1 : 0;
  if (matcher.isa<BinaryType>()) return isBinary() ? 1 : 0;
  if (matcher.isa<ListType>()) return isList() ? 1 : 0;
  if (matcher.isa<MapType>()) return isMap() ? 1 : 0;
  if (matcher.isa<ClosureType>()) return isClosure() ? 2 : 0;

  if (auto tupleTy = matcher.dyn_cast_or_null<TupleType>())
    if (tupleTy.hasStaticShape() && isTuple()) {
      auto tt = cast<TupleType>();
      if (tt.hasStaticShape() && tt.getArity() != tupleTy.getArity())
        return 0;
      else
        return 2;
    }
    else if (isTuple())
      return 2;
    else
      return 0;

  return 0;
}

bool OpaqueTermType::canTypeEverBeEqual(Type other) {
  // We have to be pessimistic if this is opaque
  if (isOpaque())
    return true;

  // Only a limited subset of non-term types can be compared to terms
  if (!other.isa<OpaqueTermType>()) {
    // Primitives
    if (isBoolean() && other.isInteger(1))
      return true;
    if (isNumber() && other.isIntOrFloat())
      return true;

    // Pointer types
    if (auto ptrTy = other.dyn_cast_or_null<PtrType>()) {
      auto innerTy = ptrTy.getInnerType();
      // Equality can never hold against raw pointers
      if (innerTy.isInteger(1))
        return false;

      // If this is a box, we can compare the inner types,
      // otherwise continue as if this is a boxable type
      if (isBox()) {
        auto boxedTy = cast<BoxType>().getBoxedType();
        return boxedTy.canTypeEverBeEqual(innerTy);
      } else {
        return canTypeEverBeEqual(innerTy);
      }
    }

    if (auto refTy = other.dyn_cast_or_null<RefType>()) {
      auto innerTy = refTy.getInnerType();

      // If this is a box, we can compare the inner types,
      // otherwise continue as if this is a boxable type
      if (isBox()) {
        auto boxedTy = cast<BoxType>().getBoxedType();
        return boxedTy.canTypeEverBeEqual(innerTy);
      } else {
        return canTypeEverBeEqual(innerTy);
      }
    }

    return false;
  }

  // Once we reach here, we're comparing terms
  auto ty = other.cast<OpaqueTermType>();

  // Again, pessimistically assume all opaque terms are equatable
  if (ty.isOpaque())
    return true;

  // Unwrap boxed types
  if (isBox()) {
    auto lhs = cast<BoxType>().getBoxedType();

    if (ty.isBox()) {
      auto rhs = ty.cast<BoxType>().getBoxedType();
      return lhs.canTypeEverBeEqual(rhs);
    }

    return lhs.canTypeEverBeEqual(other);
  }
  if (ty.isBox()) {
    auto rhs = ty.cast<BoxType>().getBoxedType();
    return canTypeEverBeEqual(rhs);
  }

  // Numbers can only be equal to numbers
  if (isNumber() && !ty.isNumber())
    return false;

  // Atom-likes can only be equal to atom-likes
  if (isAtom() && !ty.isAtom())
    return false;

  // Pids can only be equal to pids
  if (isPid() && !isPid())
    return false;

  // References can only be equal to referencess
  if (isReference() && !isReference())
    return false;

  // Aggregates of each class can only be equal to the same class
  if (isList() && !ty.isList())
    return false;

  if (isTuple() && !ty.isTuple())
    return false;

  if (isMap() && !ty.isMap())
    return false;

  if (isBinary() && !ty.isBinary())
    return false;

  // Closures can only be equal to closures
  if (isClosure() && !ty.isClosure())
    return false;

  // Rather than explicitly whitelist valid equality comparisons,
  // we identify comparisons we know can't succeed. If we hit here,
  // we're saying that we pessimistically assume equality is possible,
  // but its possible that the operands would never compare equal, we
  // just haven't explicitly identified that case above
  return true;
}

}  // namespace eir
}  // namespace lumen

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

namespace lumen {
namespace eir {

template <typename TYPE>
TYPE parseTypeSingleton(mlir::MLIRContext *context,
                        mlir::DialectAsmParser &parser) {
  Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "type expected");
    return {};
  }
  if (auto innerTy = ty.dyn_cast_or_null<OpaqueTermType>())
    return TYPE::get(innerTy);
  else
    return {};
}

struct Shape {
  Shape() { arity = -1; }
  Shape(std::vector<Type> elementTypes)
      : arity(elementTypes.size()), elementTypes(elementTypes) {}
  int arity;
  std::vector<Type> elementTypes;
};

template <typename ShapedType>
ShapedType parseShapedType(mlir::MLIRContext *context,
                           mlir::DialectAsmParser &parser, bool allowAny) {
  // Check for '*'
  llvm::SMLoc anyLoc;
  bool isAny = !parser.parseOptionalStar();
  if (allowAny && isAny) {
    // This is an "any" shape, i.e. entirely dynamic
    return ShapedType::get(context);
  } else if (!allowAny && isAny) {
    parser.emitError(anyLoc, "'*' is not allowed here");
    return nullptr;
  }

  // No '*', check for dimensions
  assert(!isAny);

  SmallVector<int64_t, 1> dims;
  llvm::SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false)) {
    // No bounds, must be a element type list
    std::vector<Type> elementTypes;
    while (true) {
      Type eleTy;
      if (parser.parseType(eleTy)) {
        break;
      }
      elementTypes.push_back(eleTy);
      if (parser.parseOptionalComma()) {
        break;
      }
    }
    if (elementTypes.size() == 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected comma-separated list of element types");
      return nullptr;
    }
    return ShapedType::get(ArrayRef(elementTypes));
  } else {
    if (dims.size() != 1) {
      parser.emitError(countLoc, "expected single integer for element count");
      return nullptr;
    }
    int64_t len = dims[0];
    if (len < 0) {
      parser.emitError(countLoc, "element count cannot be negative");
      return nullptr;
    }
    if (len >= std::numeric_limits<unsigned>::max()) {
      parser.emitError(countLoc, "element count overflow");
      return nullptr;
    }
    unsigned ulen = static_cast<unsigned>(len);
    if (parser.parseOptionalQuestion()) {
      Type eleTy;
      if (parser.parseType(eleTy)) {
        parser.emitError(parser.getNameLoc(), "expecting element type");
        return nullptr;
      }
      return ShapedType::get(ulen, eleTy);
    } else {
      Type defaultType = TermType::get(context);
      return ShapedType::get(ulen, defaultType);
    }
  }

  return ShapedType::get(context);
}

// `tuple` `<` shape `>`
//   shape ::= `*` | bounds | type_list
//   type_list ::= type (`,` type)*
//   bounds ::= dim `x` type
//   dim ::= `?` | integer
Type parseTuple(MLIRContext *context, mlir::DialectAsmParser &parser) {
  Shape shape;
  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expected tuple shape");
    return {};
  }

  TupleType result =
      parseShapedType<TupleType>(context, parser, /*allowAny=*/true);

  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected tuple shape");
    return {};
  }

  return result;
}

Type eirDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef typeNameLit;
  if (failed(parser.parseKeyword(&typeNameLit))) return {};

  auto loc = parser.getNameLoc();
  auto context = getContext();
  // `term`
  if (typeNameLit == "term") return TermType::get(context);
  // `list`
  if (typeNameLit == "list") return ListType::get(context);
  // `pid`
  if (typeNameLit == "pid") return PidType::get(context);
  // `reference`
  if (typeNameLit == "reference") return ReferenceType::get(context);
  // `number`
  if (typeNameLit == "number") return NumberType::get(context);
  // `integer`
  if (typeNameLit == "integer") return IntegerType::get(context);
  // `float`
  if (typeNameLit == "float") return FloatType::get(context);
  // `atom`
  if (typeNameLit == "atom") return AtomType::get(context);
  // `boolean`
  if (typeNameLit == "boolean") return BooleanType::get(context);
  // `fixnum`
  if (typeNameLit == "fixnum") return FixnumType::get(context);
  // `bigint`
  if (typeNameLit == "bigint") return BigIntType::get(context);
  // `nil`
  if (typeNameLit == "nil") return NilType::get(context);
  // `cons`
  if (typeNameLit == "cons") return ConsType::get(context);
  // `map`
  if (typeNameLit == "map") return MapType::get(context);
  // `closure`
  if (typeNameLit == "closure") return ClosureType::get(context);
  // `binary`
  if (typeNameLit == "binary") return BinaryType::get(context);
  // `heapbin`
  if (typeNameLit == "heapbin") return HeapBinType::get(context);
  // `procbin`
  if (typeNameLit == "procbin") return ProcBinType::get(context);
  // See parseTuple
  if (typeNameLit == "tuple") return parseTuple(context, parser);
  // `box` `<` type `>`
  if (typeNameLit == "box") return parseTypeSingleton<BoxType>(context, parser);
  // `trace_ref`
  if (typeNameLit == "trace_ref") return TraceRefType::get(context);
  // `receive_ref`
  if (typeNameLit == "receive_ref") return ReceiveRefType::get(context);

  parser.emitError(loc, "unknown EIR type " + typeNameLit);
  return {};
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void printTuple(TupleType type, llvm::raw_ostream &os,
                mlir::DialectAsmPrinter &p) {
  os << "tuple<";
  if (type.hasDynamicShape()) {
    os << '*';
  }
  auto arity = type.getArity();
  // Single element is always uniform
  if (arity == 0) {
    os << "0x?";
    return;
  }
  if (arity == 1) {
    os << "1x";
    p.printType(type.getElementType(0));
    return;
  }
  // Check for uniformity to print more compact representation
  Type ty = type.getElementType(0);
  bool uniform = true;
  for (unsigned i = 1; i < arity; i++) {
    auto elementType = type.getElementType(i);
    if (elementType != ty) {
      uniform = false;
      break;
    }
  }
  if (uniform) {
    os << arity << 'x';
    p.printType(ty);
    return;
  }

  for (unsigned i = 0; i < arity; i++) {
    p.printType(type.getElementType(i));
    if (i + 1 < arity) {
      os << ", ";
    }
  }
  os << ">";
}

void eirDialect::printType(Type ty, mlir::DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  TypeSwitch<Type>(ty)
      .Case<NoneType>([&](Type) { os << "none"; })
      .Case<TermType>([&](Type) { os << "term"; })
      .Case<ListType>([&](Type) { os << "list"; })
      .Case<PidType>([&](Type) { os << "pid"; })
      .Case<ReferenceType>([&](Type) { os << "reference"; })
      .Case<NumberType>([&](Type) { os << "number"; })
      .Case<IntegerType>([&](Type) { os << "integer"; })
      .Case<FloatType>([&](Type) { os << "float"; })
      .Case<AtomType>([&](Type) { os << "atom"; })
      .Case<BooleanType>([&](Type) { os << "bool"; })
      .Case<FixnumType>([&](Type) { os << "fixnum"; })
      .Case<BigIntType>([&](Type) { os << "bigint"; })
      .Case<NilType>([&](Type) { os << "nil"; })
      .Case<ConsType>([&](Type) { os << "cons"; })
      .Case<MapType>([&](Type) { os << "map"; })
      .Case<ClosureType>([&](Type) { os << "closure"; })
      .Case<BinaryType>([&](Type) { os << "binary"; })
      .Case<HeapBinType>([&](Type) { os << "heapbin"; })
      .Case<ProcBinType>([&](Type) { os << "procbin"; })
      .Case<TupleType>([&](Type) { printTuple(ty.cast<TupleType>(), os, p); })
      .Case<BoxType>([&](Type) {
        os << "box<";
        p.printType(ty.cast<BoxType>().getBoxedType());
        os << ">";
      })
      .Case<RefType>([&](Type) {
        os << "ref<";
        p.printType(ty.cast<RefType>().getInnerType());
        os << ">";
      })
      .Case<PtrType>([&](Type) {
        os << "ptr<";
        p.printType(ty.cast<PtrType>().getInnerType());
        os << ">";
      })
      .Case<TraceRefType>([&](Type) { os << "trace_ref"; })
      .Case<ReceiveRefType>([&](Type) { os << "receive_ref"; })
      .Default([](Type) { llvm_unreachable("unknown eir type"); });
}

}  // namespace eir
}  // namespace lumen
