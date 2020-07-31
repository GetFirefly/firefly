#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIRTypes.h"
#include "lumen/EIR/IR/EIREnums.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"

using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::mlir::LLVM::LLVMType;

namespace lumen {
namespace eir {
namespace detail {

/// A type representing a collection of other types.
struct TupleTypeStorage final
    : public mlir::TypeStorage,
      public llvm::TrailingObjects<TupleTypeStorage, Type> {
  using KeyTy = ArrayRef<Type>;

  TupleTypeStorage(unsigned arity) : TypeStorage(arity) {}

  /// Construction.
  static TupleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     ArrayRef<Type> key) {
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
  unsigned size() const { return getSubclassData(); }

  /// Return the held types.
  ArrayRef<Type> getTypes() const {
    return {getTrailingObjects<Type>(), size()};
  }
};

struct BoxTypeStorage : public mlir::TypeStorage {
  using KeyTy = Type;

  BoxTypeStorage(Type boxedType)
      : TypeStorage(boxedType.getKind()),
        boxedType(boxedType.cast<OpaqueTermType>()) {}

  /// The hash key used for uniquing.
  bool operator==(const KeyTy &key) const { return key == boxedType; }

  static BoxTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<BoxTypeStorage>()) BoxTypeStorage(key);
  }

  OpaqueTermType boxedType;
};

struct RefTypeStorage : public mlir::TypeStorage {
  using KeyTy = Type;

  RefTypeStorage(Type innerType)
      : TypeStorage(innerType.getKind()),
        innerType(innerType.cast<OpaqueTermType>()) {}

  /// The hash key used for uniquing.
  bool operator==(const KeyTy &key) const { return key == innerType; }

  static RefTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<RefTypeStorage>()) RefTypeStorage(key);
  }

  OpaqueTermType innerType;
};

struct PtrTypeStorage : public mlir::TypeStorage {
  using KeyTy = Type;

  PtrTypeStorage(Type innerType)
      : TypeStorage(innerType.getKind()), innerType(innerType) {}

  /// The hash key used for uniquing.
  bool operator==(const KeyTy &key) const { return key == innerType; }

  static PtrTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    // Initialize the memory using placement new.
    return new (allocator.allocate<PtrTypeStorage>()) PtrTypeStorage(key);
  }

  Type innerType;
};

}  // namespace detail
}  // namespace eir
}  // namespace lumen

//===----------------------------------------------------------------------===//
// Type Implementations
//===----------------------------------------------------------------------===//

namespace lumen {
namespace eir {

// Tuple<T>

TupleType TupleType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::Tuple, ArrayRef<Type>{});
}

TupleType TupleType::get(MLIRContext *context, ArrayRef<Type> elementTypes) {
  return Base::get(context, TypeKind::Tuple, elementTypes);
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
  return Base::get(context, TypeKind::Tuple, elementTypes);
}

TupleType TupleType::get(unsigned arity, Type elementType) {
  return TupleType::get(elementType.getContext(), arity, elementType);
}

TupleType TupleType::get(ArrayRef<Type> elementTypes) {
  auto context = elementTypes.front().getContext();
  return Base::get(context, TypeKind::Tuple, elementTypes);
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
      if (llvmType.isIntegerTy())
        continue;
    }
    llvm::outs() << "invalid tuple type element (" << i << "): ";
    elementType.dump();
    llvm::outs() << "\n";
    return failure();
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

// Box<T>

BoxType BoxType::get(OpaqueTermType boxedType) {
  return Base::get(boxedType.getContext(), TypeKind::Box, boxedType);
}

BoxType BoxType::get(MLIRContext *context, OpaqueTermType boxedType) {
  return Base::get(context, TypeKind::Box, boxedType);
}

BoxType BoxType::getChecked(Type type, Location location) {
  return Base::getChecked(location, TypeKind::Box, type);
}

OpaqueTermType BoxType::getBoxedType() const { return getImpl()->boxedType; }

// Ref<T>

RefType RefType::get(OpaqueTermType innerType) {
  return Base::get(innerType.getContext(), TypeKind::Ref, innerType);
}

RefType RefType::get(MLIRContext *context, OpaqueTermType innerType) {
  return Base::get(context, TypeKind::Ref, innerType);
}

RefType RefType::getChecked(Type type, Location location) {
  return Base::getChecked(location, TypeKind::Ref, type);
}

OpaqueTermType RefType::getInnerType() const { return getImpl()->innerType; }

// Ptr<T>

PtrType PtrType::get(Type innerType) {
  return Base::get(innerType.getContext(), TypeKind::Ptr, innerType);
}

PtrType PtrType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::Ptr, mlir::IntegerType::get(8, context));
}

Type PtrType::getInnerType() const { return getImpl()->innerType; }

// ReceiveRef

ReceiveRefType ReceiveRefType::get(MLIRContext *context) {
  return Base::get(context, TypeKind::ReceiveRef);
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
  switch (ty.getKind()) {
    case TypeKind::None:
      os << "none";
      break;
    case TypeKind::Term:
      os << "term";
      break;
    case TypeKind::List:
      os << "list";
      break;
    case TypeKind::Number:
      os << "number";
      break;
    case TypeKind::Integer:
      os << "integer";
      break;
    case TypeKind::Float:
      os << "float";
      break;
    case TypeKind::Atom:
      os << "atom";
      break;
    case TypeKind::Boolean:
      os << "bool";
      break;
    case TypeKind::Fixnum:
      os << "fixnum";
      break;
    case TypeKind::BigInt:
      os << "bigint";
      break;
    case TypeKind::Nil:
      os << "nil";
      break;
    case TypeKind::Cons:
      os << "cons";
      break;
    case TypeKind::Map:
      os << "map";
      break;
    case TypeKind::Closure:
      os << "closure";
      break;
    case TypeKind::Binary:
      os << "binary";
      break;
    case TypeKind::HeapBin:
      os << "heapbin";
      break;
    case TypeKind::ProcBin:
      os << "procbin";
      break;
    case TypeKind::Tuple: {
      printTuple(ty.cast<TupleType>(), os, p);
    } break;
    case TypeKind::Box: {
      auto type = ty.cast<BoxType>();
      os << "box<";
      p.printType(type.getBoxedType());
      os << '>';
    } break;
    case TypeKind::Ref: {
      auto type = ty.cast<RefType>();
      os << "ref<";
      p.printType(type.getInnerType());
      os << '>';
    } break;
    case TypeKind::Ptr: {
      auto type = ty.cast<PtrType>();
      os << "ptr<";
      p.printType(type.getInnerType());
      os << '>';
    } break;
    case TypeKind::ReceiveRef:
      os << "receive_ref";
      break;
    default:
      llvm_unreachable("unhandled EIR type");
  }
}

}  // namespace eir
}  // namespace lumen
