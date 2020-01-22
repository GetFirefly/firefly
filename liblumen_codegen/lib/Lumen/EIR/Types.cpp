#include "eir/Types.h"
#include "eir/Dialect.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

namespace L = llvm;
namespace M = mlir;

using namespace eir;

using llvm::SmallVector;

namespace eir {
namespace detail {

/// Term Types
struct TermBaseStorage : public M::TypeStorage {
  TermBaseStorage() = delete;
  TermBaseStorage(M::Type t)
      : TypeStorage(t.getKind()), implKind(t.getKind()) {}

  using KeyTy = unsigned;

  bool operator==(const KeyTy &key) const { return key == implKind; }

  unsigned getImplKind() const { return implKind; }

private:
  unsigned implKind;
};

/// Shaped Types
struct ShapedTypeStorage : public M::TypeStorage {
  ShapedTypeStorage() = delete;
  ShapedTypeStorage(const Shape &shape)
      : TypeStorage(shape.subclassData()), shape(shape) {}

  using KeyTy = Shape;

  bool operator==(const KeyTy &key) const { return key == shape; }

  Shape getShape() const { return shape; }

private:
  Shape shape;
};

/// Tuple object
struct TupleTypeStorage : public ShapedTypeStorage {
  using ShapedTypeStorage::KeyTy;
  using ShapedTypeStorage::ShapedTypeStorage;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{key.hash_value()};
    return L::hash_combine(shapeHash);
  }

  static TupleTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    auto *storage = allocator.allocate<TupleTypeStorage>();
    return new (storage) TupleTypeStorage{key};
  }

private:
  TupleTypeStorage() = delete;
  explicit TupleTypeStorage(Shape &shape) : ShapedTypeStorage(shape) {}
};

/// Closure object
struct ClosureTypeStorage : public ShapedTypeStorage {
  using ShapedTypeStorage::ShapedTypeStorage;

  using KeyTy = std::tuple<M::Type, Shape>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{fnType, getShape()};
  }

  static unsigned hashKey(const KeyTy &key) {
    auto hashVal{L::hash_combine(std::get<M::Type>(key))};
    Shape shape = std::get<Shape>(key);
    return L::hash_combine(hashVal, shape.hash_value());
  }

  static ClosureTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    auto fnType = std::get<M::Type>(key);
    auto shape = std::get<Shape>(key);
    auto *storage = allocator.allocate<ClosureTypeStorage>();
    return new (storage) ClosureTypeStorage{fnType, shape};
  }

  M::Type getFnType() const { return fnType; }

private:
  M::Type fnType;

  ClosureTypeStorage() = delete;
  explicit ClosureTypeStorage(M::Type fnTy, const Shape &shape)
      : ShapedTypeStorage(shape), fnType(fnTy) {}
};

/// Boxed object (a term descriptor)
struct BoxTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static BoxTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    auto *storage = allocator.allocate<BoxTypeStorage>();
    return new (storage) BoxTypeStorage{key};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  BoxTypeStorage() = delete;
  explicit BoxTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

/// Pointer-like object storage
struct RefTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static RefTypeStorage *construct(M::TypeStorageAllocator &allocator,
                                   M::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<RefTypeStorage>();
    return new (storage) RefTypeStorage{eleTy};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  RefTypeStorage() = delete;
  explicit RefTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

} // namespace detail
} // namespace eir

//===----------------------------------------------------------------------===//
// Type Implementations
//===----------------------------------------------------------------------===//

// Tuple<T>

TupleType eir::TupleType::get(M::MLIRContext *context, const Shape &shape) {
  return Base::get(context, EirTypes::Tuple, shape);
}

M::LogicalResult eir::TupleType::verifyConstructionInvariants(
    L::Optional<M::Location> loc, M::MLIRContext *context, const Shape &shape) {
  if (!shape.isKnown()) {
    // If this is dynamically-shaped, then there is nothing to verify
    return M::success();
  }

  // Make sure elements are word-sized/immediates, and valid
  TypeList elements = shape.getElementTypes();
  for (auto it = elements.begin(); it != elements.end(); it++) {
    M::Type ty = *it;
    if (ty.dyn_cast<AtomType>() || ty.dyn_cast<BooleanType>() ||
        ty.dyn_cast<FixnumType>() || ty.dyn_cast<FloatType>() ||
        ty.dyn_cast<NilType>() /*|| ty.dyn_cast<BoxType>()*/) {
      // This type is an immediate or box
      continue;
    } /*else if (ty.dyn_cast<PidType>() && ty.cast<PidType>().isLocal()) {
        // This PidType is local, i.e. immediate
        continue;
    }*/
    return M::failure();
  }

  return M::success();
}

// Closure<T, E>

ClosureType eir::ClosureType::get(M::MLIRContext *context, M::Type fnType) {
  Shape noShape = Shape();
  return Base::get(context, EirTypes::Closure, fnType, noShape);
}

ClosureType eir::ClosureType::get(M::MLIRContext *context, M::Type fnType,
                                  const Shape &shape) {
  return Base::get(context, EirTypes::Closure, fnType, shape);
}

M::Type eir::ClosureType::getFnType() const { return getImpl()->getFnType(); }

M::LogicalResult eir::ClosureType::verifyConstructionInvariants(
    L::Optional<M::Location> loc, M::MLIRContext *context, M::Type fnTy,
    const Shape &shape) {
  if (fnTy.dyn_cast<M::FunctionType>())
    return M::success();
  return M::failure();
}

// Box<T>

BoxType eir::BoxType::get(M::MLIRContext *context, M::Type elementType) {
  return Base::get(context, EirTypes::Box, elementType);
}

M::Type eir::BoxType::getEleTy() const { return getImpl()->getElementType(); }

M::LogicalResult
eir::BoxType::verifyConstructionInvariants(L::Optional<M::Location>,
                                           M::MLIRContext *ctx, M::Type eleTy) {
  // TODO
  return M::success();
}

// Ref<T>

RefType eir::RefType::get(M::MLIRContext *context, M::Type elementType) {
  return Base::get(context, EirTypes::Ref, elementType);
}

M::Type eir::RefType::getEleTy() const { return getImpl()->getElementType(); }

M::LogicalResult eir::RefType::verifyConstructionInvariants(
    L::Optional<M::Location> loc, M::MLIRContext *context, M::Type eleTy) {
  // It is not permitted for a reference to point to another reference
  if (eleTy.dyn_cast<RefType>())
    return M::failure();
  return M::success();
}

namespace eir {

//===----------------------------------------------------------------------===//
// Parsing
//===----------------------------------------------------------------------===//

template <typename TYPE>
TYPE parseTypeSingleton(M::MLIRContext *context, M::DialectAsmParser &parser) {
  M::Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "type expected");
    return {};
  }
  return TYPE::get(context, ty);
}

bool parseShape(M::MLIRContext *context, M::DialectAsmParser &parser,
                Shape &shape, bool allowAny) {
  // Check for '*'
  llvm::SMLoc anyLoc;
  bool isAny = !parser.parseOptionalStar();
  if (allowAny && isAny) {
    // This is an "any" shape, i.e. entirely dynamic
    return false;
  } else if (!allowAny && isAny) {
    parser.emitError(anyLoc, "'*' is not allowed here");
    return true;
  }

  // No '*', check for dimensions
  assert(!isAny);

  SmallVector<int64_t, 1> dims;
  llvm::SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false)) {
    // No bounds, must be a element type list
    TypeList elementTypes;
    while (true) {
      M::Type eleTy;
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
      return true;
    }
    shape = Shape(elementTypes);
  } else {
    if (dims.size() != 1) {
      parser.emitError(countLoc, "expected single integer for element count");
      return true;
    }
    int64_t len = dims[0];
    if (len < 0) {
      parser.emitError(countLoc, "element count cannot be negative");
      return true;
    }
    if (len >= std::numeric_limits<unsigned>::max()) {
      parser.emitError(countLoc, "element count overflow");
      return true;
    }
    unsigned ulen = static_cast<unsigned>(len);
    if (parser.parseOptionalQuestion()) {
      M::Type eleTy;
      if (parser.parseType(eleTy)) {
        parser.emitError(parser.getNameLoc(), "expecting element type");
        return true;
      }
      shape = Shape(eleTy, ulen);
    } else {
      M::Type defaultType = TermType::get(context);
      shape = Shape(defaultType, ulen);
    }
  }

  return false;
}

// `tuple` `<` shape `>`
//   shape ::= `*` | bounds | type_list
//   type_list ::= type (`,` type)*
//   bounds ::= dim `x` type
//   dim ::= `?` | integer
TupleType parseTuple(M::MLIRContext *context, M::DialectAsmParser &parser) {
  Shape shape;
  if (parser.parseLess() ||
      parseShape(context, parser, shape, /*allowAny=*/true) ||
      parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected tuple shape");
    return {};
  }
  return TupleType::get(context, shape);
}

// `closure` `<` fn_type | fn_type `,` env_shape `>`
//   env_shape ::= `*` | `[` shape `]`
//   shape ::= bounds | type_list
//   type_list ::= type (`,` type)*
//   bounds ::= dim `x` type
//   dim ::= `?` | integer
ClosureType parseClosure(M::MLIRContext *context, M::DialectAsmParser &parser) {
  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '<'");
    return {};
  }
  M::Type fnType;
  L::SMLoc fnLoc = parser.getCurrentLocation();
  if (parser.parseType(fnType)) {
    parser.emitError(fnLoc, "expected type");
    return {};
  }
  if (!fnType.dyn_cast<M::FunctionType>()) {
    parser.emitError(fnLoc, "expected function type");
    return {};
  }
  if (parser.parseOptionalComma()) {
    // This closure type is a function pointer with no env
    if (parser.parseLess()) {
      parser.emitError(parser.getCurrentLocation(), "expected '>'");
      return {};
    }
    return ClosureType::get(context, fnType);
  }
  Shape shape;
  if (parser.parseOptionalStar()) {
    if (parser.parseLSquare() ||
        parseShape(context, parser, shape, /*allowAny=*/false) ||
        parser.parseRSquare()) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected environment shape");
      return {};
    }
  }
  return ClosureType::get(context, fnType, shape);
}

bool isa_eir_type(M::Type t) {
  return inbounds(t.getKind(), M::Type::Kind::FIRST_EIR_TYPE,
                  M::Type::Kind::LAST_EIR_TYPE);
}

bool isa_std_type(M::Type t) {
  return inbounds(t.getKind(), M::Type::Kind::FIRST_STANDARD_TYPE,
                  M::Type::Kind::LAST_STANDARD_TYPE);
}

bool isa_eir_or_std_type(M::Type t) {
  return isa_eir_type(t) || isa_std_type(t);
}

M::Type EirDialect::parseType(M::DialectAsmParser &parser) const {
  L::StringRef typeNameLit;
  if (M::failed(parser.parseKeyword(&typeNameLit)))
    return {};

  auto loc = parser.getNameLoc();
  auto context = getContext();
  // `term`
  if (typeNameLit == "term")
    return TermType::get(context);
  // `atom`
  if (typeNameLit == "atom")
    return AtomType::get(context);
  // `boolean`
  if (typeNameLit == "boolean")
    return BooleanType::get(context);
  // `fixnum`
  if (typeNameLit == "boolean")
    return BooleanType::get(context);
  // `bigint`
  if (typeNameLit == "boolean")
    return BooleanType::get(context);
  // `float`
  if (typeNameLit == "float")
    return FloatType::get(context);
  // `float_packed`
  if (typeNameLit == "float_packed")
    return PackedFloatType::get(context);
  // `nil`
  if (typeNameLit == "nil")
    return NilType::get(context);
  // `cons`
  if (typeNameLit == "cons")
    return ConsType::get(context);
  // See parseTuple
  if (typeNameLit == "tuple")
    return parseTuple(context, parser);
  // `map`
  if (typeNameLit == "map")
    return MapType::get(context);
  // See parseClosure
  if (typeNameLit == "closure")
    return parseClosure(context, parser);
  // `box` `<` type `>`
  if (typeNameLit == "box")
    return parseTypeSingleton<BoxType>(context, parser);

  parser.emitError(loc, "unknown EIR type " + typeNameLit);
  return {};
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void printShape(llvm::raw_ostream &os, const Shape &shape,
                M::DialectAsmPrinter &p) {
  if (!shape.isKnown()) {
    os << '*';
  }
  TypeList elementTypes = shape.getElementTypes();
  // Single element is always uniform
  unsigned size = elementTypes.size();
  if (size == 0) {
    os << "0x?";
    return;
  }
  if (size == 1) {
    os << "1x";
    p.printType(elementTypes[0]);
    return;
  }
  // Check for uniformity to print more compact representation
  M::Type ty = elementTypes[0];
  bool uniform = true;
  for (auto it = elementTypes.begin(); it != elementTypes.end(); it++) {
    if (ty != *it) {
      uniform = false;
      break;
    }
  }
  if (uniform) {
    os << size << 'x';
    p.printType(ty);
    return;
  }

  unsigned i = 0;
  for (auto &elementTy : elementTypes) {
    i++;
    p.printType(elementTy);
    if (i < size) {
      os << ", ";
    }
  }
}

void EirDialect::printType(M::Type ty, M::DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  switch (ty.getKind()) {
  case EirTypes::Atom:
    os << "atom";
    break;
  case EirTypes::Boolean:
    os << "boolean";
    break;
  case EirTypes::Fixnum:
    os << "fixnum";
    break;
  case EirTypes::BigInt:
    os << "bigint";
    break;
  case EirTypes::Float:
    os << "float";
    break;
  case EirTypes::FloatPacked:
    os << "float_packed";
    break;
  case EirTypes::Nil:
    os << "nil";
    break;
  case EirTypes::Cons:
    os << "cons";
    break;
  case EirTypes::Tuple: {
    auto type = ty.cast<TupleType>();
    os << "tuple<";
    printShape(os, type.getShape(), p);
    os << '>';
  } break;
  case EirTypes::Map:
    os << "map";
    break;
  case EirTypes::Closure: {
    auto type = ty.cast<ClosureType>();
    os << "closure<";
    p.printType(type.getFnType());
    if (!type.hasStaticShape()) {
      os << ", *";
    } else {
      os << "[ ";
      printShape(os, type.getShape(), p);
      os << " ]";
    }
    os << '>';
  } break;
  case EirTypes::Box: {
    auto type = ty.cast<BoxType>();
    os << "box<";
    p.printType(type.getEleTy());
    os << '>';
  } break;
  default:
    llvm_unreachable("unhandled EIR type");
  }
}

} // namespace eir
