#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using ::mlir::AttributeStorage;
using ::mlir::AttributeStorageAllocator;
using ::mlir::DialectAsmPrinter;
using ::llvm::hash_combine;

using namespace lumen;
using namespace lumen::eir;

//===----------------------------------------------------------------------===//
// AtomAttr
//===----------------------------------------------------------------------===//

/// An attribute representing a binary literal value.
namespace lumen {
namespace eir {
namespace detail {
struct AtomAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Type, APInt, StringRef>;

  AtomAttributeStorage(Type type, APInt id, StringRef name = "")
      : AttributeStorage(type),
        id(std::move(id)),
        name(name) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    auto keyType = std::get<Type>(key);
    auto keyId = std::get<APInt>(key);
    return keyType == getType() && keyId == id;
  }

  static unsigned hashKey(const KeyTy &key) {
    return hash_combine(hash_value(std::get<Type>(key)),
                        hash_value(std::get<APInt>(key)));
  }


  /// Construct a new storage instance.
  static AtomAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto type = std::get<Type>(key);
    auto id = new (allocator.allocate<APInt>()) APInt(std::get<APInt>(key));
    auto name = allocator.copyInto(std::get<StringRef>(key));
    return new (allocator.allocate<AtomAttributeStorage>())
        AtomAttributeStorage(type, *id, name);
  }

  APInt id;
  StringRef name;
}; // struct AtomAttr
} // namespace detail
} // namespace eir
} // namespace lumen

AtomAttr AtomAttr::get(MLIRContext *context, APInt id, StringRef name) {
  unsigned kind = static_cast<unsigned>(AttributeKind::Atom);
  return Base::get(context, kind, AtomType::get(context), id, name);
}

APInt &AtomAttr::getValue() const { return getImpl()->id; }
StringRef AtomAttr::getStringValue() const { return getImpl()->name; }

//===----------------------------------------------------------------------===//
// BinaryAttr
//===----------------------------------------------------------------------===//

/// An attribute representing a binary literal value.
namespace lumen {
namespace eir {
namespace detail {
struct BinaryAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Type, ArrayRef<char>, APInt, APInt>;

  BinaryAttributeStorage(Type type, ArrayRef<char> value, APInt header, APInt flags)
      : AttributeStorage(type),
        value(value),
        header(std::move(header)),
        flags(std::move(flags)) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getType(), value, header, flags};
  }

  static unsigned hashKey(const KeyTy &key) {
    auto hashVal{hash_combine(std::get<Type>(key))};
    return hash_combine(hashVal,
                        hash_value(std::get<1>(key)),
                        hash_value(std::get<2>(key)),
                        hash_value(std::get<3>(key)));
  }


  /// Construct a new storage instance.
  static BinaryAttributeStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto type = std::get<Type>(key);
    auto bytes = allocator.copyInto(std::get<1>(key));
    auto header =
        new (allocator.allocate<APInt>()) APInt(std::get<2>(key));
    auto flags =
        new (allocator.allocate<APInt>()) APInt(std::get<3>(key));
    return new (allocator.allocate<BinaryAttributeStorage>())
        BinaryAttributeStorage(type, bytes, *header, *flags);
  }

  ArrayRef<char> value;
  APInt header;
  APInt flags;
}; // struct BinaryAttr
} // namespace detail
} // namespace eir
} // namespace lumen

BinaryAttr BinaryAttr::get(MLIRContext *context, ArrayRef<char> bytes, uint64_t header, uint64_t flags) {
  return get(BinaryType::get(context), bytes, header, flags);
}

/// Get an instance of a BinaryAttr with the given string and Type.
BinaryAttr BinaryAttr::get(Type type, ArrayRef<char> bytes, uint64_t header, uint64_t flags) {
  APInt hi(64, header, /*signed=*/false);
  APInt fi(64, flags, /*signed=*/false);
  unsigned kind = static_cast<unsigned>(AttributeKind::Binary);
  return Base::get(type.getContext(), kind, type, bytes, hi, fi);
}

ArrayRef<char> BinaryAttr::getValue() const { return getImpl()->value; }
APInt &BinaryAttr::getHeader() const { return getImpl()->header; }
APInt &BinaryAttr::getFlags() const { return getImpl()->flags; }

//===----------------------------------------------------------------------===//
// SeqAttr
//===----------------------------------------------------------------------===//

///  An attribute representing an array of other attributes.
namespace lumen {
namespace eir {
namespace detail {
struct SeqAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<Type, ArrayRef<Attribute>>;

  SeqAttributeStorage(Type type, ArrayRef<Attribute> value)
      : AttributeStorage(type), value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == KeyTy(getType(), value); }

  /// Construct a new storage instance.
  static SeqAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<SeqAttributeStorage>())
        SeqAttributeStorage(key.first, allocator.copyInto(key.second));
  }

  ArrayRef<Attribute> value;
};
}
} // namespace eir
} // namespace lumen

SeqAttr SeqAttr::get(Type type, ArrayRef<Attribute> value) {
  unsigned kind = static_cast<unsigned>(AttributeKind::Seq);
  return Base::get(type.getContext(), kind, type, value);
}

ArrayRef<Attribute> &SeqAttr::getValue() const { return getImpl()->value; }
