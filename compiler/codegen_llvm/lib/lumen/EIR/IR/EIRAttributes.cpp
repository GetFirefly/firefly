#include "lumen/EIR/IR/EIRAttributes.h"
#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIRTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA1.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"

using ::llvm::hash_combine;
using ::mlir::AttributeStorage;
using ::mlir::AttributeStorageAllocator;
using ::mlir::DialectAsmPrinter;

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
      : AttributeStorage(type), id(std::move(id)), name(name) {}

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
  static AtomAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                         const KeyTy &key) {
    auto type = std::get<Type>(key);
    auto id = new (allocator.allocate<APInt>()) APInt(std::get<APInt>(key));
    auto name = allocator.copyInto(std::get<StringRef>(key));
    return new (allocator.allocate<AtomAttributeStorage>())
        AtomAttributeStorage(type, *id, name);
  }

  APInt id;
  StringRef name;
};  // struct AtomAttr
}  // namespace detail
}  // namespace eir
}  // namespace lumen

AtomAttr AtomAttr::get(MLIRContext *context, APInt id, StringRef name) {
  unsigned kind = static_cast<unsigned>(AttributeKind::Atom);
  return Base::get(context, kind, AtomType::get(context), id, name);
}

APInt &AtomAttr::getValue() const { return getImpl()->id; }
StringRef AtomAttr::getStringValue() const { return getImpl()->name; }

//===----------------------------------------------------------------------===//
// APIntAttr
//===----------------------------------------------------------------------===//

/// An attribute representing an fixed-width integer literal value.
namespace lumen {
namespace eir {
namespace detail {
struct APIntAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Type, APInt, bool>;

  APIntAttributeStorage(Type type, APInt value, bool isSigned)
      : AttributeStorage(type), value(std::move(value)), isSigned(isSigned) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    auto keyType = std::get<Type>(key);
    auto keyValue = std::get<APInt>(key);
    return keyType == getType() && keyValue == value;
  }

  static unsigned hashKey(const KeyTy &key) {
    return hash_combine(hash_value(std::get<Type>(key)),
                        hash_value(std::get<APInt>(key)));
  }

  /// Construct a new storage instance.
  static APIntAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                         const KeyTy &key) {
    auto type = std::get<Type>(key);
    auto value = new (allocator.allocate<APInt>()) APInt(std::get<APInt>(key));
    auto isSigned = std::get<bool>(key);
    return new (allocator.allocate<APIntAttributeStorage>())
        APIntAttributeStorage(type, *value, isSigned);
  }

  APInt value;
  bool isSigned;
};  // struct APIntAttr
}  // namespace detail
}  // namespace eir
}  // namespace lumen

APIntAttr APIntAttr::get(MLIRContext *context, APInt value) {
  return APIntAttr::get(context, FixnumType::get(context), value);
}

APIntAttr APIntAttr::get(MLIRContext *context, Type type, APInt value) {
  auto numBits = value.getBitWidth();
  bool isSigned = value.isSignedIntN(numBits);
  unsigned kind = static_cast<unsigned>(AttributeKind::Int);
  return Base::get(context, kind, type, value, isSigned);
}

APIntAttr APIntAttr::get(MLIRContext *context, StringRef value, unsigned numBits) {
  APInt i(numBits, value, /*radix=*/10);
  bool isSigned = i.isSignedIntN(i.getBitWidth());
  unsigned kind = static_cast<unsigned>(AttributeKind::Int);
  return Base::get(context, kind, BigIntType::get(context), i, isSigned);
}

APInt &APIntAttr::getValue() const { return getImpl()->value; }

std::string APIntAttr::getValueAsString() const {
  auto value = getImpl()->value;
  bool isSigned = getImpl()->isSigned;
  return value.toString(10, /*signed=*/isSigned);
}

std::string APIntAttr::getHash() const {
  llvm::SHA1 hasher;
  StringRef bytes = getValueAsString();
  hasher.update(ArrayRef<uint8_t>((uint8_t *)const_cast<char *>(bytes.data()),
                                  bytes.size()));
  return llvm::toHex(hasher.result(), true);
}

//===----------------------------------------------------------------------===//
// APFloatAttr
//===----------------------------------------------------------------------===//

/// An attribute representing an fixed-width integer literal value.
namespace lumen {
namespace eir {
namespace detail {
struct APFloatAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Type, APFloat>;

  APFloatAttributeStorage(Type type, APFloat value)
      : AttributeStorage(type), value(std::move(value)) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    auto keyType = std::get<Type>(key);
    auto keyValue = std::get<APFloat>(key);
    return keyType == getType() && keyValue == value;
  }

  static unsigned hashKey(const KeyTy &key) {
    return hash_combine(hash_value(std::get<Type>(key)),
                        hash_value(std::get<APFloat>(key)));
  }

  /// Construct a new storage instance.
  static APFloatAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                          const KeyTy &key) {
    auto type = std::get<Type>(key);
    auto value = new (allocator.allocate<APFloat>()) APFloat(std::get<APFloat>(key));
    return new (allocator.allocate<APFloatAttributeStorage>())
        APFloatAttributeStorage(type, *value);
  }

  APFloat value;
};  // struct APFloatAttr
}  // namespace detail
}  // namespace eir
}  // namespace lumen

APFloatAttr APFloatAttr::get(MLIRContext *context, APFloat value) {
  unsigned kind = static_cast<unsigned>(AttributeKind::Float);
  return Base::get(context, kind, FloatType::get(context), value);
}

APFloat &APFloatAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// BinaryAttr
//===----------------------------------------------------------------------===//

/// An attribute representing a binary literal value.
namespace lumen {
namespace eir {
namespace detail {
struct BinaryAttributeStorage : public AttributeStorage {
  using KeyTy = std::tuple<Type, std::string, APInt, APInt>;

  BinaryAttributeStorage(Type type, StringRef bytes, APInt header, APInt flags)
      : AttributeStorage(type),
        value(bytes.data(), bytes.size()),
        header(std::move(header)),
        flags(std::move(flags)) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getType(), value, header, flags};
  }

  static unsigned hashKey(const KeyTy &key) {
    auto hashVal{hash_combine(std::get<Type>(key))};
    auto str = std::get<std::string>(key);
    return hash_combine(hashVal, hash_value(StringRef(str.data(), str.size())),
                        hash_value(std::get<2>(key)),
                        hash_value(std::get<3>(key)));
  }

  /// Construct a new storage instance.
  static BinaryAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                           const KeyTy &key) {
    auto type = std::get<Type>(key);
    auto bytes = allocator.copyInto(std::get<1>(key));
    auto header = new (allocator.allocate<APInt>()) APInt(std::get<2>(key));
    auto flags = new (allocator.allocate<APInt>()) APInt(std::get<3>(key));
    return new (allocator.allocate<BinaryAttributeStorage>())
        BinaryAttributeStorage(type, bytes, *header, *flags);
  }

  std::string value;
  APInt header;
  APInt flags;
};  // struct BinaryAttr
}  // namespace detail
}  // namespace eir
}  // namespace lumen

BinaryAttr BinaryAttr::get(MLIRContext *context, StringRef bytes,
                           uint64_t header, uint64_t flags) {
  return get(BinaryType::get(context), bytes, header, flags);
}

/// Get an instance of a BinaryAttr with the given string and Type.
BinaryAttr BinaryAttr::get(Type type, StringRef bytes, uint64_t header,
                           uint64_t flags) {
  APInt hi(64, header, /*signed=*/false);
  APInt fi(64, flags, /*signed=*/false);
  unsigned kind = static_cast<unsigned>(AttributeKind::Binary);
  return Base::get(type.getContext(), kind, type, bytes, hi, fi);
}

std::string BinaryAttr::getHash() const {
  llvm::SHA1 hasher;
  StringRef bytes = getValue();
  hasher.update(ArrayRef<uint8_t>((uint8_t *)const_cast<char *>(bytes.data()),
                                  bytes.size()));
  return llvm::toHex(hasher.result(), true);
}
StringRef BinaryAttr::getValue() const {
  auto impl = getImpl();
  return StringRef(impl->value.data(), impl->value.size());
}
APInt &BinaryAttr::getHeader() const { return getImpl()->header; }
APInt &BinaryAttr::getFlags() const { return getImpl()->flags; }
bool BinaryAttr::isPrintable() const {
  auto s = getValue();
  for (char c : s.bytes()) {
    if (!llvm::isPrint(c)) return false;
  }
  return true;
}

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
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getType(), value);
  }

  /// Construct a new storage instance.
  static SeqAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<SeqAttributeStorage>())
        SeqAttributeStorage(key.first, allocator.copyInto(key.second));
  }

  ArrayRef<Attribute> value;
};
}  // namespace detail
}  // namespace eir
}  // namespace lumen

SeqAttr SeqAttr::get(Type type, ArrayRef<Attribute> value) {
  unsigned kind = static_cast<unsigned>(AttributeKind::Seq);
  return Base::get(type.getContext(), kind, type, value);
}

ArrayRef<Attribute> &SeqAttr::getValue() const { return getImpl()->value; }
