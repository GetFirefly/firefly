#include "lumen/EIR/IR/EIRAttributes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA1.h"
#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIRTypes.h"
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
      : AttributeStorage(type), type(type), id(std::move(id)), name(name) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    auto keyType = std::get<Type>(key);
    auto keyId = std::get<APInt>(key);
    auto idBits = id.getBitWidth();
    auto keyIdBits = keyId.getBitWidth();
    if (idBits == keyIdBits) {
      return keyType == getType() && keyId == id;
    } else if (idBits < keyIdBits) {
      APInt temp(id);
      temp.zext(keyIdBits);
      return keyType == getType() && keyId == temp;
    } else {
      APInt temp(keyId);
      temp.zext(idBits);
      return keyType == getType() && id == temp;
    }
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_combine(hash_value(std::get<Type>(key)),
                        hash_value(std::get<APInt>(key)));
  }

  static KeyTy getKey(Type type, APInt value, StringRef name) {
    return KeyTy{type, value, name};
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

  Type type;
  APInt id;
  StringRef name;
};  // struct AtomAttr
}  // namespace detail
}  // namespace eir
}  // namespace lumen

AtomAttr AtomAttr::get(MLIRContext *context, APInt id, StringRef name) {
  return Base::get(context, AtomType::get(context), id, name);
}

AtomAttr AtomAttr::getChecked(APInt id, StringRef name, Location loc) {
  return Base::getChecked(loc, AtomType::get(loc.getContext()), id, name);
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
  using KeyTy = std::tuple<Type, APInt>;

  APIntAttributeStorage(Type type, APInt value)
      : AttributeStorage(type), type(type), value(std::move(value)) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    auto keyType = std::get<Type>(key);
    auto keyValue = std::get<APInt>(key);
    auto valueBits = value.getBitWidth();
    auto keyValueBits = keyValue.getBitWidth();
    if (valueBits == keyValueBits) {
      return keyType == getType() && keyValue == value;
    } else if (valueBits < keyValueBits) {
      APInt temp(value);
      temp.sext(keyValueBits);
      return keyType == getType() && keyValue == temp;
    } else {
      APInt temp(keyValue);
      temp.sext(valueBits);
      return keyType == getType() && value == temp;
    }
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_combine(hash_value(std::get<Type>(key)),
                        hash_value(std::get<APInt>(key)));
  }

  static KeyTy getKey(Type type, APInt value) { return KeyTy{type, value}; }

  /// Construct a new storage instance.
  static APIntAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                          const KeyTy &key) {
    auto type = std::get<Type>(key);
    auto value = new (allocator.allocate<APInt>()) APInt(std::get<APInt>(key));
    return new (allocator.allocate<APIntAttributeStorage>())
        APIntAttributeStorage(type, *value);
  }

  Type type;
  APInt value;
};  // struct APIntAttr
}  // namespace detail
}  // namespace eir
}  // namespace lumen

APIntAttr APIntAttr::get(MLIRContext *context, APInt value) {
  return APIntAttr::get(context, FixnumType::get(context), value);
}

APIntAttr APIntAttr::get(MLIRContext *context, Type type, APInt value) {
  return Base::get(context, type, value);
}

APIntAttr APIntAttr::get(MLIRContext *context, StringRef value,
                         unsigned numBits) {
  APInt i(numBits, value, /*radix=*/10);
  return Base::get(context, BigIntType::get(context), i);
}

APIntAttr APIntAttr::getChecked(APInt value, Location loc) {
  return APIntAttr::getChecked(FixnumType::get(loc.getContext()), value, loc);
}

APIntAttr APIntAttr::getChecked(Type type, APInt value, Location loc) {
  return Base::getChecked(loc, type, value);
}

APIntAttr APIntAttr::getChecked(StringRef value, unsigned numBits,
                                Location loc) {
  APInt i(numBits, value, /*radix=*/10);
  return Base::getChecked(loc, BigIntType::get(loc.getContext()), i);
}

APInt &APIntAttr::getValue() const { return getImpl()->value; }

std::string APIntAttr::getValueAsString() const {
  auto value = getImpl()->value;
  auto numBits = value.getBitWidth();
  bool isSigned = value.isSignedIntN(numBits);
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
      : AttributeStorage(type), type(type), value(std::move(value)) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    auto keyType = std::get<Type>(key);
    auto keyValue = std::get<APFloat>(key);
    return keyType == getType() && keyValue == value;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return hash_combine(hash_value(std::get<Type>(key)),
                        hash_value(std::get<APFloat>(key)));
  }

  static KeyTy getKey(Type type, APFloat value) { return KeyTy{type, value}; }

  /// Construct a new storage instance.
  static APFloatAttributeStorage *construct(
      AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto type = std::get<Type>(key);
    auto value =
        new (allocator.allocate<APFloat>()) APFloat(std::get<APFloat>(key));
    return new (allocator.allocate<APFloatAttributeStorage>())
        APFloatAttributeStorage(type, *value);
  }

  Type type;
  APFloat value;
};  // struct APFloatAttr
}  // namespace detail
}  // namespace eir
}  // namespace lumen

APFloatAttr APFloatAttr::get(MLIRContext *context, APFloat value) {
  return Base::get(context, FloatType::get(context), value);
}

APFloatAttr APFloatAttr::getChecked(APFloat value, Location loc) {
  return Base::getChecked(loc, FloatType::get(loc.getContext()), value);
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
        type(type),
        value(bytes.data(), bytes.size()),
        header(std::move(header)),
        flags(std::move(flags)) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy{type, value, header, flags};
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    auto hashVal{hash_combine(std::get<Type>(key))};
    auto str = std::get<std::string>(key);
    return hash_combine(hashVal, hash_value(StringRef(str.data(), str.size())),
                        hash_value(std::get<2>(key)),
                        hash_value(std::get<3>(key)));
  }

  static KeyTy getKey(Type type, std::string bytes, APInt header, APInt flags) {
    return KeyTy{type, bytes, header, flags};
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

  Type type;
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
  return Base::get(type.getContext(), type, bytes, hi, fi);
}

BinaryAttr BinaryAttr::getChecked(StringRef bytes, uint64_t header,
                                  uint64_t flags, Location loc) {
  return getChecked(BinaryType::get(loc.getContext()), bytes, header, flags,
                    loc);
}

/// Get an instance of a BinaryAttr with the given string and Type.
BinaryAttr BinaryAttr::getChecked(Type type, StringRef bytes, uint64_t header,
                                  uint64_t flags, Location loc) {
  APInt hi(64, header, /*signed=*/false);
  APInt fi(64, flags, /*signed=*/false);
  return Base::getChecked(loc, type, bytes, hi, fi);
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
      : AttributeStorage(type), type(type), value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == KeyTy(type, value); }

  static KeyTy getKey(Type type, ArrayRef<Attribute> value) {
    return KeyTy(type, value);
  }

  /// Construct a new storage instance.
  static SeqAttributeStorage *construct(AttributeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<SeqAttributeStorage>())
        SeqAttributeStorage(key.first, allocator.copyInto(key.second));
  }

  Type type;
  ArrayRef<Attribute> value;
};
}  // namespace detail
}  // namespace eir
}  // namespace lumen

SeqAttr SeqAttr::get(Type type, ArrayRef<Attribute> value) {
  return Base::get(type.getContext(), type, value);
}

SeqAttr SeqAttr::getChecked(Type type, ArrayRef<Attribute> value,
                            Location loc) {
  return Base::getChecked(loc, type, value);
}

ArrayRef<Attribute> &SeqAttr::getValue() const { return getImpl()->value; }
