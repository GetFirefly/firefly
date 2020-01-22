#include "eir/Attributes.h"
#include "eir/Types.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace M = mlir;
namespace L = llvm;

using llvm::StringRef;
using llvm::ArrayRef;

using namespace eir;

//===----------------------------------------------------------------------===//
// AtomAttr
//===----------------------------------------------------------------------===//

/// An attribute representing a binary literal value.
namespace eir {
namespace detail {
struct AtomAttributeStorage : public M::AttributeStorage {
  using KeyTy = std::tuple<StringRef, L::APInt, M::Type>;

  AtomAttributeStorage(StringRef stringValue, L::APInt value, M::Type type)
      : M::AttributeStorage(type),
        stringValue(stringValue),
        value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy{stringValue, value, getType()};
  }

  static unsigned hashKey(const KeyTy &key) {
    auto hashVal{L::hash_combine(std::get<M::Type>(key))};
    return L::hash_combine(hashVal,
                           hash_value(std::get<StringRef>(key)),
                           hash_value(std::get<L::APInt>(key)));
  }


  /// Construct a new storage instance.
  static AtomAttributeStorage *
  construct(M::AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto str = allocator.copyInto(std::get<StringRef>(key));
    auto value =
        new (allocator.allocate<L::APInt>()) L::APInt(std::get<L::APInt>(key));
    auto type = std::get<M::Type>(key);
    return new (allocator.allocate<AtomAttributeStorage>())
        AtomAttributeStorage(str, *value, type);
  }

  L::APInt value;
  StringRef stringValue;
}; // struct AtomAttr
} // namespace detail
} // namespace eir

AtomAttr AtomAttr::get(M::MLIRContext *context, StringRef stringValue, L::APInt value) {
  return get(stringValue, value, AtomType::get(context));
}

AtomAttr AtomAttr::get(StringRef stringValue, L::APInt value, M::Type type) {
  unsigned kind = static_cast<unsigned>(EirAttributes::Atom);
  return Base::get(type.getContext(), kind, stringValue, value);
}

L::APInt AtomAttr::getValue() const { return L::APInt(getImpl()->value); }
StringRef AtomAttr::getStringValue() const { return getImpl()->stringValue; }

//===----------------------------------------------------------------------===//
// BinaryAttr
//===----------------------------------------------------------------------===//

/// An attribute representing a binary literal value.
namespace eir {
namespace detail {
struct BinaryAttributeStorage : public M::AttributeStorage {
  using KeyTy = std::tuple<ArrayRef<char>, L::APInt, L::APInt, M::Type>;

  BinaryAttributeStorage(ArrayRef<char> value, L::APInt header, L::APInt flags, M::Type type)
      : M::AttributeStorage(type),
        value(value),
        header(std::move(header)),
        flags(std::move(flags)) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy{value, header, flags, getType()};
  }

  static unsigned hashKey(const KeyTy &key) {
    auto hashVal{L::hash_combine(std::get<M::Type>(key))};
    return L::hash_combine(hashVal,
                           hash_value(std::get<0>(key)),
                           hash_value(std::get<1>(key)),
                           hash_value(std::get<2>(key)));
  }


  /// Construct a new storage instance.
  static BinaryAttributeStorage *
  construct(M::AttributeStorageAllocator &allocator, const KeyTy &key) {
    auto bytes = allocator.copyInto(std::get<0>(key));
    auto header =
        new (allocator.allocate<L::APInt>()) L::APInt(std::get<1>(key));
    auto flags =
        new (allocator.allocate<L::APInt>()) L::APInt(std::get<2>(key));
    auto type = std::get<M::Type>(key);
    return new (allocator.allocate<BinaryAttributeStorage>())
        BinaryAttributeStorage(bytes, *header, *flags, type);
  }

  ArrayRef<char> value;
  L::APInt header;
  L::APInt flags;
}; // struct BinaryAttr
} // namespace detail
} // namespace eir

BinaryAttr BinaryAttr::get(M::MLIRContext *context, ArrayRef<char> bytes, uint64_t header, uint64_t flags, unsigned width) {
  return get(bytes, header, flags, width, BinaryType::get(context));
}

/// Get an instance of a BinaryAttr with the given string and Type.
BinaryAttr BinaryAttr::get(ArrayRef<char> bytes, uint64_t header, uint64_t flags, unsigned width, M::Type type) {
  L::APInt hi(width, header);
  L::APInt fi(width, flags);
  unsigned kind = static_cast<unsigned>(EirAttributes::Binary);
  return Base::get(type.getContext(), kind, bytes, hi, fi, type);
}

ArrayRef<char> BinaryAttr::getValue() const { return getImpl()->value; }
L::APInt &BinaryAttr::getHeader() const { return getImpl()->header; }
L::APInt &BinaryAttr::getFlags() const { return getImpl()->flags; }

//===----------------------------------------------------------------------===//
// SeqAttr
//===----------------------------------------------------------------------===//

///  An attribute representing an array of other attributes.
namespace eir {
namespace detail {
struct SeqAttributeStorage : public M::AttributeStorage {
  using KeyTy = std::pair<ArrayRef<M::Attribute>, M::Type>;

  SeqAttributeStorage(ArrayRef<M::Attribute> value, M::Type type)
      : M::AttributeStorage(type), value(value) {}

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == KeyTy(value, getType()); }

  /// Construct a new storage instance.
  static SeqAttributeStorage *construct(M::AttributeStorageAllocator &allocator,
                                        const KeyTy &key) {
    return new (allocator.allocate<SeqAttributeStorage>())
        SeqAttributeStorage(allocator.copyInto(key.first), key.second);
  }

  ArrayRef<M::Attribute> value;
};
}
}

SeqAttr SeqAttr::get(ArrayRef<M::Attribute> value, M::Type type) {
  unsigned kind = static_cast<int>(EirAttributes::Seq);
  return Base::get(type.getContext(), kind, value, type);
}

ArrayRef<M::Attribute> SeqAttr::getValue() const { return getImpl()->value; }
