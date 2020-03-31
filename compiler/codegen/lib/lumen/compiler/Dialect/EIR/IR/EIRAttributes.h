#ifndef EIR_ATTRIBUTES_H
#define EIR_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APInt.h"

namespace llvm {
class APInt;
}

namespace mlir {
class MLIRContext;
class Type;
}  // namespace mlir

using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::StringRef;
using ::mlir::Attribute;
using ::mlir::MLIRContext;
using ::mlir::Type;

namespace lumen {
namespace eir {

namespace detail {
struct AtomAttributeStorage;
struct BinaryAttributeStorage;
struct SeqAttributeStorage;
}  // namespace detail

namespace AttributeKind {
enum Kind {
  Atom = Attribute::FIRST_EIR_ATTR,
  Binary,
  Seq,
};
}  // namespace AttributeKind

class AtomAttr : public Attribute::AttrBase<AtomAttr, Attribute,
                                            detail::AtomAttributeStorage> {
 public:
  using Base::Base;
  using ValueType = APInt;

  static AtomAttr get(MLIRContext *context, APInt id, StringRef value = "");

  static StringRef getAttrName() { return "atom"; }

  APInt &getValue() const;
  StringRef getStringValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(AttributeKind::Atom);
  }
};

class BinaryAttr : public Attribute::AttrBase<BinaryAttr, Attribute,
                                              detail::BinaryAttributeStorage> {
 public:
  using Base::Base;
  using ValueType = StringRef;

  static StringRef getAttrName() { return "binary"; }
  /// Get an instance of a BinaryAttr with the given string, header, flags, and
  /// pointer width
  static BinaryAttr get(MLIRContext *context, StringRef bytes, uint64_t header,
                        uint64_t flags);

  static BinaryAttr get(Type type, StringRef bytes, uint64_t header,
                        uint64_t flags);

  StringRef getValue() const;
  std::string getHash() const;
  APInt &getHeader() const;
  APInt &getFlags() const;
  bool isPrintable() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(AttributeKind::Binary);
  }
};

/// Seq attributes are lists of other attributes. Used to represent
/// list and tuple constant element lists which require a type to distinguish
/// them
class SeqAttr : public Attribute::AttrBase<SeqAttr, Attribute,
                                           detail::SeqAttributeStorage> {
 public:
  using Base::Base;
  using ValueType = ArrayRef<Attribute>;

  static StringRef getAttrName() { return "seq"; }
  static SeqAttr get(Type type, ArrayRef<Attribute> value);

  ArrayRef<Attribute> &getValue() const;

  /// Support range iteration.
  using iterator = ArrayRef<Attribute>::iterator;
  using reverse_iterator = ArrayRef<Attribute>::reverse_iterator;
  iterator begin() const { return getValue().begin(); }
  iterator end() const { return getValue().end(); }
  reverse_iterator rbegin() const { return getValue().rbegin(); }
  reverse_iterator rend() const { return getValue().rend(); }
  size_t size() const { return getValue().size(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(AttributeKind::Seq);
  }
};

}  // namespace eir
}  // namespace lumen

#endif  // EIR_ATTRIBUTES_H
