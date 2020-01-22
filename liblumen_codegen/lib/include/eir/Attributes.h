#ifndef EIR_ATTRIBUTES_H
#define EIR_ATTRIBUTES_H

#include "lumen/LLVM.h"

#include "mlir/IR/Attributes.h"

namespace M = mlir;
namespace L = llvm;

namespace eir {

namespace detail {
struct AtomAttributeStorage;
struct BinaryAttributeStorage;
struct SeqAttributeStorage;
}

enum class EirAttributes {
  Atom = M::Attribute::FIRST_EIR_ATTR,
  Binary,
  Seq,
};

class AtomAttr : public M::Attribute::AttrBase<AtomAttr, M::Attribute,
                                               detail::AtomAttributeStorage> {
public:
  using Base::Base;
  using ValueType = L::APInt;

  static AtomAttr get(M::MLIRContext *context, L::StringRef value, L::APInt id);
  static AtomAttr get(L::StringRef value, L::APInt id, M::Type type);

  L::APInt getValue() const;
  L::StringRef getStringValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(EirAttributes::Atom);
  }
};

class BinaryAttr : public M::Attribute::AttrBase<BinaryAttr, M::Attribute,
                                              detail::BinaryAttributeStorage> {
public:
  using Base::Base;
  using ValueType = L::ArrayRef<char>;

  /// Get an instance of a BinaryAttr with the given string, header, flags, and pointer width
  static BinaryAttr get(M::MLIRContext *context, L::ArrayRef<char> bytes,
                        uint64_t header, uint64_t flags, unsigned width);

  static BinaryAttr get(L::ArrayRef<char> bytes, uint64_t header, uint64_t flags, unsigned width,
                        M::Type type);

  L::ArrayRef<char> getValue() const;
  L::APInt &getHeader() const;
  L::APInt &getFlags() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(EirAttributes::Binary);
  }
};

/// Seq attributes are lists of other attributes. Used to represent
/// list and tuple constant element lists which require a type to distinguish them
class SeqAttr : public M::Attribute::AttrBase<SeqAttr, M::Attribute,
                                            detail::SeqAttributeStorage> {
public:
  using Base::Base;
  using ValueType = L::ArrayRef<M::Attribute>;

  static SeqAttr get(L::ArrayRef<M::Attribute> value, M::Type type);

  L::ArrayRef<M::Attribute> getValue() const;

  /// Support range iteration.
  using iterator = L::ArrayRef<M::Attribute>::iterator;
  iterator begin() const { return getValue().begin(); }
  iterator end() const { return getValue().end(); }
  size_t size() const { return getValue().size(); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == static_cast<unsigned>(EirAttributes::Seq);
  }
};

}

#endif // EIR_ATTRIBUTES_H
