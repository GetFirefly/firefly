#ifndef EIR_ATTRIBUTES_H
#define EIR_ATTRIBUTES_H

#include "llvm/ADT/APInt.h"
#include "mlir/IR/Attributes.h"

namespace llvm {
class APInt;
}

namespace mlir {
class MLIRContext;
class Type;
}  // namespace mlir

using ::llvm::APFloat;
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::StringRef;
using ::mlir::Attribute;
using ::mlir::Location;
using ::mlir::MLIRContext;
using ::mlir::Type;

namespace lumen {
namespace eir {

namespace detail {
struct AtomAttributeStorage;
struct APIntAttributeStorage;
struct APFloatAttributeStorage;
struct BinaryAttributeStorage;
struct SeqAttributeStorage;
}  // namespace detail

namespace AttributeKind {
enum Kind {
    Atom,
    Int,
    Float,
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
    static AtomAttr getChecked(APInt id, StringRef value, Location loc);

    static StringRef getAttrName() { return "atom"; }

    APInt &getValue() const;
    StringRef getStringValue() const;
};

class APIntAttr : public Attribute::AttrBase<APIntAttr, Attribute,
                                             detail::APIntAttributeStorage> {
   public:
    using Base::Base;
    using ValueType = APInt;

    static APIntAttr get(MLIRContext *context, APInt value);
    static APIntAttr get(MLIRContext *context, Type type, APInt value);
    static APIntAttr get(MLIRContext *context, StringRef value,
                         unsigned numBits);

    static APIntAttr getChecked(APInt value, Location loc);
    static APIntAttr getChecked(Type type, APInt value, Location loc);
    static APIntAttr getChecked(StringRef value, unsigned numBits,
                                Location loc);

    static StringRef getAttrName() { return "int"; }

    APInt &getValue() const;
    std::string getValueAsString() const;
    std::string getHash() const;
};

class APFloatAttr
    : public Attribute::AttrBase<APFloatAttr, Attribute,
                                 detail::APFloatAttributeStorage> {
   public:
    using Base::Base;
    using ValueType = APFloat;

    static APFloatAttr get(MLIRContext *context, APFloat value);
    static APFloatAttr getChecked(APFloat value, Location loc);

    static StringRef getAttrName() { return "float"; }

    APFloat &getValue() const;
};

class BinaryAttr : public Attribute::AttrBase<BinaryAttr, Attribute,
                                              detail::BinaryAttributeStorage> {
   public:
    using Base::Base;
    using ValueType = StringRef;

    static StringRef getAttrName() { return "binary"; }
    /// Get an instance of a BinaryAttr with the given string, header, flags,
    /// and pointer width
    static BinaryAttr get(MLIRContext *context, StringRef bytes,
                          uint64_t header, uint64_t flags);

    static BinaryAttr get(Type type, StringRef bytes, uint64_t header,
                          uint64_t flags);

    static BinaryAttr getChecked(StringRef bytes, uint64_t header,
                                 uint64_t flags, Location loc);
    static BinaryAttr getChecked(Type type, StringRef bytes, uint64_t header,
                                 uint64_t flags, Location loc);

    StringRef getValue() const;
    std::string getHash() const;
    APInt &getHeader() const;
    APInt &getFlags() const;
    bool isPrintable() const;
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
    static SeqAttr getChecked(Type type, ArrayRef<Attribute> value,
                              Location loc);

    ArrayRef<Attribute> &getValue() const;

    /// Support range iteration.
    using iterator = ArrayRef<Attribute>::iterator;
    using reverse_iterator = ArrayRef<Attribute>::reverse_iterator;
    iterator begin() const { return getValue().begin(); }
    iterator end() const { return getValue().end(); }
    reverse_iterator rbegin() const { return getValue().rbegin(); }
    reverse_iterator rend() const { return getValue().rend(); }
    size_t size() const { return getValue().size(); }
};

}  // namespace eir
}  // namespace lumen

#endif  // EIR_ATTRIBUTES_H
