#include "lumen/EIR/IR/EIRAttributes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA1.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"

#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIRTypes.h"

using ::mlir::AttributeStorage;
using ::mlir::AttributeStorageAllocator;
using ::mlir::DialectAsmPrinter;

using namespace lumen;
using namespace lumen::eir;

//===----------------------------------------------------------------------===//
/// Tablegen Attribute Definitions
//===----------------------------------------------------------------------===//

#include "lumen/EIR/IR/EIRAttributes.cpp.inc"
#include "lumen/EIR/IR/EIRStructs.cpp.inc"

//===----------------------------------------------------------------------===//
// APIntAttributeStorage
//===----------------------------------------------------------------------===//

/// An attribute representing an fixed-width integer literal value.
namespace lumen {
namespace eir {
namespace detail {
struct APIntAttributeStorage : public AttributeStorage {
    using KeyTy = std::tuple<APInt>;

    APIntAttributeStorage(Type type, APInt value)
        : AttributeStorage(type), value(std::move(value)) {}

    /// Key equality function.
    bool operator==(const KeyTy &key) const {
        auto keyValue = std::get<APInt>(key);
        auto valueBits = value.getBitWidth();
        auto keyValueBits = keyValue.getBitWidth();
        if (valueBits == keyValueBits) {
            return keyType == getType() && keyValue == value;
        } else if (valueBits < keyValueBits) {
            APInt temp = value.sext(keyValueBits);
            return keyType == getType() && keyValue == temp;
        } else {
            APInt temp = keyValue.sext(valueBits);
            return keyType == getType() && value == temp;
        }
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_combine(std::get<Type>(key));
    }

    /// Construct a new storage instance.
    static APIntAttributeStorage *construct(
        AttributeStorageAllocator &allocator, const KeyTy &key) {
        auto type = std::get<Type>(key);
        auto value =
            new (allocator.allocate<APInt>()) APInt(std::get<APInt>(key));
        return new (allocator.allocate<APIntAttributeStorage>())
            APIntAttributeStorage(type, *value);
    }

    APInt value;
};  // struct APIntAttr
}  // namespace detail
}  // namespace eir
}  // namespace lumen

APIntAttr APIntAttr::get(MLIRContext *context, APInt value) {
    return APIntAttr::get(context, IntegerType::get(context), value);
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
    return APIntAttr::getChecked(IntegerType::get(loc.getContext()), value,
                                 loc);
}

APIntAttr APIntAttr::getChecked(Type type, APInt value, Location loc) {
    return Base::getChecked(loc, type, value);
}

APIntAttr APIntAttr::getChecked(StringRef value, unsigned numBits,
                                Location loc) {
    APInt i(numBits, value, /*radix=*/10);
    return Base::getChecked(loc, BigIntType::get(loc.getContext()), i);
}

APInt APIntAttr::getValue() const { return getImpl()->value; }

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

bool BinaryAttr::isPrintable() const {
    auto s = getValue();
    for (char c : s.bytes()) {
        if (!llvm::isPrint(c)) return false;
    }
    return true;
}
