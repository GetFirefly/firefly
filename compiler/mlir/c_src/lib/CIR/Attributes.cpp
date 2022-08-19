#include "CIR/Attributes.h"
#include "CIR/Dialect.h"
#include "CIR/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
/// AtomRef
//===----------------------------------------------------------------------===//

llvm::StringRef AtomRef::strref() const { return {data, len}; }

llvm::hash_code mlir::cir::hash_value(const AtomRef &atom) {
  return llvm::hash_code(atom.symbol);
}

template <> struct FieldParser<AtomRef> {
  static FailureOr<AtomRef> parse(AsmParser &parser) {
    int symbol;
    std::string value;
    StringRef kw1, kw2;

    if (parser.parseKeyword(kw1))
      return failure();
    if (kw1 == "id") {
      if (parser.parseInteger(symbol) || parser.parseComma() ||
          parser.parseKeyword(kw2) || kw2 != "value" || parser.parseEqual() ||
          parser.parseString(&value))
        return failure();
    } else if (kw1 == "value") {
      if (parser.parseString(&value) || parser.parseComma() ||
          parser.parseKeyword(kw2) || kw2 != "id" || parser.parseEqual() ||
          parser.parseInteger(symbol))
        return failure();
    } else {
      return failure();
    }

    size_t len = value.size();
    const char *data = strdup(value.c_str());
    return AtomRef{static_cast<size_t>(symbol), data, len};
  }
};

//===----------------------------------------------------------------------===//
/// BigIntRef
//===----------------------------------------------------------------------===//

llvm::StringRef BigIntRef::data() const { return {digits, len}; }

llvm::hash_code mlir::cir::hash_value(const BigIntRef &bigint) {
  auto data = bigint.data();
  return llvm::hash_combine((unsigned)bigint.sign,
                            llvm::hash_combine_range(data.begin(), data.end()));
}

template <> struct FieldParser<BigIntRef> {
  static FailureOr<BigIntRef> parse(AsmParser &parser) {
    Sign sign = SignNoSign;
    StringRef kw1, kw2;

    if (parser.parseKeyword(kw1))
      return failure();
    if (kw1 == "sign") {
      if (parser.parseEqual() || parser.parseKeyword(kw2) ||
          parser.parseComma() || parser.parseKeyword(kw1))
        return failure();
      if (kw2 == "minus")
        sign = SignMinus;
      else if (kw2 == "plus")
        sign = SignPlus;
    }

    if (kw1 != "digits" || parser.parseEqual())
      return failure();

    SmallVector<int8_t, 4> digits;
    auto parseElt = [&] {
      int8_t digit;
      if (parser.parseInteger(digit))
        return failure();
      digits.push_back(digit);
      return success();
    };

    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseElt,
                                       " in big integer attribute"))
      return failure();

    size_t len = digits.size();
    char *data = nullptr;
    if (len > 0) {
      data = static_cast<char *>(aligned_alloc(16, digits.size_in_bytes()));
      std::memcpy(data, digits.data(), len);
    }
    return BigIntRef{sign, data, len};
  }
};

//===----------------------------------------------------------------------===//
/// BinaryEntrySpecifier
//===----------------------------------------------------------------------===//

llvm::hash_code mlir::cir::hash_value(const BinaryEntrySpecifier &spec) {
  uint64_t value = 0;
  value |= ((uint64_t)(spec.tag)) << 32;
  value |= (uint64_t)(spec.data.raw);
  return llvm::hash_code(value);
}

static BinaryEntrySpecifier
makeBinaryEntrySpecifierFromParts(BinaryEntrySpecifierType ty,
                                  Endianness endianness, uint8_t unit,
                                  bool isSigned) {
  BinaryEntrySpecifier spec;
  spec.tag = 0;
  spec.data.raw = 0;

  switch (ty) {
  case BinaryEntrySpecifierType::Integer:
    spec.data.integer.isSigned = (uint8_t)(isSigned);
    spec.data.integer.unit = unit;
    spec.data.integer.endianness = static_cast<uint8_t>(endianness);
    break;
  case BinaryEntrySpecifierType::Float:
    spec.data.flt.unit = unit;
    spec.data.flt.endianness = static_cast<uint8_t>(endianness);
    break;
  case BinaryEntrySpecifierType::Bytes:
    spec.data.bytes.unit = unit;
    break;
  case BinaryEntrySpecifierType::Utf8:
    break;
  case BinaryEntrySpecifierType::Utf16:
  case BinaryEntrySpecifierType::Utf32:
    spec.data.utfWide.endianness = static_cast<uint8_t>(endianness);
    break;
  }
  spec.tag = static_cast<uint32_t>(ty);

  return spec;
}

template <> struct FieldParser<BinaryEntrySpecifier> {
  static FailureOr<BinaryEntrySpecifier> parse(AsmParser &parser) {
    BinaryEntrySpecifierType ty;
    Endianness endianness = Endianness::Big;
    uint8_t unit = 1;
    bool isSigned = false;
    StringRef kw1, kw2;

    // Type always comes first
    if (parser.parseKeyword(kw1))
      return failure();
    if (kw1 == "integer")
      ty = BinaryEntrySpecifierType::Integer;
    else if (kw1 == "float")
      ty = BinaryEntrySpecifierType::Float;
    else if (kw1 == "bytes")
      ty = BinaryEntrySpecifierType::Bytes;
    else if (kw1 == "utf8")
      ty = BinaryEntrySpecifierType::Utf8;
    else if (kw1 == "utf16")
      ty = BinaryEntrySpecifierType::Utf16;
    else if (kw1 == "utf32")
      ty = BinaryEntrySpecifierType::Utf32;
    else
      return failure();

    // If there are no more components, there will be no comma
    if (parser.parseComma())
      return makeBinaryEntrySpecifierFromParts(ty, endianness, unit, isSigned);

    // A keyword always comes next
    if (parser.parseKeyword(kw2))
      return failure();

    // If this is the unit, it is always the last component and must always be
    // followed by `= integer`
    if (kw2 == "unit") {
      if (parser.parseEqual() || parser.parseInteger(unit))
        return failure();
      return makeBinaryEntrySpecifierFromParts(ty, endianness, unit, isSigned);
    }

    // Otherwise it must be signed or endianness, and if it is signed, then
    // endianness and unit are present later
    if (kw2 == "signed") {
      isSigned = true;
      if (parser.parseComma() || parser.parseKeyword(kw1) ||
          parser.parseComma() || parser.parseKeyword(kw2) || kw2 != "unit" ||
          parser.parseEqual() || parser.parseInteger(unit))
        return failure();
      if (kw1 == "big") {
        endianness = Endianness::Big;
      } else if (kw1 == "little") {
        endianness = Endianness::Little;
      } else if (kw1 == "native") {
        endianness = Endianness::Native;
      } else
        return failure();
      return makeBinaryEntrySpecifierFromParts(ty, endianness, unit, isSigned);
    }

    // Endianness or unit will come next
    if (kw2 == "big") {
      endianness = Endianness::Big;
    } else if (kw2 == "little") {
      endianness = Endianness::Little;
    } else if (kw2 == "native") {
      endianness = Endianness::Native;
    } else if (kw2 != "unit") {
      // If the last keyword is not unit, this is invalid
      return failure();
    } else {
      // If there is no comma, there is no trailing unit
      if (parser.parseComma())
        return makeBinaryEntrySpecifierFromParts(ty, endianness, unit,
                                                 isSigned);
      // Unit must be next
      if (parser.parseKeyword(kw2) || kw2 != "unit" || parser.parseEqual() ||
          parser.parseInteger(unit))
        return failure();
    }

    return makeBinaryEntrySpecifierFromParts(ty, endianness, unit, isSigned);
  }
};

//===----------------------------------------------------------------------===//
/// Endianness
//===----------------------------------------------------------------------===//

template <> struct FieldParser<Endianness> {
  static FailureOr<Endianness> parse(AsmParser &parser) {
    StringRef kw;
    if (parser.parseKeyword(kw))
      return failure();

    if (kw == "big")
      return Endianness::Big;
    else if (kw == "little")
      return Endianness::Little;
    else if (kw == "native")
      return Endianness::Native;
    else
      return failure();
  }
};

//===----------------------------------------------------------------------===//
/// IsizeAttr
//===----------------------------------------------------------------------===//

template <> struct FieldParser<APInt> {
  static FailureOr<APInt> parse(AsmParser &parser) {
    APInt value;
    if (parser.parseInteger(value))
      return failure();

    return value;
  }
};

int64_t IsizeAttr::getInt() const { return getValue().getSExtValue(); }

//===----------------------------------------------------------------------===//
/// CIRFloatAttr
//===----------------------------------------------------------------------===//

template <> struct FieldParser<APFloat> {
  static FailureOr<APFloat> parse(AsmParser &parser) {
    double value;
    if (parser.parseFloat(value))
      return failure();

    return APFloat(value);
  }
};

double CIRFloatAttr::getValueAsDouble() const {
  return getValueAsDouble(getValue());
}

double CIRFloatAttr::getValueAsDouble(APFloat value) {
  if (&value.getSemantics() != &APFloat::IEEEdouble()) {
    bool losesInfo = false;
    value.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
  }
  return value.convertToDouble();
}

//===----------------------------------------------------------------------===//
/// Tablegen
//===----------------------------------------------------------------------===//

#include "CIR/CIREnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "CIR/CIRAttributes.cpp.inc"

void CIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "CIR/CIRAttributes.cpp.inc"
      >();
}
