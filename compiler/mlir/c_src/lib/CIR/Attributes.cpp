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

#define GET_ATTRDEF_CLASSES
#include "CIR/CIRAttributes.cpp.inc"

void CIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "CIR/CIRAttributes.cpp.inc"
      >();
}
