#include "lumen/EIR/IR/EIRDialect.h"

#include "llvm/Support/Format.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

#include "lumen/EIR/IR/EIRAttributes.h"
#include "lumen/EIR/IR/EIROps.h"
#include "lumen/EIR/IR/EIRTypes.h"

using namespace lumen::eir;

using ::llvm::SmallString;
using ::mlir::Attribute;
using ::mlir::DialectAsmParser;
using ::mlir::DialectAsmPrinter;

/// Create an instance of the EIR dialect, owned by the context.
///
/// This is where EIR types, operations, and attributes are registered.
void eirDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "lumen/EIR/IR/EIROps.cpp.inc"
        >();

    addTypes<
#define EIR_TERM_KIND(Name, Val) ::lumen::eir::Name##Type,
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/EIR/IR/EIREncoding.h.inc"
#undef EIR_TERM_KIND
#undef FIRST_EIR_TERM_KIND
        // These types are not term types, so are initialized manually
        RefType, PtrType, TraceRefType, ReceiveRefType>();

    addAttributes<AtomAttr, APIntAttr, APFloatAttr, BinaryAttr, SeqAttr>();
}

Operation *eirDialect::materializeConstant(mlir::OpBuilder &builder,
                                           mlir::Attribute value,
                                           mlir::Type type,
                                           mlir::Location loc) {
    if (type.isa<BooleanType>() || type.isInteger(1)) {
        if (value.isa<mlir::BoolAttr>())
            return builder.create<ConstantBoolOp>(loc, type,
                                                  value.cast<mlir::BoolAttr>());

        if (auto atomAttr = value.dyn_cast_or_null<AtomAttr>()) {
            auto id = atomAttr.getValue().getLimitedValue();
            if (id == 0 || id == 1)
                return builder.create<ConstantBoolOp>(loc, type, id == 1);

            // Invalid use of non-boolean atom as boolean
            return nullptr;
        }

        if (auto intAttr = value.dyn_cast_or_null<mlir::IntegerAttr>()) {
            auto isTrue = intAttr.getValue().getLimitedValue() == 0 ? 0 : 1;
            auto name = isTrue ? "true" : "false";
            APInt id(64, isTrue ? 1 : 0, /*signed=*/false);
            return builder.create<ConstantAtomOp>(loc, id, name);
        }

        // Unable to materialize a constant of the requested type and value
        return nullptr;
    }

    if (type.isa<AtomType>()) {
        if (auto boolAttr = value.dyn_cast_or_null<mlir::BoolAttr>()) {
            auto isTrue = boolAttr.getValue();
            auto name = isTrue ? "true" : "false";
            APInt id(64, isTrue ? 1 : 0, /*signed=*/false);
            return builder.create<ConstantAtomOp>(loc, id, name);
        }

        if (value.isa<AtomAttr>())
            return builder.create<ConstantAtomOp>(loc, type, value);

        return nullptr;
    }

    if (type.isa<FixnumType>() || type.isa<BigIntType>()) {
        if (auto intAttr = value.dyn_cast_or_null<APIntAttr>()) {
            auto val = intAttr.getValue();
            if (val.getMinSignedBits() > 47)
                return builder.create<ConstantBigIntOp>(loc, val);
            else
                return builder.create<ConstantIntOp>(loc, val);
        }
        if (auto intAttr = value.dyn_cast_or_null<mlir::IntegerAttr>()) {
            auto val = intAttr.getValue();
            if (type.isa<FixnumType>())
                return builder.create<ConstantIntOp>(loc, val);
            else
                return builder.create<ConstantBigIntOp>(loc, val);
        }

        return nullptr;
    }

    if (type.isa<FloatType>() || type.isF64()) {
        if (value.isa<APFloatAttr>())
            return builder.create<ConstantFloatOp>(loc, type, value);

        if (auto fltAttr = value.dyn_cast_or_null<mlir::FloatAttr>()) {
            auto ctx = builder.getContext();
            return builder.create<ConstantFloatOp>(
                loc, type, APFloatAttr::get(ctx, fltAttr.getValue()));
        }

        return nullptr;
    }

    if (type.isa<NilType>()) return builder.create<ConstantNilOp>(loc, type);

    if (type.isa<NoneType>()) return builder.create<ConstantNoneOp>(loc, type);

    return nullptr;
}

Attribute parseAtomAttr(DialectAsmParser &parser) {
    assert(false && "EIR dialect parsing is not fully implemented");
}

Attribute parseAPIntAttr(DialectAsmParser &parser, Type type) {
    assert(false && "EIR dialect parsing is not fully implemented");
}

Attribute parseAPFloatAttr(DialectAsmParser &parser) {
    assert(false && "EIR dialect parsing is not fully implemented");
}

Attribute parseBinaryAttr(DialectAsmParser &parser, Type type) {
    assert(false && "EIR dialect parsing is not fully implemented");
}

Attribute parseSeqAttr(DialectAsmParser &parser, Type type) {
    assert(false && "EIR dialect parsing is not fully implemented");
}

Attribute eirDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
    // Parse the kind keyword first.
    StringRef attrKind;
    if (parser.parseKeyword(&attrKind)) return {};

    if (attrKind == AtomAttr::getAttrName()) return parseAtomAttr(parser);
    if (attrKind == APIntAttr::getAttrName())
        return parseAPIntAttr(parser, type);
    if (attrKind == APFloatAttr::getAttrName()) return parseAPFloatAttr(parser);
    if (attrKind == BinaryAttr::getAttrName())
        return parseBinaryAttr(parser, type);
    if (attrKind == SeqAttr::getAttrName()) return parseSeqAttr(parser, type);

    parser.emitError(parser.getNameLoc(), "unknown EIR attribute kind: ")
        << attrKind;
    return {};
}

void eirDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
    auto &os = p.getStream();
    if (auto atomAttr = attr.dyn_cast_or_null<AtomAttr>()) {
        os << AtomAttr::getAttrName() << '<';
        os << "{ id = " << atomAttr.getValue();
        auto name = atomAttr.getStringValue();
        if (name.size() > 0) {
            os << ", value = '" << name << "'";
        }
        os << " }>";
        return;
    } else if (auto iAttr = attr.dyn_cast_or_null<APIntAttr>()) {
        os << APIntAttr::getAttrName() << '<';
        os << "{ value = " << iAttr.getValue() << " }>";
        return;
    } else if (auto floatAttr = attr.dyn_cast_or_null<APFloatAttr>()) {
        os << APFloatAttr::getAttrName() << '<';
        os << "{ value = " << floatAttr.getValue().convertToDouble() << " }>";
        return;
    } else if (auto binAttr = attr.dyn_cast_or_null<BinaryAttr>()) {
        os << BinaryAttr::getAttrName();
        os << "<{ value = ";
        if (binAttr.isPrintable()) {
            auto s = binAttr.getValue();
            os << '"' << s << '"';
        } else {
            auto bin = binAttr.getValue();
            auto size = bin.size();
            os << "0x";
            for (char c : bin.bytes()) {
                os << llvm::format_hex_no_prefix(c, 2, true);
            }
        }
        os << " }>";
        return;
    } else if (auto seqAttr = attr.dyn_cast_or_null<SeqAttr>()) {
        os << SeqAttr::getAttrName() << '<';
        os << '[';
        auto count = seqAttr.size();
        if (count > 0) {
            auto elements = seqAttr.getValue();
            bool printSeparator = count > 1;
            for (unsigned i = 0; i < count; i++) {
                p.printAttribute(elements[i]);
                if (i != (count - 1)) os << ", ";
            }
        }
        os << "]>";
        return;
    }
    llvm_unreachable("unhandled EIR type");
}
