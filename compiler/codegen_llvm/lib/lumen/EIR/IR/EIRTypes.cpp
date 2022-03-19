#include "lumen/EIR/IR/EIRTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Parser.h"

#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIREnums.h"

using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::mlir::TypeRange;

//===----------------------------------------------------------------------===//
// Tablegen Type and Type Interface Definitions
//===----------------------------------------------------------------------===//

#include "lumen/EIR/IR/EIRTypeInterfaces.h.inc"
#include "lumen/EIR/IR/EIRTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace lumen {
namespace eir {

unsigned isMatch(Type type, Type matcher) {
    // Special case handling for none
    if (type.isa<NoneType>()) return matcher.isa<NoneType>() ? 1 : 0;

    auto matcherBase = matcher.dyn_cast_or_null<TermTypeInterface>();
    if (!matcherBase) return 2;

    // Get the term type interface for convenience methods
    auto typeBase = type.cast<TermTypeInterface>();

    auto typeId = type.getTypeID();
    auto matcherTypeId = matcher.getTypeID();

    // Unresolvable statically
    if (type.isa<TermType>() || matcherBase.isa<TermType>()) return 2;

    // Guaranteed to match
    if (typeId == matcherTypeId) return 1;

    // Handle boxed types if the matcher is a box type
    if (matcherBase.isa<BoxType>() && type.isa<BoxType>()) {
        auto expected = matcher.cast<BoxType>().getPointeeType();
        auto inner = type.cast<BoxType>().getPointeeType();
        return isMatch(inner, expected);
    }

    // If the matcher is not a box, but is a boxable type, handle
    // comparing types correctly (i.e. if this is a boxed type, then
    // compare the boxed type)
    if (type.isa<BoxType>()) {
        auto inner = type.cast<BoxType>().getPointeeType();
        return isMatch(inner, matcher);
    }

    // Generic matches
    if (matcherBase.isAtomLike()) return typeBase.isAtomLike() ? 1 : 0;
    if (matcherBase.isListLike()) return typeBase.isListLike() ? 1 : 0;

    if (matcher.isa<IntegerType>()) return type.isa<IntegerType>() ? 1 : 0;
    if (matcher.isa<FloatType>()) return type.isa<FloatType>() ? 1 : 0;
    if (matcher.isa<BinaryType>()) return type.isa<BinaryType>() ? 1 : 0;
    if (matcher.isa<MapType>()) return type.isa<MapType>() ? 1 : 0;
    if (matcher.isa<ClosureType>()) return type.isa<ClosureType>() ? 2 : 0;

    if (auto tupleTy = matcher.dyn_cast_or_null<TupleType>()) {
        auto elementTypes = tupleTy.getTypes();
        auto isDynamic = elementTypes.size() == 0;
        if (auto tt = type.dyn_cast_or_null<TupleType>()) {
            auto arity = tt.size();
            if (isDynamic || arity == 0) return 2;
            if (arity == elementTypes.size()) return 2;
            return 0;
        }
        return 0;
    }

    return 0;
}

bool canTypeEverBeEqual(Type a, Type b, bool strict) {
    // If the types are the same, this is trivially true
    if (a == b) return true;

    // If types are opaque, we are optimistic and try the comparison anyway
    // This is very common, so we handle it before anything else
    if (a.isa<TermType>() || b.isa<TermType>()) return true;

    // Get the type interfaces for convenience
    auto ai = a.cast<TermTypeInterface>();
    auto bi = b.dyn_cast_or_null<TermTypeInterface>();

    // Numeric
    if (ai.isNumber()) {
        // If the other type is non-numeric, these can never compare equal
        if (bi) {
            if (!bi.isNumber()) return false;
        } else {
            if (!b.isIntOrFloat()) return false;
        }
        // If strict=false, numeric comparisons can always produce equality
        if (!strict) return true;
        // Otherwise, things are type-sensitive
        if (bi) {
            if (a.isa<IntegerType>() && b.isa<IntegerType>()) return true;
            if (a.isa<FloatType>() && b.isa<FloatType>()) return true;
        } else {
            if (a.isa<IntegerType>() && b.isIntOrIndex()) return true;
            if (a.isa<FloatType>() && b.isa<::mlir::FloatType>()) return true;
        }
        return false;
    }

    // Atoms
    if (ai.isAtomLike()) {
        // If a is a boolean, it may compare equal to other atoms, booleans and
        // i1 if strict=false, otherwise, if strict, it must match against a
        // boolean or i1
        if (a.isa<BooleanType>()) {
            if (strict)
                return b.isa<BooleanType>() || b.isInteger(1);
            else
                return (bi && bi.isAtomLike()) || b.isInteger(1);
        }
        // Otherwise, the type must be atom-like
        return (bi && bi.isAtomLike());
    }

    // Boxed Terms
    if (auto abox = a.dyn_cast_or_null<BoxType>()) {
        auto ap = abox.getPointeeType();
        if (auto bbox = b.dyn_cast_or_null<BoxType>()) {
            auto bp = bbox.getPointeeType();
            return canTypeEverBeEqual(ap, bp, strict);
        }
        return canTypeEverBeEqual(ap, b, strict);
    }

    // Tuples and closures are parameterized, so we have to evaluate them
    // manually
    if (auto at = a.dyn_cast_or_null<TupleType>()) {
        if (auto bt = a.dyn_cast_or_null<TupleType>()) {
            // If this isn't a strict comparison, skip the element-wise check
            if (!strict) return true;
            auto as = at.getTypes();
            auto bs = bt.getTypes();
            // If either tuple has unknown shape, assume they can compare equal
            if (as.size() == 0 || bs.size() == 0) return true;
            // If the arity differs, we know they can't compare equal
            if (as.size() != bs.size()) return false;
            // TODO: While we could check the element types here, we don't
            // currently propagate enough type information to benefit from the
            // extra work, so we assume that if the arity of the tuples is the
            // same, they can compare equal
            return true;
        }
        return false;
    }

    if (auto at = a.dyn_cast_or_null<ClosureType>()) {
        auto asig = at.getSignatureType();
        auto as = at.getEnvTypes();
        auto aarity = as.size();

        if (auto bt = b.dyn_cast_or_null<ClosureType>()) {
            auto bsig = bt.getSignatureType();
            auto bs = at.getEnvTypes();
            auto barity = bs.size();

            // If we don't have both a signature and env arity to compare,
            // assume they could equate, unless we have arity for both
            // environments, in which case we can return false if the envs are
            // of different arity
            if (!asig.hasValue() || !bsig.hasValue())
                if (aarity == 0 || barity == 0)
                    return true;
                else
                    return aarity == barity;

            // Mismatched env arity can never compare equal
            if (aarity != 0 && barity != 0 && aarity != barity) return false;

            // If the signatures have the same number of inputs/results, assume
            // they can equate
            auto afun = asig.getValue();
            auto bfun = bsig.getValue();

            auto ainputs = afun.getInputs();
            auto binputs = bfun.getInputs();
            if (ainputs.size() != binputs.size()) return false;

            auto aresults = afun.getResults();
            auto bresults = bfun.getResults();
            if (aresults.size() != bresults.size()) return false;

            return true;
        }

        // ClosureType and FunctionType can compare equal if both signatures
        // match and the envs are zero; OR we don't have enough information to
        // say
        if (auto bt = b.dyn_cast_or_null<FunctionType>()) {
            if (!asig.hasValue()) return true;
            if (aarity != 0) return false;

            auto afun = asig.getValue();
            auto ainputs = afun.getInputs();
            auto binputs = bt.getInputs();
            if (ainputs.size() != binputs.size()) return false;

            auto aresults = afun.getResults();
            auto bresults = bfun.getResults();
            if (aresults.size() != bresults.size()) return false;

            return true;
        }

        return false;
    }

    // At this point any other term/term comparisons won't succeed, and all
    // other type combinations are in theory unequatable, but this is an
    // optimistic predicate, so in the case where we don't know for sure that a
    // type combination can't compare equal, we assume they could, unless
    // strict=true.
    return !strict && bi == nullptr;
}

Type eirDialect::parseType(mlir::DialectAsmParser &parser) const {
    StringRef mnemonic;
    if (failed(parser.parseKeyword(&mnemonic))) {
        parser.emitError(parser.currentLocation(),
                         "expected operation mnemonic");
        return Type();
    }
    return generatedTypeParser(getContext(), parser, mnemonic);
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

void eirDialect::printType(Type ty, mlir::DialectAsmPrinter &p) const {
    if (mlir::succeeded(generatedTypePrinter(ty, printer))) return;
    ty.dump();
    llvm::report_fatal_error("unrecognized dialect type!");
}

}  // namespace eir
}  // namespace lumen
