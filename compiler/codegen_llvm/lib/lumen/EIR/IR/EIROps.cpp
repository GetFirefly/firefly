#include "lumen/EIR/IR/EIROps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "lumen/EIR/IR/EIRAttributes.h"
#include "lumen/EIR/IR/EIRTypes.h"

#include <iterator>
#include <vector>

using namespace lumen;
using namespace lumen::eir;

namespace lumen {
namespace eir {

struct cast_binder {
    Value *bind_value;

    cast_binder(Value *bv) : bind_value(bv) {}

    bool match(Operation *op) {
        if (auto castOp = dyn_cast_or_null<CastOp>(op)) {
            Value source = castOp.input();
            *bind_value = source;
            return true;
        }

        return false;
    }
};

/// Matches a ConstantIntOp

/// The matcher that matches a constant numeric operation and binds the constant
/// value.
struct constant_apint_op_binder {
    APIntAttr::ValueType *bind_value;

    /// Creates a matcher instance that binds the value to bv if match succeeds.
    constant_apint_op_binder(APIntAttr::ValueType *bv) : bind_value(bv) {}

    bool match(Operation *op) {
        Attribute attr;
        if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
            return false;
        auto type = op->getResult(0).getType();

        if (type.isa<IntegerType>())
            return mlir::detail::attr_value_binder<APIntAttr>(bind_value)
                .match(attr);
        if (auto box = type.dyn_cast<BoxType>())
            if (box.getPointeeType().isa<IntegerType>())
                return mlir::detail::attr_value_binder<APIntAttr>(bind_value)
                    .match(attr);

        return false;
    }
};

/// The matcher that matches a constant numeric operation and binds the constant
/// value.
struct constant_apfloat_op_binder {
    APFloatAttr::ValueType *bind_value;

    /// Creates a matcher instance that binds the value to bv if match succeeds.
    constant_apfloat_op_binder(APFloatAttr::ValueType *bv) : bind_value(bv) {}

    bool match(Operation *op) {
        Attribute attr;
        if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
            return false;
        auto type = op->getResult(0).getType();

        if (type.isa<FloatType>())
            return mlir::detail::attr_value_binder<APFloatAttr>(bind_value)
                .match(attr);

        return false;
    }
};

/// The matcher that matches a constant boolean operation and binds the constant
/// value.
struct constant_bool_op_binder {
    BoolAttr::ValueType *bind_value;

    /// Creates a matcher instance that binds the value to bv if match succeeds.
    constant_bool_op_binder(BoolAttr::ValueType *bv) : bind_value(bv) {}

    bool match(Operation *op) {
        Attribute attr;
        if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
            return false;
        auto type = op->getResult(0).getType();

        if (type.isa<BooleanType>()) {
            return mlir::detail::attr_value_binder<BoolAttr>(bind_value)
                .match(attr);
        }

        if (type.isa<AtomType>()) {
            if (auto atomAttr = attr.dyn_cast<AtomAttr>()) {
                auto id = atomAttr.getValue().getLimitedValue();
                if (id == 0 || id == 1) {
                    *bind_value = id == 1;
                    return true;
                }
                return false;
            }
        }

        if (type.isInteger(1)) {
            if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
                auto val = intAttr.getValue().getLimitedValue();
                if (val == 0 || val == 1) {
                    *bind_value = val == 1;
                    return true;
                }
                return false;
            }

            // We might have a boolean constant with i1 as type
            return mlir::detail::attr_value_binder<BoolAttr>(bind_value)
                .match(attr);
        }

        return false;
    }
};

struct constant_atom_op_binder {
    AtomAttr::ValueType *bind_value;

    /// Creates a matcher instance that binds the value to bv if match succeeds.
    constant_atom_op_binder(AtomAttr::ValueType *bv) : bind_value(bv) {}

    bool match(Operation *op) {
        Attribute attr;
        if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
            return false;
        auto type = op->getResult(0).getType();

        if (type.isa<AtomType>()) {
            return mlir::detail::attr_value_binder<AtomAttr>(bind_value)
                .match(attr);
        }

        if (type.isa<BooleanType>()) {
            if (auto boolAttr = attr.dyn_cast<BoolAttr>()) {
                *bind_value =
                    APInt(64, (uint64_t)boolAttr.getValue(), /*signed=*/false);
                return true;
            }
            return false;
        }

        return false;
    }
};

inline cast_binder m_Cast(Value *bind_value) { return cast_binder(bind_value); }

inline constant_apint_op_binder m_ConstInt(APIntAttr::ValueType *bind_value) {
    return constant_apint_op_binder(bind_value);
}

inline constant_apfloat_op_binder m_ConstFloat(
    APFloatAttr::ValueType *bind_value) {
    return constant_apfloat_op_binder(bind_value);
}

inline constant_bool_op_binder m_ConstBool(BoolAttr::ValueType *bind_value) {
    return constant_bool_op_binder(bind_value);
}

inline constant_atom_op_binder m_ConstAtom(AtomAttr::ValueType *bind_value) {
    return constant_atom_op_binder(bind_value);
}

static Value castToTermEquivalent(OpBuilder &builder, Value input) {
    Type inputType = input.getType();
    if (inputType.isa<TermType>()) return input;

    Type targetType;
    if (auto ptrTy = inputType.dyn_cast_or_null<PtrType>()) {
        Type innerTy = ptrTy.getPointeeType();
        if (dyn_cast<TermTypeInterface>(innerTy))
            targetType = builder.getType<BoxType>(innerTy);
        else
            targetType = builder.getType<TermType>();
    } else {
        targetType = builder.getType<TermType>();
    }

    auto castOp = builder.create<CastOp>(input.getLoc(), input, targetType);
    return castOp.getResult();
}

//===----------------------------------------------------------------------===//
// eir.func
//===----------------------------------------------------------------------===//

static ParseResult parseFuncOp(OpAsmParser &parser, OperationState &result) {
    auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                            ArrayRef<Type> results, mlir::impl::VariadicFlag,
                            std::string &) {
        return builder.getFunctionType(argTypes, results);
    };
    return mlir::impl::parseFunctionLikeOp(
        parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(OpAsmPrinter &p, FuncOp &op) {
    FunctionType fnType = op.getType();
    mlir::impl::printFunctionLikeOp(p, op, fnType.getInputs(),
                                    /*isVariadic=*/false, fnType.getResults());
}

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
    OperationState state(location, FuncOp::getOperationName());
    OpBuilder builder(location->getContext());
    FuncOp::build(builder, state, name, type, attrs);
    return cast<FuncOp>(Operation::create(state));
}

void FuncOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
    result.addRegion();
    result.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
    result.addAttribute("type", TypeAttr::get(type));
    result.attributes.append(attrs.begin(), attrs.end());
    if (argAttrs.empty()) {
        return;
    }

    if (argAttrs.empty()) return;

    unsigned numInputs = type.getNumInputs();
    assert(numInputs == argAttrs.size() &&
           "expected as many argument attribute lists as arguments");
    SmallString<8> argAttrName;
    for (unsigned i = 0, e = numInputs; i != e; ++i)
        if (auto argDict = argAttrs[i].getDictionary(builder.getContext()))
            result.addAttribute(getArgAttrName(i, argAttrName), argDict);
}

Block *FuncOp::addEntryBlock() {
    assert(empty() && "function already has an entry block");
    auto *entry = new Block();
    push_back(entry);
    entry->addArguments(getType().getInputs());
    return entry;
}

LogicalResult FuncOp::verifyType() {
    auto type = getTypeAttr().getValue();
    if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
    return success();
}

Region *FuncOp::getCallableRegion() { return &body(); }

ArrayRef<Type> FuncOp::getCallableResults() { return getType().getResults(); }

//===----------------------------------------------------------------------===//
// eir.call
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeCall : public OpRewritePattern<CallOp> {
    using OpRewritePattern<CallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CallOp op,
                                  PatternRewriter &rewriter) const override {
        auto operandsMut = op.operandsMutable();
        if (operandsMut.size() == 0) return success();

        OperandRange operands(operandsMut);
        auto index = operands.getBeginOperandIndex();
        for (auto operand : op.operands()) {
            Value o = castToTermEquivalent(rewriter, operand);
            if (o != operand) {
                operandsMut.slice(index, 1).assign(o);
            }
            index++;
        }

        return success();
    }
};
}  // end anonymous namespace.

void CallOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
    results.insert<CanonicalizeCall>(context);
}

//===----------------------------------------------------------------------===//
// eir.invoke
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeInvoke : public OpRewritePattern<InvokeOp> {
    using OpRewritePattern<InvokeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(InvokeOp op,
                                  PatternRewriter &rewriter) const override {
        auto operandsMut = op.operandsMutable();
        if (operandsMut.size() == 0) return success();

        OperandRange operands(operandsMut);
        auto index = operands.getBeginOperandIndex();
        for (auto operand : op.operands()) {
            Value o = castToTermEquivalent(rewriter, operand);
            if (o != operand) {
                operandsMut.slice(index, 1).assign(o);
            }
            index++;
        }

        return success();
    }
};
}  // end anonymous namespace.

void InvokeOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
    results.insert<CanonicalizeInvoke>(context);
}

//===----------------------------------------------------------------------===//
// eir.throw
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeThrow : public OpRewritePattern<ThrowOp> {
    using OpRewritePattern<ThrowOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ThrowOp op,
                                  PatternRewriter &rewriter) const override {
        Value kind = op.kind();
        Value k = castToTermEquivalent(rewriter, kind);
        if (k != kind) {
            auto kindOperand = op.kindMutable();
            kindOperand.assign(k);
        }

        Value reason = op.kind();
        Value r = castToTermEquivalent(rewriter, reason);
        if (r != reason) {
            auto reasonOperand = op.reasonMutable();
            reasonOperand.assign(r);
        }

        return success();
    }
};
}  // end anonymous namespace.

void ThrowOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
    results.insert<CanonicalizeThrow>(context);
}

//===----------------------------------------------------------------------===//
// eir.return
//===----------------------------------------------------------------------===//

static LogicalResult verify(ReturnOp op) {
    auto function = cast<FuncOp>(op.getParentOp());

    // The operand number and types must match the function signature.
    const auto &results = function.getType().getResults();
    if (op.getNumOperands() != results.size())
        return op.emitOpError("has ")
               << op.getNumOperands()
               << " operands, but enclosing function returns "
               << results.size();

    for (unsigned i = 0, e = results.size(); i != e; ++i)
        if (op.getOperand(i).getType() != results[i])
            return op.emitError() << "type of return operand " << i << " ("
                                  << op.getOperand(i).getType()
                                  << ") doesn't match function result type ("
                                  << results[i] << ")";

    return success();
}

namespace {
struct CanonicalizeReturn : public OpRewritePattern<ReturnOp> {
    using OpRewritePattern<ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ReturnOp op,
                                  PatternRewriter &rewriter) const override {
        auto operandsMut = op.operandsMutable();
        if (operandsMut.size() == 0) return success();

        auto funcOp = op.getOperation()->getParentOfType<FuncOp>();
        auto funcType = funcOp.getType();
        auto resultTypes = funcType.getResults();

        OperandRange operands(operandsMut);
        auto index = operands.getBeginOperandIndex();
        for (auto it : llvm::zip(op.operands(), resultTypes)) {
            Value operand = std::get<0>(it);
            Type expectedType = std::get<1>(it);
            if (operand.getType() != expectedType) {
                auto castOp =
                    rewriter.create<CastOp>(op.getLoc(), operand, expectedType);
                operandsMut.slice(index, 1).assign(castOp.getResult());
            }
            index++;
        }

        return success();
    }
};
}  // end anonymous namespace.

void ReturnOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
    results.insert<CanonicalizeReturn>(context);
}

//===----------------------------------------------------------------------===//
// eir.is_type
//===----------------------------------------------------------------------===//

static LogicalResult verify(IsTypeOp op) {
    auto typeAttr = op->getAttrOfType<TypeAttr>("type");
    if (!typeAttr)
        return op.emitOpError("requires type attribute named 'type'");

    if (!typeAttr.getValue().hasTrait<TermTypeInterface::Trait>())
        return op.emitOpError("expected term type");

    return success();
}

OpFoldResult IsTypeOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.size() == 1);

    Type matchType = getMatchType();
    Type inputType;

    Attribute constInput = operands[0];
    if (constInput) {
        if (auto typeAttr = constInput.dyn_cast_or_null<TypeAttr>()) {
            inputType = typeAttr.getValue();
        } else {
            inputType = constInput.getType();
        }
    } else {
        auto input = getOperand();
        inputType = input.getType();
    }

    // Fast path for when the types are obviously the same
    if (inputType == matchType) return BoolAttr::get(true, getContext());

    // The match type should always be a term type, but handle it gracefully in
    // the instance where it is not. We should use verify/1 to handle invalid
    // type arguments
    if (!matchType.hasTrait<TermTypeInterface::Trait>()) return nullptr;

    auto expected = dyn_cast<TermTypeInterface>(matchType);

    // If the input value is itself an opaque term type, then unwrap it
    // and delegate to the isMatch helper. This handles all term/term
    // comparisons
    if (auto inputTy = dyn_cast<TermTypeInterface>(inputType)) {
        switch (inputTy.isMatch(matchType)) {
        case 0:
            return BoolAttr::get(false, getContext());
        case 1:
            return BoolAttr::get(true, getContext());
        case 2:
            return nullptr;
        }
    }

    // The above would have handled all term types, but ptr/ref types
    // are not technically term types, however they get coerced to them, so
    // if we have a typecheck on a pointer value, we can resolve that check
    // statically if the pointee type matches the boxed type that is expected
    if (!matchType.isa<BoxType>() && !expected.isBoxable()) return nullptr;

    // Normalize the type which should be compared against the pointee type
    Type boxedTy;
    if (auto box = matchType.dyn_cast<BoxType>())
        boxedTy = box.getPointeeType();
    else
        boxedTy = expected;

    if (auto ptrTy = inputType.dyn_cast_or_null<PtrType>()) {
        Type innerTy = ptrTy.getPointeeType();
        if (auto innerTermTy = dyn_cast<TermTypeInterface>(innerTy)) {
            switch (innerTermTy.isMatch(boxedTy)) {
            case 0:
                return BoolAttr::get(false, getContext());
            case 1:
                return BoolAttr::get(true, getContext());
            case 2:
                return nullptr;
            }
        }
        return nullptr;
    }

    // Err on the side of caution and do not assume no match, as later passes
    // may change the code enough to provide us with missing information
    return nullptr;
}

//===----------------------------------------------------------------------===//
// eir.is_tuple
//===----------------------------------------------------------------------===//

static LogicalResult verify(IsTupleOp op) {
    auto numOperands = op.getNumOperands();
    if (numOperands < 1 || numOperands > 2)
        return op.emitOpError(
            "invalid number of operands, expected at least one and no more "
            "than 2");

    return success();
}

OpFoldResult IsTupleOp::fold(ArrayRef<Attribute> operands) {
    auto numOperands = getNumOperands();
    if (numOperands < 1 || numOperands > 2) return nullptr;

    auto input = getOperand(0);
    Type inputType = input.getType();
    if (auto boxType = inputType.dyn_cast_or_null<BoxType>()) {
        if (boxType.getBoxedType().isa<TupleType>())
            return BoolAttr::get(true, getContext());
        else
            return BoolAttr::get(false, getContext());
    } else if (auto ptrType = inputType.dyn_cast_or_null<PtrType>()) {
        if (auto tt = ptrType.getInnerType().dyn_cast_or_null<TupleType>())
            if (numOperands == 2 && tt.hasStaticShape()) {
                Attribute arityAttr = operands[1];
                if (!arityAttr) return nullptr;

                if (auto intAttr = arityAttr.dyn_cast_or_null<IntegerAttr>()) {
                    auto arity = intAttr.getValue();
                    if (arity == tt.getArity())
                        return BoolAttr::get(true, getContext());
                    else
                        return BoolAttr::get(false, getContext());
                } else if (auto intAttr =
                               arityAttr.dyn_cast_or_null<APIntAttr>()) {
                    auto arity = intAttr.getValue();
                    if (arity == tt.getArity())
                        return BoolAttr::get(true, getContext());
                    else
                        return BoolAttr::get(false, getContext());
                } else {
                    return nullptr;
                }
            } else {
                return BoolAttr::get(true, getContext());
            }
        else if (ptrType.getPointeeType().hasTrait<TermTypeInterface::Trait>())
            return BoolAttr::get(false, getContext());
        else
            return nullptr;
    } else if (inputType.hasTrait<TermTypeInterface::Trait>()) {
        if (!inputType.isa<TermType>())
            return BoolAttr::get(false, getContext());
        else
            return nullptr;
    }

    return nullptr;
}

//===----------------------------------------------------------------------===//
// eir.is_function
//===----------------------------------------------------------------------===//

static LogicalResult verify(IsFunctionOp op) {
    auto numOperands = op.getNumOperands();
    if (numOperands < 1 || numOperands > 2)
        return op.emitOpError(
            "invalid number of operands, expected at least one and no more "
            "than 2");

    return success();
}

OpFoldResult IsFunctionOp::fold(ArrayRef<Attribute> operands) {
    auto numOperands = getNumOperands();
    if (numOperands < 1 || numOperands > 2) return nullptr;

    auto input = getOperand(0);
    Type inputType = input.getType();
    if (auto boxType = inputType.dyn_cast_or_null<BoxType>()) {
        if (auto closureTy =
                boxType.getPointeeType().dyn_cast_or_null<ClosureType>()) {
            if (numOperands < 2) return BoolAttr::get(true, getContext());
            auto arityAttr = operands[1];
            if (!arityAttr) return nullptr;
            auto closureArity = closureTy.getArity();
            if (!closureArity.hasValue()) return nullptr;
            auto expectedArity =
                arityAttr.cast<IntegerAttr>().getValue().getLimitedValue();
            if (expectedArity == closureArity.getValue())
                return BoolAttr::get(true, getContext());
            else
                return BoolAttr::get(false, getContext());
        } else {
            return BoolAttr::get(false, getContext());
        }
    } else if (auto ptrType = inputType.dyn_cast_or_null<PtrType>()) {
        if (auto closureTy =
                ptrType.getPointeeType().dyn_cast_or_null<ClosureType>()) {
            if (numOperands < 2) return BoolAttr::get(true, getContext());
            auto arityAttr = operands[1];
            if (!arityAttr) return nullptr;
            auto closureArity = closureTy.getArity();
            if (!closureArity.hasValue()) return nullptr;
            auto expectedArity =
                arityAttr.cast<IntegerAttr>().getValue().getLimitedValue();
            if (expectedArity == closureArity.getValue())
                return BoolAttr::get(true, getContext());
            else
                return BoolAttr::get(false, getContext());
        } else if (ptrType.getPointeeType()
                       .hasTrait<TermTypeInterface::Trait>()) {
            return BoolAttr::get(false, getContext());
        } else {
            return nullptr;
        }
    } else if (inputType.hasTrait<TermTypeInterface::Trait>()) {
        if (!inputType.isa<TermType>())
            return BoolAttr::get(false, getContext());
        else
            return nullptr;
    }

    return nullptr;
}

//===----------------------------------------------------------------------===//
// eir.cast
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
    Type srcTy = getSourceType();
    Type targetTy = getTargetType();
    // Identity cast
    if (srcTy == targetTy) return input();
    return nullptr;
}

static bool areCastCompatible(Type srcType, Type destType) {
    auto srcInfo = dyn_cast<TermTypeInterface>(srcType);
    if (!srcInfo) return false;
    auto destInfo = dyn_cast<TermTypeInterface>(destType);
    if (!destInfo) return false;

    if (destType.isa<TermType>()) {
        // Casting an immediate to an opaque term is always allowed
        if (srcType.isImmediate()) return true;
        // Casting a boxed value to an opaque term is always allowed
        if (srcType.isa<BoxType>()) return true;
        // This is redundant, but technically allowed and will be eliminated via
        // canonicalization
        if (srcType.isa<TermType>()) return true;
    }
    // Casting an opaque term to any term type is always allowed
    if (srcType.isa<TermType>()) return true;
    // Box-to-box casts are always allowed
    if (srcType.isa<BoxType>() & destType.isa<BoxType>()) return true;
    // Only header types can be boxed
    if (destType.isa<BoxType>() && !srcInfo.isBoxable()) return false;
    // A cast must be to an immediate-sized type
    if (!destInfo.isImmediate()) return false;
    // Only support casts between compatible types
    if (srcInfo.isNumberLike() && !destInfo.isNumberLike()) return false;
    if (srcType.isa<IntegerType>() && !destType.isa<IntegerType>())
        return false;
    if (srcType.isa<FloatType>() && !destType.isa<FloatType>()) return false;
    if (srcInfo.isListLike() && !destInfo.isListLike()) return false;
    if (srcType.isa<BinaryType>() && !destType.isa<BinaryType>()) return false;
    // All other casts are supported
    return true;
}

static LogicalResult verify(CastOp op) {
    auto srcType = op.getSourceType();
    auto destType = op.getTargetType();
    if (srcType.hasTrait<TermTypeInterface::Trait>() &&
        destType.hasTrait<TermTypeInterface::Trait>()) {
        if (!areCastCompatible(srcType, destType)) {
            return op.emitError("operand type ")
                   << srcType << " and result type " << destType
                   << " are not cast compatible";
        }
        return success();
    }

    if ((srcType.isa<PtrType>() || srcType.isa<BoxType>()) &&
        destType.isa<PtrType>())
        return success();
    if (srcType.isIntOrFloat() && destType.isIntOrFloat()) return success();

    return op.emitError(
        "invalid cast, source and target type combination is unsupported");
}

//===----------------------------------------------------------------------===//
// eir.match
//===----------------------------------------------------------------------===//

LogicalResult lowerPatternMatch(OpBuilder &builder, Location loc,
                                Value selector,
                                ArrayRef<MatchBranch> branches) {
    auto numBranches = branches.size();
    assert(numBranches > 0 && "expected at least one branch in a match");

    auto *currentBlock = builder.getInsertionBlock();
    auto *region = currentBlock->getParent();
    auto selectorType = selector.getType();

    // Save our insertion point in the current block
    auto startIp = builder.saveInsertionPoint();

    // Create blocks for all match arms
    bool needsFallbackBranch = true;
    SmallVector<Block *, 3> blocks;
    // The first match arm is evaluated in the current block, so we
    // handle it specially
    blocks.reserve(numBranches);
    blocks.push_back(currentBlock);
    if (branches[0].isCatchAll()) {
        needsFallbackBranch = false;
    }
    // All other match arms need blocks for the evaluation of their patterns
    for (auto &branch : branches.take_back(numBranches - 1)) {
        if (branch.isCatchAll()) {
            needsFallbackBranch = false;
        }
        Block *block = builder.createBlock(region);
        block->addArgument(selectorType);
        blocks.push_back(block);
    }

    // Create fallback block, if needed, after all other match blocks, so
    // that after all other conditions have been tried, we branch to an
    // unreachable to force a trap
    Block *failed = nullptr;
    if (needsFallbackBranch) {
        failed = builder.createBlock(region);
        failed->addArgument(selectorType);
        builder.create<eir::UnreachableOp>(loc);
    }

    // Restore our original insertion point
    builder.restoreInsertionPoint(startIp);

    // Save the current insertion point, which we'll restore when lowering is
    // complete
    auto finalIp = builder.saveInsertionPoint();

    // Common types used below

    // Used whenever we need a set of empty args below
    ArrayRef<Value> emptyArgs{};

    // For each branch, populate its block with the predicate and
    // appropriate conditional branching instruction to either jump
    // to the success block, or to the next branches' block (or in
    // the case of the last branch, the 'failed' block)
    for (unsigned i = 0; i < numBranches; i++) {
        auto &b = branches[i];
        Location branchLoc = b.getLoc();
        bool isLast = i == numBranches - 1;
        Block *block = blocks[i];

        // Set our insertion point to the end of the pattern block
        builder.setInsertionPointToEnd(block);

        // Get the selector value in this block,
        // in the case of the first block, its our original
        // input selector value
        Value selectorArg;
        if (i == 0) {
            selectorArg = selector;
        } else {
            selectorArg = block->getArgument(0);
        }
        ArrayRef<Value> withSelectorArgs{selectorArg};

        // Store the next pattern to try if this one fails
        // If this is the last pattern, we validate that the
        // branch either unconditionally succeeds, or branches to
        // an unreachable op
        Block *nextPatternBlock = nullptr;
        if (!isLast) {
            nextPatternBlock = blocks[i + 1];
        } else if (needsFallbackBranch) {
            nextPatternBlock = failed;
        }

        auto dest = b.getDest();
        auto baseDestArgs = b.getDestArgs();
        auto numBaseDestArgs = baseDestArgs.size();

        // Ensure the destination block argument types are propagated
        for (unsigned i = 0; i < baseDestArgs.size(); i++) {
            BlockArgument arg = dest->getArgument(i);
            auto destArg = baseDestArgs[i];
            auto destArgTy = destArg.getType();
            if (arg.getType() != destArgTy) arg.setType(destArgTy);
        }

        switch (b.getPatternType()) {
        case MatchPatternType::Any: {
            // This unconditionally branches to its destination
            builder.create<BranchOp>(branchLoc, dest, baseDestArgs);
            break;
        }

        case MatchPatternType::Cons: {
            assert(nextPatternBlock != nullptr &&
                   "last match block must end in unconditional branch");
            auto cip = builder.saveInsertionPoint();
            // 1. Split block, and conditionally branch to split if is_cons,
            // otherwise the next pattern
            Block *split =
                builder.createBlock(region, Region::iterator(nextPatternBlock));
            builder.restoreInsertionPoint(cip);
            auto consType = builder.getType<ConsType>();
            auto boxedConsType = builder.getType<BoxType>(consType);
            auto isConsOp =
                builder.create<IsTypeOp>(branchLoc, selectorArg, boxedConsType);
            auto isConsCond = isConsOp.getResult();
            builder.create<CondBranchOp>(branchLoc, isConsCond, split,
                                         emptyArgs, nextPatternBlock,
                                         withSelectorArgs);
            // 2. In the split, extract head and tail values of the cons cell
            builder.setInsertionPointToEnd(split);
            auto ptrConsType = builder.getType<PtrType>(consType);
            auto castOp =
                builder.create<CastOp>(branchLoc, selectorArg, ptrConsType);
            auto consPtr = castOp.getResult();
            auto getHeadOp =
                builder.create<GetElementPtrOp>(branchLoc, consPtr, 0);
            auto getTailOp =
                builder.create<GetElementPtrOp>(branchLoc, consPtr, 1);
            auto headPointer = getHeadOp.getResult();
            auto tailPointer = getTailOp.getResult();
            auto headLoadOp = builder.create<LoadOp>(branchLoc, headPointer);
            auto headLoadResult = headLoadOp.getResult();
            auto tailLoadOp = builder.create<LoadOp>(branchLoc, tailPointer);
            auto tailLoadResult = tailLoadOp.getResult();
            // 3. Unconditionally branch to the destination, with head/tail as
            // additional destArgs
            unsigned i = numBaseDestArgs > 0 ? numBaseDestArgs - 1 : 0;
            dest->getArgument(i++).setType(headLoadResult.getType());
            dest->getArgument(i).setType(tailLoadResult.getType());
            SmallVector<Value, 2> destArgs(
                {baseDestArgs.begin(), baseDestArgs.end()});
            destArgs.push_back(headLoadResult);
            destArgs.push_back(tailLoadResult);
            builder.create<BranchOp>(branchLoc, dest, destArgs);
            break;
        }

        case MatchPatternType::Tuple: {
            assert(nextPatternBlock != nullptr &&
                   "last match block must end in unconditional branch");
            auto *pattern = b.getPatternTypeOrNull<TuplePattern>();
            // 1. Split block, and conditionally branch to split if is_tuple
            // w/arity N, otherwise the next pattern
            auto cip = builder.saveInsertionPoint();
            Block *split =
                builder.createBlock(region, Region::iterator(nextPatternBlock));
            builder.restoreInsertionPoint(cip);
            auto arity = pattern->getArity();
            auto tupleType = builder.getType<eir::TupleType>(arity);
            auto boxedTupleType = builder.getType<BoxType>(tupleType);
            auto isTupleOp = builder.create<IsTypeOp>(branchLoc, selectorArg,
                                                      boxedTupleType);
            auto isTupleCond = isTupleOp.getResult();
            builder.create<CondBranchOp>(branchLoc, isTupleCond, split,
                                         emptyArgs, nextPatternBlock,
                                         withSelectorArgs);
            // 2. In the split, extract the tuple elements as values
            builder.setInsertionPointToEnd(split);
            auto ptrTupleType = builder.getType<PtrType>(tupleType);
            auto castOp =
                builder.create<CastOp>(branchLoc, selectorArg, ptrTupleType);
            auto tuplePtr = castOp.getResult();
            unsigned ai = numBaseDestArgs > 0 ? numBaseDestArgs - 1 : 0;
            SmallVector<Value, 2> destArgs(
                {baseDestArgs.begin(), baseDestArgs.end()});
            destArgs.reserve(arity);
            for (int64_t i = 0; i < arity; i++) {
                auto getElemOp =
                    builder.create<GetElementPtrOp>(branchLoc, tuplePtr, i + 1);
                auto elemPtr = getElemOp.getResult();
                auto elemLoadOp = builder.create<LoadOp>(branchLoc, elemPtr);
                auto elemLoadResult = elemLoadOp.getResult();
                dest->getArgument(ai++).setType(elemLoadResult.getType());
                destArgs.push_back(elemLoadResult);
            }
            // 3. Unconditionally branch to the destination, with the tuple
            // elements as additional destArgs
            builder.create<BranchOp>(branchLoc, dest, destArgs);
            break;
        }

        case MatchPatternType::MapItem: {
            assert(nextPatternBlock != nullptr &&
                   "last match block must end in unconditional branch");
            // 1. Split block twice, and conditionally branch to the first split
            // if is_map, otherwise the next pattern
            auto cip = builder.saveInsertionPoint();
            Block *split2 =
                builder.createBlock(region, Region::iterator(nextPatternBlock));
            Block *split =
                builder.createBlock(region, Region::iterator(split2));
            builder.restoreInsertionPoint(cip);
            auto *pattern = b.getPatternTypeOrNull<MapPattern>();
            auto key = pattern->getKey();
            auto mapType = BoxType::get(builder.getType<MapType>());
            auto isMapOp =
                builder.create<IsTypeOp>(branchLoc, selectorArg, mapType);
            auto isMapCond = isMapOp.getResult();
            builder.create<CondBranchOp>(branchLoc, isMapCond, split, emptyArgs,
                                         nextPatternBlock, withSelectorArgs);
            // 2. In the split, call runtime function `is_map_key` to confirm
            // existence of the key in the map,
            //    then conditionally branch to the second split if successful,
            //    otherwise the next pattern
            builder.setInsertionPointToEnd(split);
            auto hasKeyOp =
                builder.create<MapContainsKeyOp>(branchLoc, selectorArg, key);
            auto hasKeyCond = hasKeyOp.getResult();
            builder.create<CondBranchOp>(branchLoc, hasKeyCond, split2,
                                         emptyArgs, nextPatternBlock,
                                         withSelectorArgs);
            // 3. In the second split, call runtime function `map_get` to obtain
            // the value for the key
            builder.setInsertionPointToEnd(split2);
            auto mapGetOp =
                builder.create<MapGetKeyOp>(branchLoc, selectorArg, key);
            auto valueTerm = mapGetOp.getResult();
            unsigned i = numBaseDestArgs > 0 ? numBaseDestArgs - 1 : 0;
            dest->getArgument(i).setType(valueTerm.getType());
            // 4. Unconditionally branch to the destination, with the key's
            // value as an additional destArg
            SmallVector<Value, 2> destArgs(baseDestArgs.begin(),
                                           baseDestArgs.end());
            destArgs.push_back(valueTerm);
            builder.create<BranchOp>(branchLoc, dest, destArgs);
            break;
        }

        case MatchPatternType::IsType: {
            assert(nextPatternBlock != nullptr &&
                   "last match block must end in unconditional branch");
            // 1. Conditionally branch to destination if is_<type>, otherwise
            // the next pattern
            auto *pattern = b.getPatternTypeOrNull<IsTypePattern>();
            auto expectedType = pattern->getExpectedType();
            auto isTypeOp =
                builder.create<IsTypeOp>(branchLoc, selectorArg, expectedType);
            auto isTypeCond = isTypeOp.getResult();
            builder.create<CondBranchOp>(branchLoc, isTypeCond, dest,
                                         baseDestArgs, nextPatternBlock,
                                         withSelectorArgs);
            break;
        }

        case MatchPatternType::Value: {
            assert(nextPatternBlock != nullptr &&
                   "last match block must end in unconditional branch");
            // 1. Conditionally branch to dest if the value matches the
            // selector,
            //    passing the value as an additional destArg
            auto *pattern = b.getPatternTypeOrNull<ValuePattern>();
            auto expected = pattern->getValue();
            auto isEq =
                builder.create<CmpEqOp>(branchLoc, selectorArg, expected,
                                        /*strict=*/true);
            auto isEqCond = isEq.getResult();
            builder.create<CondBranchOp>(branchLoc, isEqCond, dest,
                                         baseDestArgs, nextPatternBlock,
                                         withSelectorArgs);
            break;
        }

        case MatchPatternType::Binary: {
            // 1. Split block, and conditionally branch to split if is_bitstring
            // (or is_binary), otherwise the next pattern
            // 2. In the split, conditionally branch to destination if
            // construction of the head value succeeds,
            //    otherwise the next pattern
            // NOTE: The exact semantics depend on the binary specification
            // type, and what is optimal in terms of checks. The success of the
            // overall branch results in two additional destArgs being passed to
            // the destination block, the decoded entry (head), and the rest of
            // the binary (tail)
            auto *pattern = b.getPatternTypeOrNull<BinaryPattern>();
            auto spec = pattern->getSpec();
            auto size = pattern->getSize();
            Operation *op;
            switch (spec.tag) {
            case BinarySpecifierType::Integer: {
                auto payload = spec.payload.i;
                bool isSigned = payload.isSigned;
                auto endianness = payload.endianness;
                auto unit = payload.unit;
                op = builder.create<BinaryMatchIntegerOp>(
                    branchLoc, selectorArg, isSigned, endianness, unit, size);
                break;
            }
            case BinarySpecifierType::Utf8: {
                op = builder.create<BinaryMatchUtf8Op>(branchLoc, selectorArg,
                                                       size);
                break;
            }
            case BinarySpecifierType::Utf16: {
                auto endianness = spec.payload.es.endianness;
                op = builder.create<BinaryMatchUtf16Op>(branchLoc, selectorArg,
                                                        endianness, size);
                break;
            }
            case BinarySpecifierType::Utf32: {
                auto endianness = spec.payload.es.endianness;
                op = builder.create<BinaryMatchUtf32Op>(branchLoc, selectorArg,
                                                        endianness, size);
                break;
            }
            case BinarySpecifierType::Float: {
                auto payload = spec.payload.f;
                op = builder.create<BinaryMatchFloatOp>(branchLoc, selectorArg,
                                                        payload.endianness,
                                                        payload.unit, size);
                break;
            }
            case BinarySpecifierType::Bytes:
            case BinarySpecifierType::Bits: {
                auto payload = spec.payload.us;
                op = builder.create<BinaryMatchRawOp>(branchLoc, selectorArg,
                                                      payload.unit, size);
                break;
            }
            }
            Value matched = op->getResult(0);
            Value rest = op->getResult(1);
            Value success = op->getResult(2);
            unsigned i = numBaseDestArgs > 0 ? numBaseDestArgs - 1 : 0;
            dest->getArgument(i++).setType(matched.getType());
            dest->getArgument(i).setType(rest.getType());
            SmallVector<Value, 2> destArgs(baseDestArgs.begin(),
                                           baseDestArgs.end());
            destArgs.push_back(matched);
            destArgs.push_back(rest);
            builder.create<CondBranchOp>(branchLoc, success, dest, destArgs,
                                         nextPatternBlock, withSelectorArgs);
            break;
        }
        }
    }
    builder.restoreInsertionPoint(finalIp);
    return success();
}

//===----------------------------------------------------------------------===//
// cmp.eq
//===----------------------------------------------------------------------===//

static APInt normalizeAPInt(APInt lhs, unsigned bitWidth) {
    auto lhsBits = lhs.getBitWidth();
    if (lhsBits >= bitWidth) return lhs;

    return lhs.sext(bitWidth);
}

// APInt comparisons require equivalent bit width
static bool areAPIntsEqual(APInt &lhs, APInt &rhs) {
    auto lhsBits = lhs.getBitWidth();
    auto rhsBits = rhs.getBitWidth();

    if (lhsBits == rhsBits) return lhs == rhs;

    if (lhsBits < rhsBits) {
        APInt temp = lhs.sext(rhsBits);
        return temp == rhs;
    }

    APInt temp = rhs.sext(lhsBits);
    return lhs == temp;
}

OpFoldResult CmpEqOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.size() == 2 && "binary op takes two operands");

    Attribute lhsOperand = operands[0];
    Attribute rhsOperand = operands[1];

    bool strict;
    if (getAttrOfType<mlir::UnitAttr>("is_strict"))
        strict = true;
    else
        strict = false;

    // If one operand is constant but not the other, move the constant
    // to the left, so that canonicalization can assume that non-constant
    // operands are always on the right hand side
    if (!lhsOperand || !rhsOperand) {
        Value l = lhs();
        Value r = rhs();
        auto lhsOperandMut = lhsMutable();
        auto rhsOperandMut = rhsMutable();

        // If both are non-constant, we can't do anything further
        if (!lhsOperand && !rhsOperand) {
            // Before we bail, if we have a term type on one side
            // and a non-term type on another, move the term type
            // to the right-hand operand
            Type lTy = l.getType();
            Type rTy = r.getType();
            // Nothing to do
            if (lTy.hasTrait<TermTypeInterface::Trait>() &&
                rTy.hasTrait<TermTypeInterface::Trait>())
                return nullptr;
            // Term is already on the right
            if (rTy.hasTrait<TermTypeInterface::Trait>()) return nullptr;

            // We have a term on the left, move to the right
            if (lTy.hasTrait<TermTypeInterface::Trait>()) {
                lhsOperandMut.assign(r);
                rhsOperandMut.assign(l);
            }

            return nullptr;
        }

        // If the non-constant operand is already on the right, we're done
        if (!rhsOperand) return nullptr;

        lhsOperandMut.assign(r);
        rhsOperandMut.assign(l);
        return nullptr;
    }

    // Atom-likes
    if (auto lhsAtom = lhsOperand.dyn_cast_or_null<AtomAttr>()) {
        if (auto rhsAtom = rhsOperand.dyn_cast_or_null<AtomAttr>()) {
            bool areEqual =
                areAPIntsEqual(lhsAtom.getValue(), rhsAtom.getValue());
            return BoolAttr::get(areEqual, getContext());
        }
        if (auto rhsBool = rhsOperand.dyn_cast_or_null<BoolAttr>()) {
            bool areEqual = lhsAtom.getValue().getLimitedValue() ==
                            (unsigned)(rhsBool.getValue());
            return BoolAttr::get(areEqual, getContext());
        }
        return nullptr;
    }

    // Boolean-likes
    if (auto lhsBool = lhsOperand.dyn_cast_or_null<BoolAttr>()) {
        if (auto rhsAtom = rhsOperand.dyn_cast_or_null<AtomAttr>()) {
            bool areEqual = (unsigned)(lhsBool.getValue()) ==
                            rhsAtom.getValue().getLimitedValue();
            return BoolAttr::get(areEqual, getContext());
        }
        if (auto rhsBool = rhsOperand.dyn_cast_or_null<BoolAttr>()) {
            bool areEqual = lhsBool.getValue() == rhsBool.getValue();
            return BoolAttr::get(areEqual, getContext());
        }
        return nullptr;
    }

    // Integers
    if (auto lhsInt = lhsOperand.dyn_cast_or_null<APIntAttr>()) {
        if (auto rhsInt = rhsOperand.dyn_cast_or_null<APIntAttr>()) {
            bool areEqual =
                areAPIntsEqual(lhsInt.getValue(), rhsInt.getValue());
            return BoolAttr::get(areEqual, getContext());
        }
        if (auto rhsInt = rhsOperand.dyn_cast_or_null<IntegerAttr>()) {
            auto rVal = rhsInt.getValue();
            bool areEqual = areAPIntsEqual(lhsInt.getValue(), rVal);
            return BoolAttr::get(areEqual, getContext());
        }
        // TODO: If non-strict, we can compare ints to floats
        return nullptr;
    }

    // Floats
    if (auto lhsFloat = lhsOperand.dyn_cast_or_null<APFloatAttr>()) {
        if (auto rhsFloat = rhsOperand.dyn_cast_or_null<APFloatAttr>()) {
            bool areEqual = lhsFloat.getValue() == rhsFloat.getValue();
            return BoolAttr::get(areEqual, getContext());
        }
        // TODO: If non-strict, we can compare ints to floats
    }

    return nullptr;
}

namespace {
static bool canTypesEverBeEqual(Type lhsTy, Type rhsTy, bool strict) {
    if (lhsTy == rhsTy) return true;

    if (auto lhs = lhsTy.dyn_cast_or_null<TermTypeInterface>()) {
        return lhs.canTypeEverBeEqual(rhsTy, strict);
    }

    if (auto rhs = rhsTy.dyn_cast_or_null<TermTypeInterface>()) {
        return rhs.canTypeEverBeEqual(lhsTy, strict);
    }

    if (lhsTy.isIntOrFloat() && rhsTy.isIntOrFloat()) return true;

    return false;
}

struct CanonicalizeEqualityComparison : public OpRewritePattern<CmpEqOp> {
    using OpRewritePattern<CmpEqOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(CmpEqOp op,
                                  PatternRewriter &rewriter) const override {
        auto lhs = op.lhs();
        auto rhs = op.rhs();
        auto resultTy = op.getType();
        auto i1Ty = rewriter.getI1Type();
        bool strict;
        if (op.getAttrOfType<mlir::UnitAttr>("is_strict"))
            strict = true;
        else
            strict = false;

        // We deal specially with eliminating redundant boolean comparisons,
        // since they occur frequently as a result of lowering, and are often
        // trivially removed
        bool lhsVal;
        bool rhsVal;
        auto lhsBoolPattern = m_ConstBool(&lhsVal);
        auto rhsBoolPattern = m_ConstBool(&rhsVal);
        if (matchPattern(lhs, lhsBoolPattern)) {
            // Left-side is a constant boolean value, check right side
            if (matchPattern(rhs, rhsBoolPattern)) {
                // Both operands are boolean, constify
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy,
                                                            lhsVal == rhsVal);
                return success();
            }

            // Right-hand side isn't a boolean, but is it a constant?
            // If so, we know this comparison will be false
            if (matchPattern(rhs, mlir::m_Constant())) {
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy,
                                                            false);
                return success();
            }
        } else if (!lhs.isa<BlockArgument>() &&
                   matchPattern(rhs, rhsBoolPattern)) {
            // Left-hand side isn't a boolean, but is it a constant?
            // If so, we know this comparison will be false
            if (matchPattern(lhs, mlir::m_Constant())) {
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy,
                                                            false);
                return success();
            }
        }

        auto lhsTy = lhs.getType();
        auto rhsTy = rhs.getType();
        auto termTy = rewriter.getType<TermType>();

        // If the types of the operands can never be equal, then we have our
        // answer
        if (!canTypesEverBeEqual(lhsTy, rhsTy, strict)) {
            rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy, false);
            return success();
        }

        // Handle constant atoms/integers/floats
        APInt lhsInt;
        if (matchPattern(lhs, m_ConstInt(&lhsInt))) {
            APInt rhsInt;
            if (matchPattern(rhs, m_ConstInt(&rhsInt))) {
                bool areEqual = areAPIntsEqual(lhsInt, rhsInt);
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy,
                                                            areEqual);
                return success();
            }
        }

        APInt lhsAtom;
        if (matchPattern(lhs, m_ConstAtom(&lhsAtom))) {
            APInt rhsAtom;
            if (matchPattern(rhs, m_ConstAtom(&rhsAtom))) {
                bool areEqual = areAPIntsEqual(lhsAtom, rhsAtom);
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy,
                                                            areEqual);
                return success();
            }
        }

        auto &semantics = llvm::APFloatBase::IEEEdouble();
        APFloat lhsFloat(semantics, APInt::getNullValue(64));
        if (matchPattern(lhs, m_ConstFloat(&lhsFloat))) {
            APFloat rhsFloat(semantics, APInt::getNullValue(64));
            if (matchPattern(rhs, m_ConstFloat(&rhsFloat))) {
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(
                    op, resultTy, lhsFloat == rhsFloat);
                return success();
            }
        }

        // One case we often see due to canonicalization is a sequence of
        // constant -> cast -> cmp.eq, where the cast is not needed, so
        // recognize this sequence and handle it specially
        //
        // There is also a variant where one operand is a constant cast,
        // and the other is a constant, but not both, and we handle that
        // here as well
        bool lhsBool, rhsBool;
        if (matchPattern(lhs, mlir::m_Op<CastOp>(m_ConstBool(&lhsBool)))) {
            if (matchPattern(rhs, mlir::m_Op<CastOp>(m_ConstBool(&rhsBool)))) {
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy,
                                                            lhsBool == rhsBool);
                return success();
            } else if (matchPattern(rhs, m_ConstBool(&rhsBool))) {
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy,
                                                            lhsBool == rhsBool);
                return success();
            }
        } else if (matchPattern(rhs,
                                mlir::m_Op<CastOp>(m_ConstBool(&rhsBool)))) {
            if (matchPattern(lhs, m_ConstBool(&lhsBool))) {
                rewriter.replaceOpWithNewOp<ConstantBoolOp>(op, resultTy,
                                                            lhsBool == rhsBool);
                return success();
            }
        }

        // If the source types match, we're good
        if (lhsTy == rhsTy) {
            return success();
        }

        // Handle mixed bools even for non-constant operands
        if (lhsTy.isInteger(1)) {
            if (rhsTy.isa<BooleanType>()) {
                auto rhsCast = rewriter.create<CastOp>(op.getLoc(), rhs, i1Ty);
                rewriter.replaceOpWithNewOp<CmpEqOp>(
                    op, lhs, rhsCast.getResult(), strict);
                return success();
            }
        } else if (rhsTy.isInteger(1)) {
            if (lhsTy.isa<BooleanType>()) {
                auto lhsCast = rewriter.create<CastOp>(op.getLoc(), lhs, i1Ty);
                rewriter.replaceOpWithNewOp<CmpEqOp>(op, lhsCast.getResult(),
                                                     rhs, strict);
                return success();
            }
        }

        /*
        // Another common case is a sequence of:
        //
        //   val/op -> cast \
        //                  cmp.eq
        //   val/op -> cast /
        //
        // The casts are generally to term type, and the original
        // values can be lowered to a primitive comparison instead.
        //
        // The fold operation canonicalizes the order of
        // the operands so we can match these patterns pretty
        // simply.
        //
        // For now, we're simply looking at boolean comparisons,
        // since we often have a mixture of i1 and !eir.bool values,
        // resulting in these kind of redundant casts. If the target
        // type of the operands is either i1 or !eir.bool, we can
        // optimize to compare with i1 and remove useless casts
        //
        // If the source types match, we also strip the casts in those
        // cases, but it is unlikely that the IR ever enters that state,
        // since the casts are only introduced when the types differ; but
        // it is possible that optimization may result in such IR being
        // generated, so we handle it while we're here
        if (lhsTy.isa<TermType>() && rhsTy.isa<TermType>()) {
          Value lhsVal;
          if (matchPattern(lhs, m_Cast(&lhsVal))) {
            Value rhsVal;
            if (matchPattern(rhs, m_Cast(&rhsVal))) {
              Type lhsValTy = lhsVal.getType();
              Type rhsValTy = rhsVal.getType();

              // Handle mixed bools
              if (lhsValTy.isInteger(1)) {
                if (rhsValTy.isa<BooleanType>()) {
                  auto rhsCast = rewriter.create<CastOp>(op.getLoc(), rhsVal,
        i1Ty); rewriter.replaceOpWithNewOp<CmpEqOp>(op, resultTy, lhsVal,
        rhsCast.getResult()); return success();
                }
              } else if (rhsValTy.isInteger(1)) {
                if (lhsValTy.isa<BooleanType>()) {
                  auto lhsCast = rewriter.create<CastOp>(op.getLoc(), lhsVal,
        i1Ty); rewriter.replaceOpWithNewOp<CmpEqOp>(op, resultTy,
        lhsCast.getResult(), rhsVal); return success();
                }
              }

              // Strip redundant casts if sources are of the same type
              if (lhsValTy == rhsValTy) {
                rewriter.replaceOpWithNewOp<CmpEqOp>(op, resultTy, lhsVal,
        rhsVal); return success();
              }
            }
          }
        }
        */

        // At this point, one operand is non-constant, or both are constants
        // but do not strictly match. For now, we fall back to calling the
        // builtin, but in the future we should be reason more about these
        // comparisons

        if (!lhsTy.isa<TermType>()) {
            auto lhsCast = rewriter.create<CastOp>(op.getLoc(), lhs, termTy);
            auto lhsOperands = op.lhsMutable();
            lhsOperands.assign(lhsCast.getResult());
        }
        if (!rhsTy.isa<TermType>()) {
            auto rhsCast = rewriter.create<CastOp>(op.getLoc(), rhs, termTy);
            auto rhsOperands = op.rhsMutable();
            rhsOperands.assign(rhsCast.getResult());
        }
        return success();
    }
};
}  // end anonymous namespace

void CmpEqOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
    results.insert<CanonicalizeEqualityComparison>(context);
}

//===----------------------------------------------------------------------===//
// eir.math.*
//===----------------------------------------------------------------------===//

using BinaryIntegerFnT = std::function<Optional<APInt>(APInt &, APInt &)>;
using BinaryFloatFnT = std::function<Optional<APFloat>(APFloat &, APFloat &)>;

static Optional<APInt> foldBinaryIntegerOp(ArrayRef<Attribute> operands,
                                           BinaryIntegerFnT fun) {
    assert(operands.size() == 2 && "binary op takes two operands");

    Attribute lhsAttr = operands[0];
    Attribute rhsAttr = operands[1];

    if (!lhsAttr || !rhsAttr) return llvm::None;

    APInt lhs;
    APInt rhs;
    if (auto lhsInt = lhsAttr.dyn_cast_or_null<APIntAttr>()) {
        lhs = lhsInt.getValue();
    } else if (auto lhsInt = lhsAttr.dyn_cast_or_null<IntegerAttr>()) {
        lhs = lhsInt.getValue();
    } else {
        return llvm::None;
    }
    if (auto rhsInt = rhsAttr.dyn_cast_or_null<APIntAttr>()) {
        rhs = rhsInt.getValue();
    } else if (auto rhsInt = rhsAttr.dyn_cast_or_null<IntegerAttr>()) {
        rhs = rhsInt.getValue();
    } else {
        return llvm::None;
    }

    auto lhsBits = lhs.getBitWidth();
    auto rhsBits = rhs.getBitWidth();
    unsigned defaultBitWidth = 64;
    auto bitWidth = std::max({defaultBitWidth, lhsBits, rhsBits});

    APInt left = normalizeAPInt(lhs, bitWidth);
    APInt right = normalizeAPInt(rhs, bitWidth);

    return fun(left, right);
}

static Optional<APFloat> foldBinaryFloatOp(ArrayRef<Attribute> operands,
                                           BinaryFloatFnT fun) {
    assert(operands.size() == 2 && "binary op takes two operands");

    Attribute lhs = operands[0];
    Attribute rhs = operands[1];

    if (!lhs || !rhs) return llvm::None;

    auto &semantics = llvm::APFloatBase::IEEEdouble();
    APFloat left(semantics);
    APFloat right(semantics);
    if (auto lhsFlt = lhs.dyn_cast_or_null<APFloatAttr>()) {
        left = lhsFlt.getValue();
    } else if (auto lhsFlt = lhs.dyn_cast_or_null<mlir::FloatAttr>()) {
        left = lhsFlt.getValue();
    } else if (auto lhsInt = lhs.dyn_cast_or_null<APIntAttr>()) {
        APInt li = lhsInt.getValue();
        left = APFloat(semantics, APInt::getNullValue(64));
        auto status = left.convertFromAPInt(li, /*signed=*/true,
                                            APFloat::rmNearestTiesToEven);
        if (status != APFloat::opStatus::opOK) return llvm::None;
    } else if (auto lhsInt = lhs.dyn_cast_or_null<IntegerAttr>()) {
        APInt li = lhsInt.getValue();
        left = APFloat(semantics, APInt::getNullValue(64));
        auto status = left.convertFromAPInt(li, /*signed=*/true,
                                            APFloat::rmNearestTiesToEven);
        if (status != APFloat::opStatus::opOK) return llvm::None;
    } else {
        return llvm::None;
    }
    if (auto rhsFlt = rhs.dyn_cast_or_null<APFloatAttr>()) {
        right = rhsFlt.getValue();
    } else if (auto rhsFlt = rhs.dyn_cast_or_null<mlir::FloatAttr>()) {
        right = rhsFlt.getValue();
    } else if (auto rhsInt = rhs.dyn_cast_or_null<APIntAttr>()) {
        APInt ri = rhsInt.getValue();
        right = APFloat(semantics, APInt::getNullValue(64));
        auto status = right.convertFromAPInt(ri, /*signed=*/true,
                                             APFloat::rmNearestTiesToEven);
        if (status != APFloat::opStatus::opOK) return llvm::None;
    } else if (auto rhsInt = rhs.dyn_cast_or_null<IntegerAttr>()) {
        APInt ri = rhsInt.getValue();
        right = APFloat(semantics, APInt::getNullValue(64));
        auto status = right.convertFromAPInt(ri, /*signed=*/true,
                                             APFloat::rmNearestTiesToEven);
        if (status != APFloat::opStatus::opOK) return llvm::None;
    } else {
        return llvm::None;
    }

    if (left.isNaN() || right.isNaN()) return llvm::None;

    return fun(left, right);
}

OpFoldResult NegOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.size() == 1 && "unary op takes one operand");
    Attribute attr = operands[0];

    if (!attr) return nullptr;

    if (auto valInt = attr.dyn_cast_or_null<APIntAttr>()) {
        auto val = valInt.getValue();
        val.negate();
        return APIntAttr::get(getContext(), val);
    } else if (auto valInt = attr.dyn_cast_or_null<IntegerAttr>()) {
        auto val = valInt.getValue();
        val.negate();
        return APIntAttr::get(getContext(), val);
    }

    if (auto flt = attr.dyn_cast_or_null<APFloatAttr>())
        return APFloatAttr::get(getContext(), llvm::neg(flt.getValue()));
    else if (auto flt = attr.dyn_cast_or_null<mlir::FloatAttr>())
        return APFloatAttr::get(getContext(), llvm::neg(flt.getValue()));

    return nullptr;
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
    auto intResult = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) { return lhs + rhs; });

    if (intResult.hasValue()) {
        return APIntAttr::get(getContext(), intResult.getValue());
    }

    auto floatResult = foldBinaryFloatOp(
        operands, [](APFloat &lhs, APFloat &rhs) { return lhs + rhs; });

    if (floatResult.hasValue()) {
        return APFloatAttr::get(getContext(), floatResult.getValue());
    }

    return nullptr;
}

OpFoldResult SubOp::fold(ArrayRef<Attribute> operands) {
    auto intResult = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) { return lhs - rhs; });

    if (intResult.hasValue()) {
        return APIntAttr::get(getContext(), intResult.getValue());
    }

    auto floatResult = foldBinaryFloatOp(
        operands, [](APFloat &lhs, APFloat &rhs) { return lhs - rhs; });

    if (floatResult.hasValue()) {
        return APFloatAttr::get(getContext(), floatResult.getValue());
    }

    return nullptr;
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) { return lhs * rhs; });

    if (result.hasValue()) {
        return APIntAttr::get(getContext(), result.getValue());
    }

    auto floatResult = foldBinaryFloatOp(
        operands, [](APFloat &lhs, APFloat &rhs) { return lhs * rhs; });

    if (floatResult.hasValue()) {
        return APFloatAttr::get(getContext(), floatResult.getValue());
    }

    return nullptr;
}

OpFoldResult FDivOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryFloatOp(
        operands, [](APFloat &lhs, APFloat &rhs) { return lhs / rhs; });

    if (result.hasValue()) {
        return APFloatAttr::get(getContext(), result.getValue());
    }

    return nullptr;
}

OpFoldResult DivOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) { return lhs.sdiv(rhs); });

    if (result.hasValue()) {
        return APIntAttr::get(getContext(), result.getValue());
    }

    return nullptr;
}

OpFoldResult RemOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) { return lhs.srem(rhs); });

    if (result.hasValue()) {
        return APIntAttr::get(getContext(), result.getValue());
    }

    return nullptr;
}

OpFoldResult BandOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) { return lhs & rhs; });

    if (result.hasValue()) {
        return APIntAttr::get(getContext(), result.getValue());
    }

    return nullptr;
}

OpFoldResult BorOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) { return lhs | rhs; });

    if (result.hasValue()) {
        return APIntAttr::get(getContext(), result.getValue());
    }

    return nullptr;
}

OpFoldResult BxorOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) { return lhs ^ rhs; });

    if (result.hasValue()) {
        return APIntAttr::get(getContext(), result.getValue());
    }

    return nullptr;
}

OpFoldResult BslOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) -> Optional<APInt> {
            // APInt doesn't support some valid Erlang shifts
            if (rhs.isNegative()) return llvm::None;

            auto shiftWidth = rhs.getLimitedValue();

            // We can't handle shifts larger than this
            if (shiftWidth > 64) return llvm::None;

            // Zero-extend to new width
            auto lhsBits = lhs.getMinSignedBits();
            auto requiredBits = lhsBits + shiftWidth;
            auto newLhs = lhs.zextOrSelf(requiredBits);
            return newLhs.shl(shiftWidth);
        });

    if (result.hasValue()) {
        return APIntAttr::get(getContext(), result.getValue());
    }

    return nullptr;
}

OpFoldResult BsrOp::fold(ArrayRef<Attribute> operands) {
    auto result = foldBinaryIntegerOp(
        operands, [](APInt &lhs, APInt &rhs) -> Optional<APInt> {
            if (rhs.isNegative()) return llvm::None;

            return lhs.lshr(rhs);
        });

    if (result.hasValue()) {
        return APIntAttr::get(getContext(), result.getValue());
    }

    return nullptr;
}

//===----------------------------------------------------------------------===//
// eir.constant.*
//===----------------------------------------------------------------------===//

OpFoldResult ConstantIntOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

OpFoldResult ConstantBigIntOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

OpFoldResult ConstantFloatOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

OpFoldResult ConstantBoolOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

OpFoldResult ConstantAtomOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

OpFoldResult ConstantNilOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

OpFoldResult ConstantNoneOp::fold(ArrayRef<Attribute> operands) {
    assert(operands.empty() && "constant has no operands");
    return getValue();
}

//===----------------------------------------------------------------------===//
// eir.neg
//===----------------------------------------------------------------------===//

namespace {
/// Fold negations of constants into negated constants
struct ApplyConstantNegations : public OpRewritePattern<NegOp> {
    using OpRewritePattern<NegOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(NegOp op,
                                  PatternRewriter &rewriter) const override {
        auto rhs = op.rhs();

        APInt intVal;
        auto intPattern = mlir::m_Op<CastOp>(m_ConstInt(&intVal));
        if (matchPattern(rhs, intPattern)) {
            auto castOp = dyn_cast<CastOp>(rhs.getDefiningOp());
            auto castType = castOp.getType();
            intVal.negate();
            auto newInt = rewriter.create<ConstantIntOp>(op.getLoc(), intVal);
            rewriter.replaceOpWithNewOp<CastOp>(op, newInt.getResult(),
                                                castType);
            return success();
        }

        APFloat fltVal(0.0);
        auto floatPattern = mlir::m_Op<CastOp>(m_ConstFloat(&fltVal));
        if (matchPattern(rhs, floatPattern)) {
            auto castType = dyn_cast<CastOp>(rhs.getDefiningOp()).getType();
            APFloat newFltVal = -fltVal;
            auto newFlt =
                rewriter.create<ConstantFloatOp>(op.getLoc(), newFltVal);
            rewriter.replaceOpWithNewOp<CastOp>(op, newFlt.getResult(),
                                                castType);
            return success();
        }

        return failure();
    }
};
}  // end anonymous namespace

void NegOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
    results.insert<ApplyConstantNegations>(context);
}

//===----------------------------------------------------------------------===//
// eir.malloc
//===----------------------------------------------------------------------===//

static LogicalResult verify(MallocOp op) {
    auto resultType = op.getResult().getType();
    if (!resultType.isa<PtrType>())
        return op.emitOpError("result must be of pointer type");

    auto allocType = op.getAllocType();
    if (auto allocTypeInfo = dyn_cast<TermTypeInterface>(allocType)) {
        if (!allocTypeInfo.isBoxable())
            return op.emitOpError("cannot malloc an unboxable term type");

        // Optional<arity> in `arguments` is `Value()`, that is `Value(nullptr)`
        // if not given and not an `Optional<Value>`.  MLIR `arguments`
        // `Optional` is unfortunately not related to `llvm::Optional`.
        Value arityVal = op.arity();
        if (arityVal != nullptr) {
            if (allocTypeInfo.isImmediate())
                return op.emitOpError(
                    "it is invalid to specify arity with statically-sized "
                    "type");
        } else {
            if (!allocTypeInfo.isImmediate())
                return op.emitOpError(
                    "cannot malloc a type with dynamic extent without "
                    "specifying "
                    "arity");
        }
    } else {
        return op.emitOpError(
            "it is currently unsupported to malloc non-term types");
    }

    return success();
}

namespace {
/// Fold malloc operations with no uses. Malloc has side effects on the heap,
/// but can still be deleted if it has zero uses.
struct SimplifyDeadMalloc : public OpRewritePattern<MallocOp> {
    using OpRewritePattern<MallocOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MallocOp alloc,
                                  PatternRewriter &rewriter) const override {
        if (alloc.use_empty()) {
            rewriter.eraseOp(alloc);
            return success();
        }
        return failure();
    }
};
}  // end anonymous namespace.

void MallocOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                           MLIRContext *context) {
    results.insert<SimplifyDeadMalloc>(context);
}

//===----------------------------------------------------------------------===//
// eir.invoke
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange> InvokeOp::getMutableSuccessorOperands(
    unsigned index) {
    assert(index < getNumSuccessors() && "invalid successor index");
    return index == okIndex ? llvm::None : Optional(errDestOperandsMutable());
}

//===----------------------------------------------------------------------===//
// eir.yield.check
//===----------------------------------------------------------------------===//

Optional<MutableOperandRange> YieldCheckOp::getMutableSuccessorOperands(
    unsigned index) {
    assert(index < getNumSuccessors() && "invalid successor index");
    return index == trueIndex ? trueDestOperandsMutable()
                              : falseDestOperandsMutable();
}

Block *YieldCheckOp::getSuccessorForOperands(ArrayRef<Attribute> operands) {
    if (IntegerAttr condAttr = operands.front().dyn_cast_or_null<IntegerAttr>())
        return condAttr.getValue().isOneValue() ? trueDest() : falseDest();
    return nullptr;
}

//===----------------------------------------------------------------------===//
// eir.cons
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeCons : public OpRewritePattern<ConsOp> {
    using OpRewritePattern<ConsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ConsOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        Value head = op.head();
        Value h = castToTermEquivalent(rewriter, head);
        if (h != head) {
            auto headOperand = op.headMutable();
            headOperand.assign(h);
        }

        Value tail = op.tail();
        Value t = castToTermEquivalent(rewriter, tail);
        if (t != tail) {
            auto tailOperand = op.tailMutable();
            tailOperand.assign(t);
        }

        return success();
    }
};
}  // end anonymous namespace.

void ConsOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
    results.insert<CanonicalizeCons>(context);
}

//===----------------------------------------------------------------------===//
// eir.list
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeList : public OpRewritePattern<ListOp> {
    using OpRewritePattern<ListOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ListOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        auto elementOperandsMut = op.elementsMutable();
        if (elementOperandsMut.size() == 0) return success();

        OperandRange elementOperands(elementOperandsMut);
        auto index = elementOperands.getBeginOperandIndex();
        for (auto element : op.elements()) {
            Value e = castToTermEquivalent(rewriter, element);
            if (e != element) {
                elementOperandsMut.slice(index, 1).assign(e);
            }
            index++;
        }

        return success();
    }
};
}  // end anonymous namespace.

void ListOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
    results.insert<CanonicalizeList>(context);
}

//===----------------------------------------------------------------------===//
// eir.tuple
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeTuple : public OpRewritePattern<TupleOp> {
    using OpRewritePattern<TupleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TupleOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        auto elementOperandsMut = op.elementsMutable();
        if (elementOperandsMut.size() == 0) return success();

        OperandRange elementOperands(elementOperandsMut);
        auto index = elementOperands.getBeginOperandIndex();
        for (auto element : op.elements()) {
            Value e = castToTermEquivalent(rewriter, element);
            if (e != element) {
                elementOperandsMut.slice(index, 1).assign(e);
            }
            index++;
        }

        return success();
    }
};
}  // end anonymous namespace.

void TupleOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
    results.insert<CanonicalizeTuple>(context);
}

//===----------------------------------------------------------------------===//
// eir.map.*
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeMap : public OpRewritePattern<MapOp> {
    using OpRewritePattern<MapOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(MapOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        auto argsOperandsMut = op.argsMutable();
        if (argsOperandsMut.size() == 0) return success();

        OperandRange argsOperands(argsOperandsMut);
        auto index = argsOperands.getBeginOperandIndex();
        for (auto arg : op.args()) {
            Value a = castToTermEquivalent(rewriter, arg);
            if (a != arg) {
                argsOperandsMut.slice(index, 1).assign(a);
            }
            index++;
        }

        return success();
    }
};

template <typename OpType>
struct CanonicalizeMapMutation : public OpRewritePattern<OpType> {
    using OpRewritePattern<OpType>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpType op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        Value map = op.map();
        Value m = castToTermEquivalent(rewriter, map);
        if (m != map) {
            auto mapOperand = op.mapMutable();
            mapOperand.assign(m);
        }

        Value key = op.key();
        Value k = castToTermEquivalent(rewriter, key);
        if (k != key) {
            auto keyOperand = op.keyMutable();
            keyOperand.assign(k);
        }

        Value val = op.val();
        Value v = castToTermEquivalent(rewriter, val);
        if (v != val) {
            auto valOperand = op.valMutable();
            valOperand.assign(v);
        }

        return success();
    }
};

template <typename OpType>
struct CanonicalizeMapKeyOp : public OpRewritePattern<OpType> {
    using OpRewritePattern<OpType>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpType op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        Value map = op.map();
        Value m = castToTermEquivalent(rewriter, map);
        if (m != map) {
            auto mapOperand = op.mapMutable();
            mapOperand.assign(m);
        }

        Value key = op.key();
        Value k = castToTermEquivalent(rewriter, key);
        if (k != key) {
            auto keyOperand = op.keyMutable();
            keyOperand.assign(k);
        }

        return success();
    }
};
}  // end anonymous namespace.

void MapOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                        MLIRContext *context) {
    results.insert<CanonicalizeMap>(context);
}

void MapInsertOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
    results.insert<CanonicalizeMapMutation<MapInsertOp>>(context);
}

void MapUpdateOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
    results.insert<CanonicalizeMapMutation<MapUpdateOp>>(context);
}

void MapContainsKeyOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeMapKeyOp<MapContainsKeyOp>>(context);
}

void MapGetKeyOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
    results.insert<CanonicalizeMapKeyOp<MapGetKeyOp>>(context);
}

//===----------------------------------------------------------------------===//
// eir.binary.push
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeBinaryPush : public OpRewritePattern<BinaryPushOp> {
    using OpRewritePattern<BinaryPushOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(BinaryPushOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        Value value = op.value();
        Value v = castToTermEquivalent(rewriter, value);
        if (v != value) {
            auto valueOperand = op.valueMutable();
            valueOperand.assign(v);
        }

        // Optional<size> in `arguments` is `Value()`, that is `Value(nullptr)`
        // if not given and not an `Optional<Value>`.  MLIR `arguments`
        // `Optional` is unfortunately not related to `llvm::Optional`.
        Value size = op.size();
        if (size == nullptr) return success();

        Value s = castToTermEquivalent(rewriter, size);
        if (s != size) {
            auto sizeOperand = op.sizeMutable();
            sizeOperand.assign(s);
        }

        return success();
    }
};
}  // end anonymous namespace.

void BinaryPushOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeBinaryPush>(context);
}

//===----------------------------------------------------------------------===//
// eir.binary.match.*
//===----------------------------------------------------------------------===//

namespace {
template <typename OpType>
struct CanonicalizeSizedBinaryMatch : public OpRewritePattern<OpType> {
    using OpRewritePattern<OpType>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpType op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        Value bin = op.bin();
        Value b = castToTermEquivalent(rewriter, bin);
        if (b != bin) {
            auto binOperand = op.binMutable();
            binOperand.assign(b);
        }

        Optional<Value> sizeOpt = op.size();
        if (!sizeOpt.hasValue()) return success();

        Value size = sizeOpt.getValue();
        Value s = castToTermEquivalent(rewriter, size);
        if (s != size) {
            auto sizeOperand = op.sizeMutable();
            sizeOperand.assign(s);
        }

        return success();
    }
};
}  // end anonymous namespace.

void BinaryMatchRawOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeSizedBinaryMatch<BinaryMatchRawOp>>(context);
}

void BinaryMatchIntegerOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeSizedBinaryMatch<BinaryMatchIntegerOp>>(context);
}

void BinaryMatchFloatOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeSizedBinaryMatch<BinaryMatchFloatOp>>(context);
}

void BinaryMatchUtf8Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeSizedBinaryMatch<BinaryMatchUtf8Op>>(context);
}

void BinaryMatchUtf16Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeSizedBinaryMatch<BinaryMatchUtf16Op>>(context);
}

void BinaryMatchUtf32Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeSizedBinaryMatch<BinaryMatchUtf32Op>>(context);
}

//===----------------------------------------------------------------------===//
// eir.receive.*
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizeReceiveStart : public OpRewritePattern<ReceiveStartOp> {
    using OpRewritePattern<ReceiveStartOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ReceiveStartOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.use_empty()) {
            rewriter.eraseOp(op);
            return success();
        }

        Value timeout = op.timeout();
        Value t = castToTermEquivalent(rewriter, timeout);
        if (t != timeout) {
            auto operand = op.timeoutMutable();
            operand.assign(t);
        }

        return success();
    }
};
}  // end anonymous namespace.

void ReceiveStartOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
    results.insert<CanonicalizeReceiveStart>(context);
}

//===----------------------------------------------------------------------===//
// eir.print
//===----------------------------------------------------------------------===//

namespace {
struct CanonicalizePrint : public OpRewritePattern<PrintOp> {
    using OpRewritePattern<PrintOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PrintOp op,
                                  PatternRewriter &rewriter) const override {
        Value thing = op.thing();
        Value t = castToTermEquivalent(rewriter, thing);
        if (t != thing) {
            auto operand = op.thingMutable();
            operand.assign(t);
        }

        return success();
    }
};
}  // end anonymous namespace.

void PrintOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
    results.insert<CanonicalizePrint>(context);
}

}  // namespace eir
}  // namespace lumen

//===----------------------------------------------------------------------===//
// TableGen Output
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "lumen/EIR/IR/EIROps.cpp.inc"
