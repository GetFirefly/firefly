#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"

#include "lumen/EIR/IR/EIROps.h"
#include "lumen/EIR/IR/EIRTypes.h"
#include "lumen/term/Encoding.h"

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;
using ::mlir::Attribute;
using ::mlir::Block;
using ::mlir::FlatSymbolRefAttr;
using ::mlir::Location;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::OperationState;
using ::mlir::Region;
using ::mlir::Type;
using ::mlir::Value;

Operation *getDefinition(Value val);

template <typename OpTy>
OpTy getDefinition(Value val) {
    Operation *definition = getDefinition(val);
    return llvm::dyn_cast_or_null<OpTy>(definition);
}

extern "C" MlirOperation mlirEirGetDefinition(MlirValue value) {
    return wrap(getDefinition(unwrap(value)));
}

extern "C" MlirOperation mlirEirGetClosureDefinition(MlirValue closure) {
    if (auto closureOp = getDefinition<ClosureOp>(unwrap(closure))) {
        return wrap(closureOp.getOperation());
    } else {
        return MlirOperation{nullptr};
    }
}

#define DEFINE_C_API_STRUCT(name, storage) \
    struct name {                          \
        storage *ptr;                      \
    };                                     \
    typedef struct name name

DEFINE_C_API_STRUCT(MlirOpBuilder, void);

DEFINE_C_API_PTR_METHODS(MlirOpBuilder, mlir::OpBuilder);

// Types

#define DEFINE_C_API_TYPE(name)                                       \
    extern "C" MlirType mlirEirGetType #name(MlirOpBuilder builder) { \
        return unwrap(builder)->getType<name #Type>();                \
    }

#define DEFINE_C_API_TYPE_WITH_SUBTYPE(name)                        \
    extern "C" MlirType mlirEirGetType #name(MlirOpBuilder builder, \
                                             MlirType ty) {         \
        return unwrap(builder)->getType<name #Type>(unwrap(ty));    \
    }

#define DEFINE_C_API_TYPE_WITH_ELEMENTS(name)                                  \
    extern "C" MlirType mlirEirGetType #name(MlirOpBuilder builder) {          \
        return unwrap(builder)->getType<name #Type>();                         \
    }                                                                          \
    extern "C" MlirType mlirEirGetType #name #WithArity(MlirOpBuilder builder, \
                                                        size_t arity) {        \
        return unwrap(builder)->getType<name #Type>(arity);                    \
    }                                                                          \
    extern "C" MlirType mlirEirGetType #name #WithElements(                    \
        MlirOpBuilder builder, MlirType *elems, size_t len) {                  \
        SmallVector<Type, 2> storage;                                          \
        ArrayRef<Type> elements = unwrapList(elems, len, storage);             \
        return unwrap(builder)->getType<name #Type>(elements);                 \
    }

DEFINE_C_API_TYPE(None);
DEFINE_C_API_TYPE(Term);
DEFINE_C_API_TYPE(List);
DEFINE_C_API_TYPE(Number);
DEFINE_C_API_TYPE(Integer);
DEFINE_C_API_TYPE(Float);
DEFINE_C_API_TYPE(Atom);
DEFINE_C_API_TYPE(Boolean);
DEFINE_C_API_TYPE(Nil);
DEFINE_C_API_TYPE(Cons);
DEFINE_C_API_TYPE(Map);
DEFINE_C_API_TYPE(Binary);
DEFINE_C_API_TYPE(Pid);
DEFINE_C_API_TYPE(Reference);
DEFINE_C_API_TYPE(TraceRef);
DEFINE_C_API_TYPE(ReceiveRef);
DEFINE_C_API_TYPE_WITH_SUBTYPE(Box);
DEFINE_C_API_TYPE_WITH_SUBTYPE(Ptr);
DEFINE_C_API_TYPE_WITH_SUBTYPE(Ref);
DEFINE_C_API_TYPE_WITH_ELEMENTS(Tuple);
DEFINE_C_API_TYPE_WITH_ELEMENTS(Closure);

extern "C" MlirType mlirEirGetPointeeType(MlirType t, size_t index) {
    Type ty = unwrap(t);
    if (!ty) return MlirType{nullptr};

    if (auto ptrTy = ty.dyn_cast_or_null<PtrType>()) {
        auto innerTy = ptrTy.getPointeeType();
        if (auto tupleTy = innerType.dyn_cast_or_null<TupleType>()) {
            auto elements = tupleTy.getTypes();
            if (elements.size() > 0)
                return wrap(elements[index]);
            else
                return wrap(TermType::get(ty.getContext()));
        } else if (innerType.isa<ConsType>()) {
            return wrap(TermType::get(ty.getContext()));
        } else {
            return MlirType{nullptr};
        }
    } else {
        return MlirType{nullptr};
    }
}

// Operations

#define DEFINE_DYN_CAST_IMPL(name)                                          \
    extern "C" MlirOperation mlirOperationDynCast #name(MlirOperation op) { \
        Operation *ptr =                                                    \
            unwrap(op) if (ptr != nullptr && isa<#name #>(op)) return op;   \
        return MlirOperation{nullptr};                                      \
    }

DEFINE_DYN_CAST_IMPL(FuncOp);
DEFINE_DYN_CAST_IMPL(ClosureOp);
DEFINE_DYN_CAST_IMPL(UnpackEnvOp);
DEFINE_DYN_CAST_IMPL(IsTypeOp);
DEFINE_DYN_CAST_IMPL(LogicalAndOp);
DEFINE_DYN_CAST_IMPL(LogicalOrOp);
DEFINE_DYN_CAST_IMPL(CmpEqOp);
DEFINE_DYN_CAST_IMPL(CmpLtOp);
DEFINE_DYN_CAST_IMPL(CmpLteOp);
DEFINE_DYN_CAST_IMPL(CmpGtOp);
DEFINE_DYN_CAST_IMPL(CmpGteOp);
DEFINE_DYN_CAST_IMPL(NegOp);
DEFINE_DYN_CAST_IMPL(AddOp);
DEFINE_DYN_CAST_IMPL(SubOp);
DEFINE_DYN_CAST_IMPL(MulOp);
DEFINE_DYN_CAST_IMPL(DivOp);
DEFINE_DYN_CAST_IMPL(FDivOp);
DEFINE_DYN_CAST_IMPL(RemOp);
DEFINE_DYN_CAST_IMPL(BslOp);
DEFINE_DYN_CAST_IMPL(BsrOp);
DEFINE_DYN_CAST_IMPL(BandOp);
DEFINE_DYN_CAST_IMPL(BorOp);
DEFINE_DYN_CAST_IMPL(BxorOp);
DEFINE_DYN_CAST_IMPL(BranchOp);
DEFINE_DYN_CAST_IMPL(CondBranchOp);
DEFINE_DYN_CAST_IMPL(CallOp);
DEFINE_DYN_CAST_IMPL(InvokeOp);
DEFINE_DYN_CAST_IMPL(LandingPadOp);
DEFINE_DYN_CAST_IMPL(ReturnOp);
DEFINE_DYN_CAST_IMPL(YieldOp);
DEFINE_DYN_CAST_IMPL(YieldCheckOp);
DEFINE_DYN_CAST_IMPL(UnreachableOp);
DEFINE_DYN_CAST_IMPL(ThrowOp);
DEFINE_DYN_CAST_IMPL(IncrementReductionsOp);
DEFINE_DYN_CAST_IMPL(CastOp);
DEFINE_DYN_CAST_IMPL(MallocOp);
DEFINE_DYN_CAST_IMPL(LoadOp);
DEFINE_DYN_CAST_IMPL(GetElementPtrOp);
DEFINE_DYN_CAST_IMPL(PrintOp);
DEFINE_DYN_CAST_IMPL(NullOp);
DEFINE_DYN_CAST_IMPL(ConstantIntOp);
DEFINE_DYN_CAST_IMPL(ConstantFloatOp);
DEFINE_DYN_CAST_IMPL(ConstantBigIntOp);
DEFINE_DYN_CAST_IMPL(ConstantBooleanOp);
DEFINE_DYN_CAST_IMPL(ConstantAtomOp);
DEFINE_DYN_CAST_IMPL(ConstantBinaryOp);
DEFINE_DYN_CAST_IMPL(ConstantNilOp);
DEFINE_DYN_CAST_IMPL(ConstantNoneOp);
DEFINE_DYN_CAST_IMPL(ConstantTupleOp);
DEFINE_DYN_CAST_IMPL(ConstantListOp);
DEFINE_DYN_CAST_IMPL(ConstantMapOp);
DEFINE_DYN_CAST_IMPL(ConsOp);
DEFINE_DYN_CAST_IMPL(ListOp);
DEFINE_DYN_CAST_IMPL(TupleOp);
DEFINE_DYN_CAST_IMPL(TraceCaptureOp);
DEFINE_DYN_CAST_IMPL(TracePrintOp);
DEFINE_DYN_CAST_IMPL(TraceConstructOp);
DEFINE_DYN_CAST_IMPL(MapOp);
DEFINE_DYN_CAST_IMPL(MapInsertOp);
DEFINE_DYN_CAST_IMPL(MapUpdateOp);
DEFINE_DYN_CAST_IMPL(MapContainsKeyOp);
DEFINE_DYN_CAST_IMPL(MapGetKeyOp);
DEFINE_DYN_CAST_IMPL(BinaryStartOp);
DEFINE_DYN_CAST_IMPL(BinaryFinishOp);
DEFINE_DYN_CAST_IMPL(BinaryPushOp);
DEFINE_DYN_CAST_IMPL(BinaryMatchRawOp);
DEFINE_DYN_CAST_IMPL(BinaryMatchIntegerOp);
DEFINE_DYN_CAST_IMPL(BinaryMatchFloatOp);
DEFINE_DYN_CAST_IMPL(BinaryMatchUtf8Op);
DEFINE_DYN_CAST_IMPL(BinaryMatchUtf16Op);
DEFINE_DYN_CAST_IMPL(BinaryMatchUtf32Op);
DEFINE_DYN_CAST_IMPL(ReceiveStartOp);
DEFINE_DYN_CAST_IMPL(ReceiveWaitOp);
DEFINE_DYN_CAST_IMPL(ReceiveMessageOp);
DEFINE_DYN_CAST_IMPL(ReceiveDoneOp);

// Recursively searches for the Operation which defines the given value
Operation *getDefinition(Value val) {
    if (auto arg = val.dyn_cast_or_null<BlockArgument>()) {
        Block *block = arg.getOwner();
        // If this block is the entry block, then we can't get the definition
        if (block->isEntryBlock()) return nullptr;
        // If this block has no predecessors, then we can't get the definition
        if (block->hasNoPredecessors()) return nullptr;
        // If there is a single predecessor, check the value passed as argument
        // to this block.
        //
        // If this block has multiple predecessors, we need to check if the
        // argument traces back to a single value; otherwise there are different
        // values in different branches, and we can't get a single definition
        Operation *result = nullptr;
        for (Block *pred : block->getPredecessors()) {
            auto index = arg.getArgNumber();
            Operation *found = nullptr;
            pred->walk([&](BranchOpInterface branchInterface) {
                for (auto it = pred->succ_begin(), e = pred->succ_end();
                     it != e; ++it) {
                    // If the successor isn't our block, we don't care
                    Block *succ = *it;
                    if (block != succ) continue;
                    // No operands, nothing to do
                    auto maybeSuccessorOperands =
                        branchInterface.getSuccessorOperands(it.getIndex());
                    if (!maybeSuccessorOperands.hasValue()) continue;
                    // Otherwise take a look at the value passed as the
                    // successor block argument
                    auto successorOperands = maybeSuccessorOperands.getValue();
                    Value candidate = successorOperands[index];
                    Operation *def = getDefinition(candidate);
                    if (found && def != found) return WalkResult::interrupt();
                    found = def;
                }
                return WalkResult::advance();
            });
            // If this result doesn't match the last, we've found a conflict
            if (result && found != result) return nullptr;
            result = found;
        }

        return nullptr;
    }
    return val.getDefiningOp();
}
