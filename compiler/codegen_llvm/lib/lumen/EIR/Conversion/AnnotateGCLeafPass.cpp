#include "lumen/EIR/Conversion/AnnotateGCLeafPass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

using ::mlir::BoolAttr;
using ::mlir::Builder;
using ::mlir::DialectRegistry;
using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::OperationPass;
using ::mlir::PassWrapper;
using ::mlir::WalkResult;

using ::llvm::StringRef;
using ::llvm::StringSwitch;

namespace LLVM = ::mlir::LLVM;

/// Forward declaration for function that indicates whether
/// a given callee function name is that of a known gc leaf function
bool isKnownLeaf(StringRef);

/// This analysis walks a function and determines if it contains
/// any operations which allocate, or calls a function not known to
/// be a GC leaf function.
struct GCLeafAnalysis {
    GCLeafAnalysis(LLVM::LLVMFuncOp funcOp) : isGCLeaf(true) {
        // For now all external functions are assumed to allocate
        if (funcOp.isExternal()) {
            isGCLeaf = isKnownLeaf(funcOp.getName());
            return;
        }

        // Allow manually overriding this analysis
        if (auto isLeafAttr =
                funcOp.getAttrOfType<BoolAttr>("gc-leaf-function")) {
            isGCLeaf = isLeafAttr.getValue();
            return;
        }

        // Walk all child operations looking for those which represent
        // allocations.
        //
        // Currently the only operations that indicate an allocation are calls
        // to runtime functions which allocate. The calls which we know do not
        // allocate are a set we know a priori
        funcOp.walk([&](Operation *op) {
            if (auto callOp = dyn_cast_or_null<LLVM::CallOp>(op)) {
                // If the callee is not a known allocation-free function, this
                // is not a leaf
                auto callee = callOp.callee();
                if (isKnownLeaf(callee.getValue())) {
                    return WalkResult::advance();
                }
                isGCLeaf = false;
                return WalkResult::interrupt();
            }
            if (auto callOp = dyn_cast_or_null<LLVM::InvokeOp>(op)) {
                // If the callee is not a known allocation-free function, this
                // is not a leaf
                auto callee = callOp.callee();
                if (isKnownLeaf(callee.getValue())) {
                    return WalkResult::advance();
                }
                isGCLeaf = false;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
    }

    bool isGCLeaf;
};

/// This pass annotates every LLVM function that does not allocate,
/// or call a runtime function that can allocate, with the attribute
/// `gc-leaf-function`. Such functions do not require safepoint polls
/// for garbage collection, and so such annotation is important to apply
/// where possible.
///
/// NOTE: This pass does not use FunctionPass as that skips external functions.
struct AnnotateGCLeafPass
    : public PassWrapper<AnnotateGCLeafPass, OperationPass<LLVM::LLVMFuncOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() override {
        LLVM::LLVMFuncOp op = getOperation();
        Builder builder(op.getParentOfType<ModuleOp>());

        auto &analysis = getAnalysis<GCLeafAnalysis>();

        if (analysis.isGCLeaf) {
            op.setAttr("gc-leaf-function", builder.getBoolAttr(true));
        } else {
            op.removeAttr("gc-leaf-function");
        }
    }
};

namespace lumen {
namespace eir {
std::unique_ptr<mlir::Pass> createAnnotateGCLeafPass() {
    return std::make_unique<AnnotateGCLeafPass>();
}
}  // namespace eir
}  // namespace lumen

/// A function containing the known allocation-free functions, this
/// is mostly a workaround for the fact that generally speaking we can't
/// know whether a function allocates or not, but we know that a number
/// of runtime functions do _not_ allocate on the process heap, and so are
/// safe to call from functions marked as leaves.
bool isKnownLeaf(StringRef callee) {
    // Compiler intrinsics are generally allocation free, with a few
    // exceptions. We handle them here.
    if (callee.startswith("__lumen_builtin")) {
        // All binary/receive operations can allocate
        if (callee.startswith("__lumen_builtin_binary") ||
            callee.startswith("__lumen_builtin_receive"))
            return false;

        // All remaining builtins do not allocate, except a couple
        return StringSwitch<bool>(callee)
            .Case("__lumen_builtin_malloc", false)
            .Case("__lumen_builtin_map.new", false)
            .Case("__lumen_builtin_map.insert", false)
            .Case("__lumen_builtin_map.update", false)
            .Default(true);
    }

    // BIFs in the `erlang` module generally allocate, but not all,
    // those that do not are special-cased here
    return StringSwitch<bool>(callee)
        .Case("erlang:error/1", true)
        .Case("erlang:error/2", true)
        .Case("erlang:exit/1", true)
        .Case("erlang:throw/1", true)
        .Case("erlang:raise/3", true)
        .Case("erlang:print/1", true)
        .Case("erlang:spawn/1", true)
        .Case("erlang:fail/1", true)
        .Case("erlang:+/2", true)
        .Case("erlang:-/1", true)
        .Case("erlang:-/2", true)
        .Case("erlang:*/2", true)
        .Case("erlang:div/2", true)
        .Case("erlang:rem/2", true)
        .Case("erlang://2", true)
        .Case("erlang:bsl/2", true)
        .Case("erlang:bsr/2", true)
        .Case("erlang:band/2", true)
        .Case("erlang:bnot/2", true)
        .Case("erlang:bor/2", true)
        .Case("erlang:bxor/2", true)
        .Case("erlang:=:=/2", true)
        .Case("erlang:=/=/2", true)
        .Case("erlang:==/2", true)
        .Case("erlang:/=/2", true)
        .Case("erlang:</2", true)
        .Case("erlang:=</2", true)
        .Case("erlang:>/2", true)
        .Case("erlang:>=/2", true)
        .Default(false);
}
