#include "lumen/EIR/Builder/Passes.h"
#include "lumen/EIR/IR/EIRDialect.h"
#include "lumen/EIR/IR/EIROps.h"
#include "lumen/EIR/IR/EIRTypes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/Support/Casting.h"

using ::mlir::OpBuilder;
using ::mlir::DialectRegistry;
using ::mlir::PassWrapper;
using ::mlir::OperationPass;
using ::mlir::Block;
using ::mlir::BlockArgument;
using ::mlir::Value;
using ::mlir::Location;
using ::mlir::Operation;
using ::mlir::OpOperand;

using ::llvm::isa;
using ::llvm::dyn_cast_or_null;
using ::llvm::cast;

namespace {

using namespace ::lumen::eir;
    
void forAllTraceUses(OpBuilder &, Location, Value, Value, unsigned);
    
struct InsertTraceConstructorsPass
    : public PassWrapper<InsertTraceConstructorsPass, OperationPass<FuncOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect, mlir::LLVM::LLVMDialect,
                    lumen::eir::eirDialect>();
  }

  void runOnOperation() override {
    FuncOp op = getOperation();
    if (op.isExternal())
      return;

    OpBuilder builder(op.getParentOfType<ModuleOp>());
    op.walk([&](LandingPadOp landingPad) {
        Value trace = landingPad.trace();
        forAllTraceUses(builder, landingPad.getLoc(), trace, Value(), 0);
    });

    return;
  }
};

void forAllTraceUses(OpBuilder &builder, Location loc, Value root,
                     Value traceTerm, unsigned depth) {
  for (OpOperand &use : root.getUses()) {
    // Get operation which owns the trace at this point in the control-flow
    // graph
    Operation *user = use.getOwner();
    // If the trace is passed to a branch/conditional branch, then we need to
    // follow it into the successors and update usages that we find in those
    // blocks. While we're doing so, we ensure the type is correct
    if (auto brOp = dyn_cast_or_null<BranchOp>(user)) {
      auto index = use.getOperandNumber();
      auto successor = brOp.getDest();
      BlockArgument arg = successor->getArgument(index);
      arg.setType(builder.getType<TraceRefType>());
      forAllTraceUses(builder, loc, arg, traceTerm, depth + 1);
      continue;
    } else if (auto condbrOp = dyn_cast_or_null<CondBranchOp>(user)) {
      auto index = use.getOperandNumber();
      auto numTrueOperands = condbrOp.getNumTrueOperands();
      if (index > numTrueOperands) {
        auto successor = condbrOp.getFalseDest();
        BlockArgument arg = successor->getArgument(index - numTrueOperands);
        arg.setType(builder.getType<TraceRefType>());
        forAllTraceUses(builder, loc, arg, traceTerm, depth + 1);
      } else {
        auto successor = condbrOp.getTrueDest();
        BlockArgument arg = successor->getArgument(index);
        arg.setType(builder.getType<TraceRefType>());
        forAllTraceUses(builder, loc, arg, traceTerm, depth + 1);
      }
      continue;
    }

    // Certain operations use the trace reference directly:
    //
    // - ThrowOp uses it when rethrowing an exception
    // - TracePrintOp uses it to print the stacktrace that was captured
    //
    // In all of the above cases, we just skip over those uses
    if (isa<ThrowOp>(user) || isa<TracePrintOp>(user)) {
      continue;
    }

    // At this point, we have an operation that either uses the trace as a term,
    // and so requires construction, or is itself a trace constructor (which
    // expects the trace reference). In both cases, if a trace term was
    // constructed already, we need to determine if that construction dominates
    // this use, and if so, use it instead of the raw trace ref. If it doesn't
    // dominate this use, then we need to either insert a constructor, or leave
    // the constructor we've encountered in place.
    //
    // NOTE: This is not complete, we are not doing proper dominance analysis,
    // and I'm waiting to hit a case where we're generating code that requires
    // it before doing something that expensive in a canonicalization pass,
    // since we'd want to reuse the results of that analysis as much as
    // possible. In short, if we've already constructed a trace term, I'm
    // essentially assuming that it dominates all following uses we encounter,
    // and this code is basically just trying to validate that.
    bool traceTermDominatesUse = false;
    if (traceTerm) {
      Operation *traceTermOp = traceTerm.getDefiningOp();
      Block *traceTermBlock = traceTermOp->getBlock();
      Block *userBlock = user->getBlock();
      if (traceTermBlock == userBlock) {
        // Sanity check
        assert(traceTermOp->isBeforeInBlock(user) &&
               "bug in trace constructor placement");
        traceTermDominatesUse = true;
      } else {
        // Otherwise, check to see if this block is preceded by the
        // block containing the previously constructed trace term.
        //
        // If it is, then go ahead and replace, if it isn't, then
        // we probably need proper dominance analysis here, so raise
        // an assertion
        for (Block *pred : userBlock->getPredecessors()) {
          if (traceTermBlock == pred) {
            traceTermDominatesUse = true;
            break;
          }
        }
        assert(traceTermDominatesUse &&
               "expected trace constructor to dominate this use");
      }
    }

    // If this is a trace constructor use, then based on the results of the
    // above analysis, either replace it, or leave it in place and use the
    // result as the trace term for subsequent users
    if (auto traceCtor = dyn_cast_or_null<TraceConstructOp>(user)) {
      if (traceTerm && traceTermDominatesUse) {
        user->replaceAllUsesWith(ValueRange(traceTerm));
        continue;
      } else {
        traceTerm = traceCtor.trace();
        continue;
      }
    }

    // This is some other operation, which will universally assume that it
    // is a term, not a raw trace reference, so either construct the term right
    // at the use, or use the result of a constructor that was already present
    if (traceTerm && traceTermDominatesUse) {
      use.set(traceTerm);
    } else {
      builder.setInsertionPoint(user);
      traceTerm = builder.create<TraceConstructOp>(loc, root);
      use.set(traceTerm);
    }
  }
}
}


namespace lumen {
namespace eir {
std::unique_ptr<mlir::Pass> createInsertTraceConstructorsPass() {
  return std::make_unique<InsertTraceConstructorsPass>();
}
}
}
