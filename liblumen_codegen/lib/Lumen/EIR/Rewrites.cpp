#include "eir/Types.h"
#include "eir/Ops.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/ErrorHandling.h"

namespace M = mlir;

using namespace eir;

namespace eir {
// Returns 0 for false, 1 for true, 2 for unknown
static unsigned matchType(M::Type valueType, M::Type matchType) {
  // TermType is a generic term, so can't be statically resolved
  auto valueBaseType = valueType.dyn_cast_or_null<TermBase>();
  if (!valueBaseType)
    return 2;

  return valueBaseType.isMatch(matchType);
}
}

/// This rewrite pattern optimizes usages of eir.is_type that has statically resolvable
/// type information to boolean constants of either i1 or !eir.boolean type,
/// depending on the result type of the operation.
struct ReplaceStaticallyResolvableIsTypeOps : public M::OpRewritePattern<IsTypeOp> {
  /// We register this pattern to match every eir.is_type operation in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  ReplaceStaticallyResolvableIsTypeOps(M::MLIRContext *context)
      : OpRewritePattern<IsTypeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  M::PatternMatchResult
  matchAndRewrite(IsTypeOp op, M::PatternRewriter &rewriter) const override {
    auto valueType = op.getValueType();
    auto matchType = op.getMatchType();

    // !eir.is_type %0 : T { "type" = T } : U => true : U
    //   - T must be a concrete type, i.e. not !eir.term
    //
    // !eir.is_type %0 : T1 { "type" = T2 } : U => false : U
    //   - T1 is either not concrete, or
    //   - T1 is a concrete type, and does not match T2
    switch (eir::matchType(valueType, matchType)) {
    case 0:
      rewriteToConstantBool(op, rewriter, false);
      return matchSuccess();
    case 1:
      rewriteToConstantBool(op, rewriter, true);
      return matchSuccess();
    case 2:
      return matchFailure();
    default:
      llvm_unreachable("unexpected value returned from matchType!");
    }
  }

  void
  rewriteToConstantBool(IsTypeOp op, M::PatternRewriter &rewriter, bool value) const {
    auto resultType = op.getResultType();

    if (resultType.isInteger(1)) {
      // Build constant boolean as i1
      rewriter.replaceOpWithNewOp<M::ConstantOp>(op, rewriter.getBoolAttr(value));
    } else {
      // Build constant atom equivalent of boolean value
      assert(resultType.isa<BooleanType>());
      rewriter.replaceOpWithNewOp<ConstantAtomOp>(op,
                                                  value ? "true" : "false",
                                                  L::APInt(64, value ? 1 : 0));
    }
  }
};

/// Register our patterns as canonicalization patterns on the IsTypeOp so
/// that they can be picked up by the canonicalization framework.
void IsTypeOp::getCanonicalizationPatterns(M::OwningRewritePatternList &results,
                                           M::MLIRContext *context) {
  results.insert<ReplaceStaticallyResolvableIsTypeOps>(context);
}
