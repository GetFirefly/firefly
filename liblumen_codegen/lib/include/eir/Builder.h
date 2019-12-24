#ifndef EIR_BUILDER_H
#define EIR_BUILDER_H

#include "eir/Context.h"
#include "eir/Types.h"

#include "llvm-c/Types.h"
#include "llvm/Support/CBindingWrapping.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

#include "llvm-c/Core.h"

namespace mlir {
class Location;
class FuncOp;
class Block;
class Builder;
} // namespace mlir

typedef struct MLIROpaqueLocation *MLIRLocationRef;
typedef struct MLIROpaqueFuncOp *MLIRFunctionOpRef;
typedef struct MLIROpaqueBlock *MLIRBlockRef;
typedef struct MLIROpaqueBuilder *MLIRBuilderRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::Location, MLIRLocationRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::FuncOp, MLIRFunctionOpRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::Block, MLIRBlockRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::Builder, MLIRBuilderRef);

namespace eir {

/// A source location in EIR
struct Span {
  // The starting byte index of a span
  int start;
  // The end byte index of a span
  int end;
};

class ModuleBuilder;

enum class EirType {
  Unknown = 0,
  Term,
  Atom,
  Boolean,
  Fixnum,
  BigInt,
  Float,
  FloatPacked,
  Nil,
  Cons,
  Tuple,
  Map,
  Closure,
  HeapBin,
  Box,
  Ref,
};

static EirTypes fromRust(EirType t);

struct Arg {
  EirType ty;
  unsigned span_start;
  unsigned span_end;
};

} // namespace eir

typedef struct OpaqueModuleBuilder *MLIRModuleBuilderRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(eir::ModuleBuilder, MLIRModuleBuilderRef);

extern "C" {
MLIRModuleBuilderRef MLIRCreateModuleBuilder(MLIRContextRef context,
                                             const char *name);

MLIRModuleRef MLIRFinalizeModuleBuilder(MLIRModuleBuilderRef builder);

MLIRFunctionOpRef MLIRCreateFunction(MLIRModuleBuilderRef builder,
                                     const char *name, const eir::Arg *argv,
                                     int argc, eir::EirType type);

MLIRBlockRef MLIRAppendEntryBlock(MLIRModuleBuilderRef builder,
                                  MLIRFunctionOpRef f);
MLIRBlockRef MLIRAppendBasicBlock(MLIRModuleBuilderRef builder,
                                  MLIRFunctionOpRef f);

void MLIRBlockPositionAtEnd(MLIRModuleBuilderRef builder, MLIRBlockRef block);

void MLIRBuildBr(MLIRModuleBuilderRef builder, MLIRBlockRef dest);

void MLIRBuildUnreachable(MLIRModuleBuilderRef builder);

MLIRLocationRef EIRSpanToMLIRLocation(unsigned start, unsigned end);

MLIRLocationRef MLIRCreateLocation(MLIRContextRef context, const char *filename,
                                   unsigned line, unsigned column);
}

#endif // EIR_BUILDER_H
