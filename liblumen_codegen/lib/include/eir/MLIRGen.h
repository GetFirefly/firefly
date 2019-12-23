#ifndef EIR_MLIR_GEN_H
#define EIR_MLIR_GEN_H

#include "eir/Context.h"
#include "lumen/LLVM.h"
#include "lumen/Target.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

#include "llvm-c/Core.h"
#include "llvm/Support/CBindingWrapping.h"

#include <memory>

typedef struct MLIROpaqueModuleBuilder *MLIRModuleBuilderRef;
typedef struct MLIROpaqueModuleOp *MLIRModuleRef;
typedef struct MLIROpaqueLocation *MLIRLocationRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::ModuleOp, MLIRModuleRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::Location, MLIRLocationRef);

using llvm::StringRef;

namespace eir {

enum class TargetDialect {
  Unknown,
  TargetNone,
  TargetEIR,
  TargetStandard,
  TargetLLVM,
};

/// A source location in EIR
struct Span {
  // The starting byte index of a span
  int start;
  // The end byte index of a span
  int end;
};

class EirModule;

class MLIRModuleBuilder {
public:
  MLIRModuleBuilder(M::MLIRContext &context) : builder(&context) {
    // Create an empty module into which we can codegen functions
    theModule = M::ModuleOp::create(builder.getUnknownLoc());
  }

private:
  /// The module we're building, essentially equivalent to the EIR module
  M::ModuleOp theModule;

  /// The builder is used for generating IR inside of functions in the module,
  /// it is very similar to the LLVM builder
  M::OpBuilder builder;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<M::FuncOp> functionMap;

  M::Location loc(Span span);
};

} // namespace eir

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(eir::MLIRModuleBuilder,
                                   MLIRModuleBuilderRef);

extern "C" MLIRModuleRef MLIRParseFile(MLIRContextRef context,
                                       const char *filename);

extern "C" MLIRModuleRef MLIRParseBuffer(MLIRContextRef context,
                                         LLVMMemoryBufferRef buffer);

extern "C" MLIRModuleRef MLIRLowerModule(MLIRContextRef context,
                                         MLIRModuleRef mod,
                                         eir::TargetDialect dialect,
                                         LLVMLumenCodeGenOptLevel opt);

extern "C" LLVMModuleRef MLIRLowerToLLVMIR(MLIRModuleRef m,
                                           LLVMLumenCodeGenOptLevel opt,
                                           LLVMLumenCodeGenSizeLevel size,
                                           LLVMTargetMachineRef tm);

extern "C" MLIRModuleBuilderRef MLIRCreateModuleBuilder(MLIRContextRef context,
                                                        const char *name);

extern "C" MLIRLocationRef EIRSpanToMLIRLocation(unsigned start, unsigned end);

extern "C" MLIRLocationRef MLIRCreateLocation(MLIRContextRef context,
                                              const char *filename,
                                              unsigned line, unsigned column);

extern "C" bool MLIREmitToFileDescriptor(MLIRModuleRef m,
#if defined(_WIN32)
                                         HANDLE handle,
#else
                                         int fd,
#endif
                                         char **errorMessage);

#endif // EIR_MLIR_GEN_H
