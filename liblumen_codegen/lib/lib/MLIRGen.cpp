#include "eir/MLIRGen.h"
#include "eir/Passes.h"

// On Windows we have a custom output stream type that
// can wrap the raw file handle we get from Rust
#if defined(_WIN32)
#include "lumen/Support/raw_win32_handle_ostream.h"
#endif

#include "mlir/Analysis/Verifier.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm-c/Core.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdlib>

using namespace eir;

namespace M = mlir;
namespace L = llvm;

MLIRModuleRef MLIRParseFile(MLIRContextRef context, const char *filename) {
  M::MLIRContext *ctx = unwrap(context);
  assert(ctx != nullptr && "invalid MLIRContext pointer");
  StringRef inputFilePath(filename);

  // Parse the input mlir.
  auto mod = M::parseSourceFile(inputFilePath, ctx);
  if (!mod) {
    L::errs() << "Error can't load file " << inputFilePath << "\n";
    return nullptr;
  }

  // We're doing our own memory management, so extract the module from
  // its owning reference
  return wrap(new M::ModuleOp(mod.release()));
}

MLIRModuleRef MLIRLowerModule(MLIRContextRef context, MLIRModuleRef m,
                              TargetDialect dialect,
                              LLVMLumenCodeGenOptLevel opt) {
  M::MLIRContext *ctx = unwrap(context);
  M::ModuleOp *mod = unwrap(m);
  L::CodeGenOpt::Level optLevel = fromRust(opt);

  M::PassManager pm(&*ctx);
  M::applyPassManagerCLOptions(pm);

  bool enableOpt = optLevel >= L::CodeGenOpt::Level::None;
  bool lowerToStandard = dialect >= TargetDialect::TargetStandard;
  bool lowerToLLVM = dialect >= TargetDialect::TargetLLVM;

  if (enableOpt) {
    // Perform high-level inlining
    pm.addPass(M::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    M::OpPassManager &optPM = pm.nest<M::FuncOp>();
    optPM.addPass(M::createCanonicalizerPass());
    optPM.addPass(M::createCSEPass());
  }

  if (lowerToStandard) {
    // Lower the EIR dialect, then clean up
    pm.addPass(eir::createLowerToStandardPass());

    M::OpPassManager &optPM = pm.nest<M::FuncOp>();
    optPM.addPass(M::createCanonicalizerPass());
    optPM.addPass(M::createCSEPass());

    // Add optimizations if enabled
    if (enableOpt) {
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createMemRefDataFlowOptPass());
    }
  }

  if (lowerToLLVM) {
    // Lower from Standard to LLVM dialect
    pm.addPass(eir::createLowerToLLVMPass());
  }

  M::OwningModuleRef ownedMod(*mod);
  if (M::failed(pm.run(*ownedMod))) {
    return nullptr;
  }

  return wrap(new M::ModuleOp(ownedMod.release()));
}

LLVMModuleRef MLIRLowerToLLVMIR(MLIRModuleRef m, LLVMLumenCodeGenOptLevel opt,
                                LLVMLumenCodeGenSizeLevel size,
                                LLVMTargetMachineRef tm) {
  M::ModuleOp *mod = unwrap(m);
  L::TargetMachine *targetMachine = unwrap(tm);
  L::CodeGenOpt::Level optLevel = fromRust(opt);
  unsigned sizeLevel = fromRust(size);

  bool enableOpt = optLevel >= L::CodeGenOpt::Level::None;
  if (sizeLevel > 0) {
    optLevel = L::CodeGenOpt::Level::Default;
  }

  M::OwningModuleRef ownedMod(*mod);
  auto llvmModPtr = M::translateModuleToLLVMIR(*ownedMod);
  if (!llvmModPtr) {
    L::errs() << "Failed to emit LLVM IR!\n";
    return nullptr;
  }

  M::ExecutionEngine::setupTargetTriple(llvmModPtr.get());

  // Optionally run an optimization pipeline
  auto optPipeline =
      M::makeOptimizingTransformer(optLevel, sizeLevel, targetMachine);
  if (auto err = optPipeline(llvmModPtr.get())) {
    L::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }

  return wrap(llvmModPtr.release());
}

MLIRModuleBuilderRef MLIRCreateModuleBuilder(MLIRContextRef context,
                                             const char *name) {
  M::MLIRContext *ctx = unwrap(context);
  auto builder = new MLIRModuleBuilder(*ctx);
  return wrap(builder);
}

MLIRLocationRef MLIRCreateLocation(MLIRContextRef context, const char *filename,
                                   unsigned line, unsigned column) {
  M::MLIRContext *ctx = unwrap(context);
  StringRef FileName(filename);
  M::Location loc = M::FileLineColLoc::get(FileName, line, column, ctx);
  return wrap(&loc);
}

/// Helper conversion for an EIR IR location to an MLIR location.
M::Location MLIRModuleBuilder::loc(Span span) {
  MLIRLocationRef fileLocRef = EIRSpanToMLIRLocation(span.start, span.end);
  M::Location *fileLoc = unwrap(fileLocRef);
  return *fileLoc;
}

#if defined(_WIN32)
bool MLIREmitToFileDescriptor(MLIRModuleRef m, HANDLE handle,
                              char **errorMessage) {
  L::raw_win32_handle_ostream stream(handle, /*shouldClose=*/false,
                                     /*unbuffered=*/false);
#else
bool MLIREmitToFileDescriptor(MLIRModuleRef m, int fd, char **errorMessage) {
  L::raw_fd_ostream stream(fd, /*shouldClose=*/false, /*unbuffered=*/false);
#endif
  M::ModuleOp *mod = unwrap(m);
  mod->print(stream);
  if (stream.has_error()) {
    std::error_code error = stream.error();
    *errorMessage = strdup(error.message().c_str());
    return true;
  }
  stream.flush();
  return false;
}

LLVMMemoryBufferRef MLIREmitToMemoryBuffer(MLIRModuleRef m) {
  M::ModuleOp *mod = unwrap(m);
  L::SmallString<0> codeString;
  L::raw_svector_ostream oStream(codeString);
  mod->print(oStream);
  StringRef data = oStream.str();
  return LLVMCreateMemoryBufferWithMemoryRangeCopy(data.data(), data.size(),
                                                   "");
}
