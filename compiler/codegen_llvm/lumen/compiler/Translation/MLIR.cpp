#include "lumen/mlir/MLIR.h"

#include "lumen/compiler/Dialect/EIR/IR/EIRDialect.h"
#include "lumen/compiler/Dialect/EIR/Transforms/Passes.h"
#include "lumen/compiler/Target/TargetInfo.h"
#include "lumen/llvm/Target.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"

using ::lumen::CodeGenOptLevel;
using ::lumen::OptLevel;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OpPassManager;
using ::mlir::OwningModuleRef;
using ::mlir::PassManager;
using ::llvm::TargetMachine;
using ::llvm::LLVMContext;
using ::llvm::unwrap;

extern "C" void MLIRRegisterDialects(MLIRContextRef mlirCtx, LLVMContextRef llvmCtx) {
  MLIRContext *mlirContext = unwrap(mlirCtx);
  LLVMContext *llvmContext = unwrap(llvmCtx);

  // Register the LLVM and EIR dialects with MLIR, providing them
  // with the current thread's LLVMContext.
  //
  // NOTE: The dialect constructors internally call registerDialect,
  // which moves ownership of the dialect objects to the MLIRContext, 
  // so we don't have to manage them ourselves.
  auto *llvmDialect = new mlir::LLVM::LLVMDialect(mlirContext, llvmContext);
  auto *eirDialect = new lumen::eir::EirDialect(mlirContext);
}

extern "C" MLIRPassManagerRef MLIRCreatePassManager(MLIRContextRef context,
                                                    LLVMTargetMachineRef tm,
                                                    OptLevel opt,
                                                    bool enableTiming,
                                                    bool enableStatistics) {
  MLIRContext *ctx = unwrap(context);
  TargetMachine *targetMachine = unwrap(tm);
  CodeGenOptLevel optLevel = toLLVM(opt);

  auto pm = new PassManager(ctx);
  mlir::applyPassManagerCLOptions(*pm);
  if (enableTiming) pm->enableTiming();
  if (enableStatistics) pm->enableStatistics();

  bool enableOpt = optLevel >= CodeGenOptLevel::None;

  if (enableOpt) {
    // Perform high-level inlining
    // pm.addPass(mlir::createInlinerPass());

    OpPassManager &optPM = pm->nest<::lumen::eir::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  lumen::eir::buildEIRTransformPassPipeline(*pm, targetMachine);

  // Add optimizations if enabled
  if (enableOpt) {
    OpPassManager &optPM = pm->nest<::mlir::LLVM::LLVMFuncOp>();
    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  return wrap(pm);
}
