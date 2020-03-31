#ifndef LUMEN_COMPILER_DIALECT_EIR_TRANSFORMS_PASSES_H_
#define LUMEN_COMPILER_DIALECT_EIR_TRANSFORMS_PASSES_H_

#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"

#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace llvm {
class TargetMachine;
}

namespace lumen {
namespace eir {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required EIR
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion to EIR/etc>
//   buildEIRTransformPassPipeline & run
//   <run target serialization/etc>
void buildEIRTransformPassPipeline(mlir::OpPassManager &passManager,
                                   llvm::TargetMachine *targetMachine);

//===----------------------------------------------------------------------===//
// Module Analysis and Assignment
//===----------------------------------------------------------------------===//

// Gathers all module-level global init/deinit functions into single locations
// such that the runtime can init/deinit everything at once.
// std::unique_ptr<mlir::OpPassBase<eir::ModuleOp>>
// createGlobalInitializationPass();

}  // namespace eir
}  // namespace lumen

#endif  // IREE_COMPILER_DIALECT_VM_TRANSFORMS_PASSES_H_
