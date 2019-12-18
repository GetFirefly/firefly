#ifndef EIR_PASSES_H
#define EIR_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace eir {
/// Create a pass for lowering a subset of EIR operations to the Standard
/// dialect
std::unique_ptr<mlir::Pass> createLowerToStandardPass();

/// Create a pass for lowering the remaining EIR and Standard dialect
/// operations, to the LLVM dialect, in preparation for code generation via
/// LLVM.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace eir

#endif
