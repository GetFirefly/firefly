#pragma once

#include "CIR/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace cf {
class ControlFlowDialect;
}
namespace func {
class FuncDialect;
}
namespace LLVM {
class LLVMDialect;
}
namespace scf {
class SCFDialect;
}

namespace cir {
#define GEN_PASS_CLASSES
#include "CIR/Passes.h.inc"
} // namespace cir
} // namespace mlir
