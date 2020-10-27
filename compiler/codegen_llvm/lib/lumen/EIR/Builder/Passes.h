#ifndef LUMEN_BUILDER_PASSES_H
#define LUMEN_BUILDER_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace lumen {
namespace eir {
std::unique_ptr<mlir::Pass> createInsertTraceConstructorsPass();
}
}  // namespace lumen

#endif
