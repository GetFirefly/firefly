#ifndef LUMEN_BUILDER_PASSES_H
#define LUMEN_BUILDER_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace lumen {
namespace eir {
std::unique_ptr<mlir::Pass> createInsertTraceConstructorsPass();
}
}  // namespace lumen

#endif
