#ifndef LUMEN_EIR_CONVERSION_ANNOTATEGCLEAF_H_
#define LUMEN_EIR_CONVERSION_ANNOTATEGCLEAF_H_

#include "mlir/Pass/Pass.h"

#include <memory>

namespace lumen {
namespace eir {
std::unique_ptr<mlir::Pass> createAnnotateGCLeafPass();
}  // namespace eir
}  // namespace lumen

#endif
