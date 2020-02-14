#ifndef LUMEN_COMPILER_DIALECT_EIR_CONVERSION_EIRTOSTANDARD_H_
#define LUMEN_COMPILER_DIALECT_EIR_CONVERSION_EIRTOSTANDARD_H_

#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace lumen {
namespace eir {

// Populates conversion patterns from the EIR dialect to the Standard dialect.
void populateEIRToStandardPatterns(MLIRContext *context,
                                   SymbolTable &importSymbols,
                                   OwningRewritePatternList &patterns,
                                   TypeConverter &typeConverter);

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>> createConvertEIRToStandardPass();

}  // namespace eir
}  // namespace lumen

#endif
