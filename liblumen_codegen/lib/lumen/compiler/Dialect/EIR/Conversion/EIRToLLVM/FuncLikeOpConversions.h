#ifndef LUMEN_EIR_CONVERSION_FUNCLIKE_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_FUNCLIKE_OP_CONVERSION

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"

namespace lumen {
namespace eir {
class FuncOpConversion;
class ClosureOpConversion;

void populateFuncLikeOpConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context,
                                          LLVMTypeConverter &converter,
                                          TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_FUNCLIKE_OP_CONVERSION
