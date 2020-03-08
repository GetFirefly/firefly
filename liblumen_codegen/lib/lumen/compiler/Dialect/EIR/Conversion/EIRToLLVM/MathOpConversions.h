#ifndef LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"

namespace lumen {
namespace eir {
class AddOpConversion;
class SubOpConversion;
class MulOpConversion;
class DivOpConversion;
class RemOpConversion;
class FDivOpConversion;
class BandOpConversion;
class BorOpConversion;
class BxorOpConversion;
class BslOpConversion;
class BsrOpConversion;

void populateMathOpConversionPatterns(OwningRewritePatternList &patterns,
                                      MLIRContext *context,
                                      LLVMTypeConverter &converter,
                                      TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION
