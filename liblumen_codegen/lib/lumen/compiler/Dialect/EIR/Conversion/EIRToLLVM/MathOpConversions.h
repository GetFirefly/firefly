#ifndef LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"

namespace lumen {
namespace eir {
// Arithmetic
class AddOpConversion;
class DivOpConversion;
class FDivOpConversion;
class MulOpConversion;
class RemOpConversion;
class SubOpConversion;
// Bitwise
class BandOpConversion;
class BorOpConversion;
class BslOpConversion;
class BsrOpConversion;
class BxorOpConversion;
// Logical
// class LogicalAndOpConversion;
// class LogicalOrOpConversion;

void populateMathOpConversionPatterns(OwningRewritePatternList &patterns,
                                      MLIRContext *context,
                                      LLVMTypeConverter &converter,
                                      TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION
