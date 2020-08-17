#ifndef LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
// Arithmetic
class AddOpConversion;
class DivOpConversion;
class FDivOpConversion;
class MulOpConversion;
class RemOpConversion;
class SubOpConversion;
class NegOpConversion;
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
                                      EirTypeConverter &converter,
                                      TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_MATH_OP_CONVERSION
