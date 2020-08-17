#ifndef LUMEN_EIR_CONVERSION_BINARY_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_BINARY_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
class BinaryStartOpConversion;
class BinaryFinishOpConversion;
class BinaryPushOpConversion;
class BinaryMatchRawOpConversion;
class BinaryMatchIntegerOpConversion;
class BinaryMatchFloatOpConversion;
class BinaryMatchUtf8OpConversion;
class BinaryMatchUtf16OpConversion;
class BinaryMatchUtf32OpConversion;

void populateBinaryOpConversionPatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *context,
                                        EirTypeConverter &converter,
                                        TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_BINARY_OP_CONVERSION
