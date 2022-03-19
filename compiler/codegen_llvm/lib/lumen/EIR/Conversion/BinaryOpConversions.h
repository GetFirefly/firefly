#ifndef LUMEN_EIR_CONVERSION_BINARY_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_BINARY_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
struct BinaryStartOpConversion;
struct BinaryFinishOpConversion;
struct BinaryPushOpConversion;
struct BinaryMatchRawOpConversion;
struct BinaryMatchIntegerOpConversion;
struct BinaryMatchFloatOpConversion;
struct BinaryMatchUtf8OpConversion;
struct BinaryMatchUtf16OpConversion;
struct BinaryMatchUtf32OpConversion;

void populateBinaryOpConversionPatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *context,
                                        EirTypeConverter &converter,
                                        TargetPlatform &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_BINARY_OP_CONVERSION
