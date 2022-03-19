#ifndef LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
struct NullOpConversion;
struct ConstantAtomOpConversion;
struct ConstantBoolOpConversion;
struct ConstantBigIntOpConversion;
struct ConstantBinaryOpConversion;
struct ConstantFloatOpConversion;
struct ConstantFloatOpToStdConversion;
struct ConstantIntOpConversion;
struct ConstantListOpConversion;
struct ConstantMapOpConversion;
struct ConstantNilOpConversion;
struct ConstantNoneOpConversion;
struct ConstantTupleOpConversion;

void populateConstantOpConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context,
                                          EirTypeConverter &converter,
                                          TargetPlatform &platform);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION
