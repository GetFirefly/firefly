#ifndef LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
class NullOpConversion;
class ConstantAtomOpConversion;
class ConstantBoolOpConversion;
class ConstantBigIntOpConversion;
class ConstantBinaryOpConversion;
class ConstantFloatOpConversion;
class ConstantFloatOpToStdConversion;
class ConstantIntOpConversion;
class ConstantListOpConversion;
class ConstantMapOpConversion;
class ConstantNilOpConversion;
class ConstantNoneOpConversion;
class ConstantTupleOpConversion;

void populateConstantOpConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context,
                                          EirTypeConverter &converter,
                                          TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION
