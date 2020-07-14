#ifndef LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"

namespace lumen {
namespace eir {
class NullOpConversion;
class ConstantAtomOpConversion;
class ConstantBigIntOpConversion;
class ConstantBinaryOpConversion;
class ConstantFloatOpConversion;
class ConstantFloatOpToStdConversion;
class ConstantIntOpConversion;
class ConstantListOpConversion;
// class ConstantMapOpConversion;
class ConstantNilOpConversion;
class ConstantNoneOpConversion;
class ConstantTupleOpConversion;

void populateConstantOpConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context,
                                          LLVMTypeConverter &converter,
                                          TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_CONSTANT_OP_CONVERSION
