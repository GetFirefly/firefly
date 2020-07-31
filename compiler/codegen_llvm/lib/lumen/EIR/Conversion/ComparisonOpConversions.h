#ifndef LUMEN_EIR_CONVERSION_COMPARISION_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_COMPARISION_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
class CmpEqOpConversion;
class CmpNeqOpConversion;
class CmpLtOpConversion;
class CmpLteOpConversion;
class CmpGtOpConversion;
class CmpGteOpConversion;

void populateComparisonOpConversionPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *context,
                                            LLVMTypeConverter &converter,
                                            TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_COMPARISION_OP_CONVERSION
