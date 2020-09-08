#ifndef LUMEN_EIR_CONVERSION_BUILTIN_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_BUILTIN_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
class IncrementReductionsOpConversion;
class IsTypeOpConversion;
class IsTupleOpConversion;
class IsFunctionOpConversion;
class PrintOpConversion;
class TraceCaptureOpConversion;
class TraceConstructOpConversion;
class TracePrintOpConversion;

void populateBuiltinOpConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context,
                                         EirTypeConverter &converter,
                                         TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_BUILTIN_OP_CONVERSION
