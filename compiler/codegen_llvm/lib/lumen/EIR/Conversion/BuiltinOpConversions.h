#ifndef LUMEN_EIR_CONVERSION_BUILTIN_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_BUILTIN_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
struct IncrementReductionsOpConversion;
struct IsTypeOpConversion;
struct IsTupleOpConversion;
struct IsFunctionOpConversion;
struct PrintOpConversion;
struct TraceCaptureOpConversion;
struct TraceConstructOpConversion;
struct TracePrintOpConversion;

void populateBuiltinOpConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *context,
                                         EirTypeConverter &converter,
                                         TargetPlatform &platform);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_BUILTIN_OP_CONVERSION
