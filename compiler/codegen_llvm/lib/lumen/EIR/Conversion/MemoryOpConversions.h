#ifndef LUMEN_EIR_CONVERSION_MEMORY_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_MEMORY_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
struct CastOpConversion;
struct GetElementPtrOpConversion;
struct LoadOpConversion;
struct MallocOpConversion;

void populateMemoryOpConversionPatterns(OwningRewritePatternList &patterns,
                                        MLIRContext *context,
                                        EirTypeConverter &converter,
                                        TargetPlatform &platform);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_MEMORY_OP_CONVERSION
