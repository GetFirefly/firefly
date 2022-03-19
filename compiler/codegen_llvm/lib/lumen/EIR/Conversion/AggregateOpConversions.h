#ifndef LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
struct ConsOpConversion;
struct ListOpConversion;
struct TupleOpConversion;

void populateAggregateOpConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *context,
                                           EirTypeConverter &converter,
                                           TargetPlatform &platform);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION
