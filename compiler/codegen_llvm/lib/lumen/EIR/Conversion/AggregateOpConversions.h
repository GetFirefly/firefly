#ifndef LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
class ConsOpConversion;
class TupleOpConversion;

void populateAggregateOpConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *context,
                                           EirTypeConverter &converter,
                                           TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION
