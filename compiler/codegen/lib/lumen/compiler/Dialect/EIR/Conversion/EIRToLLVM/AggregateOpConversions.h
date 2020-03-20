#ifndef LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"

namespace lumen {
namespace eir {
class ConsOpConversion;
class TupleOpConversion;

void populateAggregateOpConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *context,
                                           LLVMTypeConverter &converter,
                                           TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_AGGREGATE_OP_CONVERSION
