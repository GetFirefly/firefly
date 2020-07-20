#ifndef LUMEN_EIR_CONVERSION_MAP_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_MAP_OP_CONVERSION

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"

namespace lumen {
namespace eir {
class ConstructMapOpConversion;
class MapInsertOpConversion;
class MapUpdateOpConversion;

void populateMapOpConversionPatterns(OwningRewritePatternList &patterns,
                                     MLIRContext *context,
                                     LLVMTypeConverter &converter,
                                     TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_MAP_OP_CONVERSION
