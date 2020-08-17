#ifndef LUMEN_EIR_CONVERSION_MAP_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_MAP_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
class ConstructMapOpConversion;
class MapInsertOpConversion;
class MapUpdateOpConversion;
class MapIsKeyOpConversion;
class MapGetKeyOpConversion;

void populateMapOpConversionPatterns(OwningRewritePatternList &patterns,
                                     MLIRContext *context,
                                     EirTypeConverter &converter,
                                     TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_MAP_OP_CONVERSION
