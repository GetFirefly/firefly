#ifndef LUMEN_EIR_CONVERSION_FUNCLIKE_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_FUNCLIKE_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
struct FuncOpConversion;
struct ClosureOpConversion;
struct UnpackEnvOpConversion;

void populateFuncLikeOpConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *context,
                                          EirTypeConverter &converter,
                                          TargetPlatform &platform);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_FUNCLIKE_OP_CONVERSION
