#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/MapOpConversions.h"

namespace lumen {
namespace eir {

void populateMapOpConversionPatterns(OwningRewritePatternList &_patterns,
                                     MLIRContext *_context,
                                     LLVMTypeConverter &_converter,
                                     TargetInfo &_targetInfo) {
  /*
  patterns.insert<ConstructMapOpConversion,
                  MapInsertOpConversion,
                  MapUpdateOpConversion>(context, converter, targetInfo);
  */
}

}  // namespace eir
}  // namespace lumen
