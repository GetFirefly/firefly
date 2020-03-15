#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/BinaryOpConversions.h"

namespace lumen {
namespace eir {

void populateBinaryOpConversionPatterns(OwningRewritePatternList &_patterns,
                                        MLIRContext *_context,
                                        LLVMTypeConverter &_converter,
                                        TargetInfo &_targetInfo) {
  // patterns.insert<BinaryPushOpConversion>(context, converter, targetInfo);
}

}  // namespace eir
}  // namespace lumen
