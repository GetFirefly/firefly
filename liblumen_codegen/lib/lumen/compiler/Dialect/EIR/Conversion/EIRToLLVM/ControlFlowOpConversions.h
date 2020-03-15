#ifndef LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION

#include "lumen/compiler/Dialect/EIR/Conversion/EIRToLLVM/ConversionSupport.h"

namespace lumen {
namespace eir {
// class ApplyOpConversion;
class BranchOpConversion;
class CondBranchOpConversion;
// class CallIndirectOpConversion;
class CallOpConversion;
class ReturnOpConversion;
class ThrowOpConversion;
class UnreachableOpConversion;
class YieldOpConversion;
class YieldCheckOpConversion;

void populateControlFlowOpConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context,
                                             LLVMTypeConverter &converter,
                                             TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION
