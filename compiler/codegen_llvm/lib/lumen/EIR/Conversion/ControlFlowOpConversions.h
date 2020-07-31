#ifndef LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
// class ApplyOpConversion;
class BranchOpConversion;
class CondBranchOpConversion;
// class CallIndirectOpConversion;
class CallOpConversion;
class CallClosureOpConversion;
class InvokeOpConversion;
class InvokeClosureOpConversion;
class LandingPadOp;
class ReturnOpConversion;
class ThrowOpConversion;
class UnreachableOpConversion;
class YieldOpConversion;
class YieldCheckOpConversion;
class ReceiveStartOpConversion;
class ReceiveWaitOpConversion;
class ReceiveDoneOpConversion;

void populateControlFlowOpConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context,
                                             LLVMTypeConverter &converter,
                                             TargetInfo &targetInfo);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION
