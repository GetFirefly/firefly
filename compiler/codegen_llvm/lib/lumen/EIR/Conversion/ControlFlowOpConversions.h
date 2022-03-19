#ifndef LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION
#define LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION

#include "lumen/EIR/Conversion/ConversionSupport.h"

namespace lumen {
namespace eir {
struct BranchOpConversion;
struct CondBranchOpConversion;
struct CallOpConversion;
struct InvokeOpConversion;
struct LandingPadOpConversion;
struct ReturnOpConversion;
struct ThrowOpConversion;
struct UnreachableOpConversion;
struct YieldOpConversion;
struct YieldCheckOpConversion;
struct ReceiveStartOpConversion;
struct ReceiveWaitOpConversion;
struct ReceiveMessageOpConversion;
struct ReceiveDoneOpConversion;

void populateControlFlowOpConversionPatterns(OwningRewritePatternList &patterns,
                                             MLIRContext *context,
                                             EirTypeConverter &converter,
                                             TargetPlatform &platform);
}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_EIR_CONVERSION_CONTROLFLOW_OP_CONVERSION
