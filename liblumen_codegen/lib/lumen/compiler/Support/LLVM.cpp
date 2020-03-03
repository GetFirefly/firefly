#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "mlir/ExecutionEngine/OptUtils.h"

extern "C" void LLVMLumenInitializePasses() { mlir::initializeLLVMPasses(); }

extern "C" void LLVMLumenPrintPasses() {
  mlir::initializeLLVMPasses();

  struct LumenPassListener : llvm::PassRegistrationListener {
    void passEnumerate(const llvm::PassInfo *Info) {
      auto passArg = Info->getPassArgument();
      auto passName = Info->getPassName();
      if (!passArg.empty()) {
        // These unsigned->signed casts could theoretically overflow, but
        // realistically never will (and even if, the result is implementation
        // defined, rather than plain UB)
        printf("%32.*s | %.*s\n", (int)passArg.size(), passArg.data(),
               (int)passName.size(), passName.data());
      }
    }
  } listener;

  auto *registry = llvm::PassRegistry::getPassRegistry();
  registry->enumerateWith(&listener);
}
