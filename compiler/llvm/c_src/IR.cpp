#include "llvm-c/Core.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Alignment.h"

using ::llvm::MaybeAlign;
using ::llvm::StringRef;
using ::llvm::ArrayRef;
using ::llvm::OperandBundleDef;
using ::llvm::Value;
using ::llvm::unwrap;
using ::llvm::wrap;
using ::llvm::LLVMContext;
using ::llvm::DiagnosticInfo;

using DiagnosticHookFn = void (*)(const DiagnosticInfo &, void*);

extern "C" LLVMContextRef LLVMLumenCreateContext(
        bool discardValueNames, 
        DiagnosticHookFn diagnosticHook) {
    auto *ctx = new LLVMContext();
    ctx->setDiscardValueNames(discardValueNames);

    // Handle diagnostics
    if (diagnosticHook != nullptr) {
        ctx->setDiagnosticHandlerCallBack(diagnosticHook);
    }

    return wrap(ctx);
}

extern "C" LLVMValueRef LLVMLumenBuildMemCpy(LLVMBuilderRef b,
                                             LLVMValueRef dst, unsigned dstAlign,
                                             LLVMValueRef src, unsigned srcAlign,
                                             LLVMValueRef size, bool isVolatile) {
  return wrap(unwrap(b)->CreateMemCpy(
      unwrap(dst), MaybeAlign(dstAlign),
      unwrap(src), MaybeAlign(srcAlign),
      unwrap(size), isVolatile));
}

extern "C" LLVMValueRef LLVMLumenBuildMemMove(LLVMBuilderRef b,
                                              LLVMValueRef dst, unsigned dstAlign,
                                              LLVMValueRef src, unsigned srcAlign,
                                              LLVMValueRef size, bool isVolatile) {
  return wrap(unwrap(b)->CreateMemMove(
      unwrap(dst), MaybeAlign(dstAlign),
      unwrap(src), MaybeAlign(srcAlign),
      unwrap(size), isVolatile));
}


extern "C" LLVMValueRef LLVMLumenBuildMemSet(LLVMBuilderRef b,
                                             LLVMValueRef dst, unsigned dstAlign,
                                             LLVMValueRef val,
                                             LLVMValueRef size,
                                             bool isVolatile) {
    return wrap(unwrap(b)->CreateMemSet(
                unwrap(dst), unwrap(val), unwrap(size), MaybeAlign(dstAlign), isVolatile));
}


extern "C" LLVMValueRef LLVMLumenGetOrInsertFunction(LLVMModuleRef m,
                                                     const char *name,
                                                     size_t nameLen,
                                                     LLVMTypeRef functionTy) {
  auto ty = unwrap<llvm::FunctionType>(functionTy);
  return wrap(unwrap(m)->getOrInsertFunction(StringRef(name, nameLen), ty)
                  .getCallee());
}

extern "C" LLVMValueRef
LLVMLumenGetOrInsertGlobal(LLVMModuleRef m, const char *name, size_t nameLen, LLVMTypeRef ty) {
  StringRef nameRef(name, nameLen);
  return wrap(unwrap(m)->getOrInsertGlobal(nameRef, unwrap(ty)));
}

extern "C" void LLVMLumenSetComdat(LLVMModuleRef m, LLVMValueRef v,
                                   const char *name, size_t nameLen) {
  llvm::Triple targetTriple(unwrap(m)->getTargetTriple());
  auto *gv = unwrap<llvm::GlobalObject>(v);
  if (!targetTriple.isOSBinFormatMachO()) {
    StringRef nameRef(name, nameLen);
    gv->setComdat(unwrap(m)->getOrInsertComdat(nameRef));
  }
}

extern "C" void LLVMLumenUnsetComdat(LLVMValueRef v) {
  auto *gv = unwrap<llvm::GlobalObject>(v);
  gv->setComdat(nullptr);
}

extern "C" OperandBundleDef *LLVMLumenBuildOperandBundleDef(const char *name,
                                                            unsigned nameLen,
                                                            LLVMValueRef *inputs,
                                                            unsigned numInputs) {
  return new OperandBundleDef(std::string(name, nameLen), makeArrayRef(unwrap(inputs), numInputs));
}

extern "C" void LLVMLumenFreeOperandBundleDef(OperandBundleDef *Bundle) {
  delete Bundle;
}

extern "C" LLVMValueRef LLVMLumenBuildCall(LLVMBuilderRef b, LLVMValueRef fn,
                                           LLVMValueRef *args, unsigned numArgs,
                                           OperandBundleDef *bundle) {
  unsigned len = bundle ? 1 : 0;
  ArrayRef<OperandBundleDef> bundles(bundle, len);
  return wrap(unwrap(b)->CreateCall(unwrap(fn), makeArrayRef(unwrap(args), numArgs), bundles));
}

extern "C" LLVMValueRef
LLVMLumenBuildInvoke(LLVMBuilderRef b, LLVMValueRef fn, LLVMValueRef *args,
                     unsigned numArgs, LLVMBasicBlockRef thenBlk,
                     LLVMBasicBlockRef catchBlk, OperandBundleDef *bundle,
                     const char *name) {
  unsigned len = bundle ? 1 : 0;
  ArrayRef<OperandBundleDef> bundles(bundle, len);
  return wrap(unwrap(b)->CreateInvoke(unwrap(fn), unwrap(thenBlk), unwrap(catchBlk),
                                      makeArrayRef(unwrap(args), numArgs),
                                      bundles, name));
}





