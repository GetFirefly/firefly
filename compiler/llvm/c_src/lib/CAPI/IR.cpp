#include "firefly/llvm/IR.h"

#include "llvm-c/Core.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Alignment.h"

using namespace llvm;

LLVMValueRef LLVMFireflyBuildMemCpy(LLVMBuilderRef b, LLVMValueRef dst,
                                    unsigned dstAlign, LLVMValueRef src,
                                    unsigned srcAlign, LLVMValueRef size,
                                    bool isVolatile) {
  return wrap(unwrap(b)->CreateMemCpy(unwrap(dst), MaybeAlign(dstAlign),
                                      unwrap(src), MaybeAlign(srcAlign),
                                      unwrap(size), isVolatile));
}

LLVMValueRef LLVMFireflyBuildMemMove(LLVMBuilderRef b, LLVMValueRef dst,
                                     unsigned dstAlign, LLVMValueRef src,
                                     unsigned srcAlign, LLVMValueRef size,
                                     bool isVolatile) {
  return wrap(unwrap(b)->CreateMemMove(unwrap(dst), MaybeAlign(dstAlign),
                                       unwrap(src), MaybeAlign(srcAlign),
                                       unwrap(size), isVolatile));
}

LLVMValueRef LLVMFireflyBuildMemSet(LLVMBuilderRef b, LLVMValueRef dst,
                                    unsigned dstAlign, LLVMValueRef val,
                                    LLVMValueRef size, bool isVolatile) {
  return wrap(unwrap(b)->CreateMemSet(unwrap(dst), unwrap(val), unwrap(size),
                                      MaybeAlign(dstAlign), isVolatile));
}

LLVMOperandBundleDef LLVMFireflyBuildOperandBundleDef(const char *name,
                                                      unsigned nameLen,
                                                      LLVMValueRef *inputs,
                                                      unsigned numInputs) {
  return wrap(new OperandBundleDef(std::string(name, nameLen),
                                   makeArrayRef(unwrap(inputs), numInputs)));
}

void LLVMFireflyFreeOperandBundleDef(LLVMOperandBundleDef bundle) {
  delete unwrap(bundle);
}

LLVMValueRef LLVMFireflyBuildCall(LLVMBuilderRef b, LLVMValueRef calleeValue,
                                  LLVMTypeRef calleeType, LLVMValueRef *args,
                                  unsigned numArgs,
                                  LLVMOperandBundleDef bundleDef) {
  OperandBundleDef *bundle = unwrap(bundleDef);
  unsigned len = bundle ? 1 : 0;
  ArrayRef<OperandBundleDef> bundles(bundle, len);
  Value *callee = unwrap(calleeValue);
  FunctionType *calleeTy = cast<FunctionType>(unwrap(calleeType));
  return wrap(unwrap(b)->CreateCall(
      calleeTy, callee, makeArrayRef(unwrap(args), numArgs), bundles));
}

extern "C" LLVMValueRef
LLVMFireflyBuildInvoke(LLVMBuilderRef b, LLVMValueRef calleeValue,
                       LLVMTypeRef calleeType, LLVMValueRef *args,
                       unsigned numArgs, LLVMBasicBlockRef thenBlk,
                       LLVMBasicBlockRef catchBlk,
                       LLVMOperandBundleDef bundleDef, const char *name) {
  OperandBundleDef *bundle = unwrap(bundleDef);
  unsigned len = bundle ? 1 : 0;
  ArrayRef<OperandBundleDef> bundles(bundle, len);
  Value *callee = unwrap(calleeValue);
  FunctionType *calleeTy = cast<FunctionType>(unwrap(calleeType));
  return wrap(unwrap(b)->CreateInvoke(
      calleeTy, callee, unwrap(thenBlk), unwrap(catchBlk),
      makeArrayRef(unwrap(args), numArgs), bundles, name));
}
