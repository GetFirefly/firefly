#pragma once

#include "firefly/llvm/CAPI.h"

#include "llvm-c/Core.h"

namespace llvm {
class Value;
template <typename T> class OperandBundleDefT;
using OperandBundleDef = OperandBundleDefT<Value *>;
} // namespace llvm

extern "C" {
DEFINE_C_API_STRUCT(LLVMOperandBundleDef, void);
}

DEFINE_C_API_PTR_METHODS(LLVMOperandBundleDef, llvm::OperandBundleDef);

extern "C" {
MLIR_CAPI_EXPORTED LLVMValueRef LLVMFireflyBuildMemCpy(
    LLVMBuilderRef b, LLVMValueRef dst, unsigned dstAlign, LLVMValueRef src,
    unsigned srcAlign, LLVMValueRef size, bool isVolatile);

MLIR_CAPI_EXPORTED LLVMValueRef LLVMFireflyBuildMemMove(
    LLVMBuilderRef b, LLVMValueRef dst, unsigned dstAlign, LLVMValueRef src,
    unsigned srcAlign, LLVMValueRef size, bool isVolatile);

MLIR_CAPI_EXPORTED LLVMValueRef
LLVMFireflyBuildMemSet(LLVMBuilderRef b, LLVMValueRef dst, unsigned dstAlign,
                       LLVMValueRef val, LLVMValueRef size, bool isVolatile);

MLIR_CAPI_EXPORTED LLVMOperandBundleDef
LLVMFireflyBuildOperandBundleDef(const char *name, unsigned nameLen,
                                 LLVMValueRef *inputs, unsigned numInputs);

MLIR_CAPI_EXPORTED void
LLVMFireflyFreeOperandBundleDef(LLVMOperandBundleDef bundle);

MLIR_CAPI_EXPORTED LLVMValueRef LLVMFireflyBuildCall(
    LLVMBuilderRef b, LLVMValueRef calleeValue, LLVMTypeRef calleeType,
    LLVMValueRef *args, unsigned numArgs, LLVMOperandBundleDef bundle);

MLIR_CAPI_EXPORTED LLVMValueRef LLVMFireflyBuildInvoke(
    LLVMBuilderRef b, LLVMValueRef calleeValue, LLVMTypeRef calleeType,
    LLVMValueRef *args, unsigned numArgs, LLVMBasicBlockRef thenBlk,
    LLVMBasicBlockRef catchBlk, LLVMOperandBundleDef bundle, const char *name);
}
