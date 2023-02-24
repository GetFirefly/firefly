#pragma once

#include "firefly/llvm/CAPI.h"
#include "firefly/llvm/Target.h"
#include "llvm-c/Core.h"

namespace OptStage {
enum Stage {
  PreLinkNoLTO,
  PreLinkThinLTO,
  PreLinkFatLTO,
  ThinLTO,
  FatLTO,
};
}

namespace LLVMFireflyPassBuilderOptLevel {
enum Level {
  O0 = 0,
  O1,
  O2,
  O3,
  Os,
  Oz,
};
}

extern "C" {

typedef void (*LLVMFireflySelfProfileBeforePassCallback)(
    /*profiler*/ void *, /*pass name*/ const char *, /*ir name*/ const char *);
typedef void (*LLVMFireflySelfProfileAfterPassCallback)(/*profiler*/ void *);

struct SanitizerOptions {
  bool memory;
  bool thread;
  bool address;
  bool recover;
  int memoryTrackOrigins;
};

struct OptimizerConfig {
  const char *passPipeline;
  LLVMFireflyPassBuilderOptLevel::Level optLevel;
  OptStage::Stage stage;
  SanitizerOptions sanitizer;
  bool debug;
  bool verify;
  bool useThinLTOBuffers;
  bool disableSimplifyLibCalls;
  bool emitSummaryIndex;
  bool emitModuleHash;
  bool preserveUseListOrder;
  void *profiler;
  LLVMFireflySelfProfileBeforePassCallback beforePass;
  LLVMFireflySelfProfileAfterPassCallback afterPass;
};
}

extern "C" {
MLIR_CAPI_EXPORTED void LLVMFireflyInitializePasses();

MLIR_CAPI_EXPORTED void LLVMTimeTraceProfilerInitialize();

MLIR_CAPI_EXPORTED void LLVMTimeTraceProfilerFinish(const char *fileName);

MLIR_CAPI_EXPORTED void LLVMFireflyPrintPasses();

MLIR_CAPI_EXPORTED bool LLVMFireflyOptimize(LLVMModuleRef m,
                                            LLVMTargetMachineRef tm,
                                            OptimizerConfig *conf,
                                            char **errorMessage);
}
