#include "firefly/llvm/Passes.h"
#include "firefly/llvm/Target.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"
//#include "llvm/Transforms/Scalar/PlaceSafepoints.h"
//#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/Timer.h"
#include "llvm/Transforms/Scalar/RewriteStatepointsForGC.h"
#include "llvm/Transforms/Utils/CanonicalizeAliases.h"
#include "llvm/Transforms/Utils/NameAnonGlobals.h"

#include <set>
#include <stdio.h>
#include <vector>
#include <optional>

using namespace llvm;

void LLVMFireflyInitializePasses() {
  auto &registry = *llvm::PassRegistry::getPassRegistry();
  llvm::initializeCore(registry);
  llvm::initializeCodeGen(registry);
  llvm::initializeScalarOpts(registry);
  llvm::initializeVectorization(registry);
  llvm::initializeIPO(registry);
  llvm::initializeAnalysis(registry);
  llvm::initializeTransformUtils(registry);
  llvm::initializeInstCombine(registry);
  llvm::initializeTarget(registry);
}

void LLVMTimeTraceProfilerInitialize() {
  llvm::timeTraceProfilerInitialize(/*timeTraceGranularity*/ 0,
                                    /*procName*/ "firefly");
}

void LLVMTimeTraceProfilerFinish(const char *fileName) {
  StringRef fn(fileName);
  std::error_code ec;
  llvm::raw_fd_ostream out(fn, ec, llvm::sys::fs::CD_CreateAlways);

  llvm::timeTraceProfilerWrite(out);
  llvm::timeTraceProfilerCleanup();
}

void LLVMFireflyPrintPasses() {
  LLVMFireflyInitializePasses();

  struct FireflyPassListener : llvm::PassRegistrationListener {
    void passEnumerate(const llvm::PassInfo *Info) override {
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

std::string getWrappedIRName(const llvm::Any &wrappedIR) {
  if (any_isa<const Module *>(wrappedIR))
    return any_cast<const Module *>(wrappedIR)->getName().str();
  if (any_isa<const llvm::Function *>(wrappedIR))
    return any_cast<const llvm::Function *>(wrappedIR)->getName().str();
  if (any_isa<const llvm::Loop *>(wrappedIR))
    return any_cast<const llvm::Loop *>(wrappedIR)->getName().str();
  if (any_isa<const llvm::LazyCallGraph::SCC *>(wrappedIR))
    return any_cast<const llvm::LazyCallGraph::SCC *>(wrappedIR)->getName();
  return "<UNKNOWN>";
}

void LLVMSelfProfileInitializeCallbacks(
    PassInstrumentationCallbacks &pic, void *selfProfiler,
    LLVMFireflySelfProfileBeforePassCallback beforePassCallback,
    LLVMFireflySelfProfileAfterPassCallback afterPassCallback) {
  pic.registerBeforeNonSkippedPassCallback(
      [selfProfiler, beforePassCallback](StringRef pass, llvm::Any ir) {
        std::string passName = pass.str();
        std::string irName = getWrappedIRName(ir);
        beforePassCallback(selfProfiler, passName.c_str(), irName.c_str());
      });

  pic.registerAfterPassCallback(
      [selfProfiler, afterPassCallback](StringRef pass, llvm::Any ir,
                                        const llvm::PreservedAnalyses &) {
        afterPassCallback(selfProfiler);
      });

  pic.registerAfterPassInvalidatedCallback(
      [selfProfiler, afterPassCallback](StringRef pass,
                                        const llvm::PreservedAnalyses &) {
        afterPassCallback(selfProfiler);
      });

  pic.registerBeforeAnalysisCallback(
      [selfProfiler, beforePassCallback](StringRef pass, llvm::Any ir) {
        std::string passName = pass.str();
        std::string irName = getWrappedIRName(ir);
        beforePassCallback(selfProfiler, passName.c_str(), irName.c_str());
      });

  pic.registerAfterAnalysisCallback(
      [selfProfiler, afterPassCallback](StringRef pass, llvm::Any ir) {
        afterPassCallback(selfProfiler);
      });
}

static OptimizationLevel fromRust(LLVMFireflyPassBuilderOptLevel::Level level) {
  switch (level) {
  case LLVMFireflyPassBuilderOptLevel::O0:
    return OptimizationLevel::O0;
  case LLVMFireflyPassBuilderOptLevel::O1:
    return OptimizationLevel::O1;
  case LLVMFireflyPassBuilderOptLevel::O2:
    return OptimizationLevel::O2;
  case LLVMFireflyPassBuilderOptLevel::O3:
    return OptimizationLevel::O3;
  case LLVMFireflyPassBuilderOptLevel::Os:
    return OptimizationLevel::Os;
  case LLVMFireflyPassBuilderOptLevel::Oz:
    return OptimizationLevel::Oz;
  }
}

inline static bool sanitizerEnabled(SanitizerOptions &opts) {
  return opts.memory || opts.thread || opts.address;
}

bool LLVMFireflyOptimize(LLVMModuleRef m, LLVMTargetMachineRef tm,
                         OptimizerConfig *conf, char **errorMessage) {
  TargetMachine *targetMachine = unwrap(tm);
  Module *mod = unwrap(m);
  OptimizerConfig config = *conf;

  auto optLevel = fromRust(config.optLevel);

  llvm::PipelineTuningOptions tuningOpts;
  tuningOpts.LoopUnrolling = false;
  tuningOpts.LoopInterleaving = false;
  tuningOpts.LoopVectorization = false;
  tuningOpts.SLPVectorization = false;

  bool debug = config.debug;
  bool verify = config.verify;

  // Enable standard instrumentation callbacks
  llvm::PassInstrumentationCallbacks pic;
  llvm::StandardInstrumentations si(debug);
  si.registerCallbacks(pic);

  auto *profiler = config.profiler;
  if (profiler) {
    LLVMSelfProfileInitializeCallbacks(pic, profiler, config.beforePass,
                                       config.afterPass);
  }

  // Populate the analysis managers with their respective passes
  PassBuilder pb(targetMachine, tuningOpts, None, &pic);

  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cam;
  ModuleAnalysisManager mam;

  fam.registerPass([&] { return pb.buildDefaultAAPipeline(); });

  auto triple = targetMachine->getTargetTriple();
  std::unique_ptr<llvm::TargetLibraryInfoImpl> tlii(
      new llvm::TargetLibraryInfoImpl(triple));
  if (config.disableSimplifyLibCalls)
    tlii->disableAllFunctions();
  fam.registerPass([&] { return llvm::TargetLibraryAnalysis(*tlii); });

  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cam, mam);

  // We manually collect pipeline callbacks so we can apply them at O0, where
  // the PassBuilder does not create a pipeline.
  std::vector<std::function<void(ModulePassManager &, OptimizationLevel)>>
      pipelineStartEPCallbacks;
  std::vector<std::function<void(ModulePassManager &, OptimizationLevel)>>
      optimizerLastEPCallbacks;

  if (verify) {
    pipelineStartEPCallbacks.push_back(
        [](ModulePassManager &pm, OptimizationLevel _level) {
          pm.addPass(llvm::VerifierPass());
        });
  }

  auto sanitizer = config.sanitizer;
  if (sanitizerEnabled(sanitizer)) {
    if (sanitizer.memory) {
      llvm::MemorySanitizerOptions mso(
          sanitizer.memoryTrackOrigins, sanitizer.recover,
          /*compilerKernel=*/false, /*eagerChecks=*/true);
      optimizerLastEPCallbacks.push_back(
          [mso](ModulePassManager &pm, OptimizationLevel level) {
            pm.addPass(llvm::ModuleMemorySanitizerPass(mso));
          });
    }
    if (sanitizer.thread) {
      optimizerLastEPCallbacks.push_back(
          [](ModulePassManager &pm, OptimizationLevel level) {
            pm.addPass(llvm::ModuleThreadSanitizerPass());
            pm.addPass(llvm::createModuleToFunctionPassAdaptor(
                llvm::ThreadSanitizerPass()));
          });
    }
    if (sanitizer.address) {
      optimizerLastEPCallbacks.push_back(
          [sanitizer](ModulePassManager &pm, OptimizationLevel level) {
            llvm::AddressSanitizerOptions sanitizerOpts;
            sanitizerOpts.Recover = sanitizer.recover;
            sanitizerOpts.UseAfterScope = true;
            pm.addPass(llvm::ModuleAddressSanitizerPass(sanitizerOpts));
          });
    }
  }

  // Add passes to generate safepoints/stack maps
  /*
  pipelineStartEPCallbacks.push_back(
      [](ModulePassManager &pm, OptimizationLevel level) {
        //
  pm.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::PlaceSafepoints()));
        pm.addPass(llvm::RewriteStatepointsForGC());
      });
  */

  ModulePassManager mpm;

  // If there is a pipeline provided, parse it and populate the pass manager
  // with it
  if (optLevel == OptimizationLevel::O0) {
    for (const auto &c : pipelineStartEPCallbacks)
      pb.registerPipelineStartEPCallback(c);

    for (const auto &c : optimizerLastEPCallbacks)
      pb.registerOptimizerLastEPCallback(c);

    mpm = pb.buildO0DefaultPipeline(optLevel, /* PreLinkLTO */ false);
  } else {
    for (const auto &c : pipelineStartEPCallbacks)
      pb.registerPipelineStartEPCallback(c);
    if (config.stage != OptStage::PreLinkThinLTO) {
      for (const auto &c : optimizerLastEPCallbacks)
        pb.registerOptimizerLastEPCallback(c);
    }

    switch (config.stage) {
    case OptStage::PreLinkNoLTO:
      mpm = pb.buildPerModuleDefaultPipeline(optLevel, /*ltoPreLink=*/false);
      break;
    case OptStage::PreLinkThinLTO:
      mpm = pb.buildThinLTOPreLinkDefaultPipeline(optLevel);
      for (const auto &c : optimizerLastEPCallbacks)
        c(mpm, optLevel);
      break;
    case OptStage::PreLinkFatLTO:
      mpm = pb.buildLTOPreLinkDefaultPipeline(optLevel);
      config.useThinLTOBuffers = false;
      break;
    case OptStage::ThinLTO:
      mpm = pb.buildThinLTODefaultPipeline(optLevel, nullptr);
      break;
    case OptStage::FatLTO:
      mpm = pb.buildLTODefaultPipeline(optLevel, nullptr);
      break;
    }

    if (config.useThinLTOBuffers) {
      mpm.addPass(llvm::CanonicalizeAliasesPass());
      mpm.addPass(llvm::NameAnonGlobalPass());
    }

    mpm.run(*mod, mam);
  }

  return false;
}
