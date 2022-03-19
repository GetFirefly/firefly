#include <llvm-c/Error.h>
#include <llvm-c/LLJIT.h>
#include <llvm-c/Orc.h>
#include <llvm-c/Types.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ExecutionEngine/JITLink/EHFrameSupport.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

#include <memory>

using ::llvm::Expected;
using ::llvm::orc::DynamicLibrarySearchGenerator;
using ::llvm::orc::IRCompileLayer;
using ::llvm::orc::StaticLibraryDefinitionGenerator;

typedef struct LLVMOpaqueOrcIRCompilerRef *LLVMOrcIRCompilerRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::ExecutionSession,
                                   LLVMOrcExecutionSessionRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::MemoryBuffer, LLVMMemoryBufferRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::Module, LLVMModuleRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::DefinitionGenerator,
                                   LLVMOrcDefinitionGeneratorRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::IRCompileLayer::IRCompiler,
                                   LLVMOrcIRCompilerRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::JITTargetMachineBuilder,
                                   LLVMOrcJITTargetMachineBuilderRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::LLJITBuilder,
                                   LLVMOrcLLJITBuilderRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::orc::ObjectLinkingLayer,
                                   LLVMOrcObjectLayerRef);

extern "C" LLVMOrcExecutionSessionRef LLVMOrcCreateExecutionSession() {
  auto *es = new llvm::orc::ExecutionSession();
  return wrap(es);
}

extern "C" void
LLVMOrcDisposeExecutionSession(LLVMOrcExecutionSessionRef session) {
  llvm::orc::ExecutionSession *es = unwrap(session);
  delete es;
}

extern "C" void
LLVMOrcLLJITBuilderSetExecutionSession(LLVMOrcLLJITBuilderRef builder,
                                       LLVMOrcExecutionSessionRef session) {
  llvm::orc::ExecutionSession *es = unwrap(session);
  unwrap(builder)->setExecutionSession(
      std::unique_ptr<llvm::orc::ExecutionSession>(es));
}

extern "C" void LLVMOrcObjectLinkingLayerAddEHFrameRegistrationPlugin(
    LLVMOrcObjectLayerRef layer) {
  auto oll = unwrap(layer);
  auto &session = oll->getExecutionSession();
  auto registrar = std::make_unique<llvm::jitlink::InProcessEHFrameRegistrar>();
  auto plugin = std::make_unique<llvm::orc::EHFrameRegistrationPlugin>(
      session, std::move(registrar));
  oll->addPlugin(std::move(plugin));
}

extern "C" LLVMErrorRef LLVMOrcCreateDynamicLibrarySearchGeneratorLoad(
    LLVMOrcDefinitionGeneratorRef *result, const char *fileName,
    char globalPrefix) {
  auto gen = DynamicLibrarySearchGenerator::Load(fileName, globalPrefix);
  if (!gen) {
    *result = nullptr;
    return llvm::wrap(gen.takeError());
  }
  *result = wrap(gen->release());
  return LLVMErrorSuccess;
}

extern "C" LLVMErrorRef LLVMOrcCreateStaticLibraryDefinitionGeneratorLoad(
    LLVMOrcDefinitionGeneratorRef *result, LLVMOrcObjectLayerRef layer,
    const char *fileName) {
  llvm::orc::ObjectLayer *l = unwrap(layer);
  auto gen = StaticLibraryDefinitionGenerator::Load(*l, fileName);
  if (!gen) {
    *result = nullptr;
    return llvm::wrap(gen.takeError());
  }
  *result = wrap(gen->release());
  return LLVMErrorSuccess;
}

typedef LLVMErrorRef (*LLVMOrcLLJITBuilderCompileFunctionCreatorFunction)(
    LLVMOrcIRCompilerRef *, void *, LLVMOrcJITTargetMachineBuilderRef);

extern "C" void LLVMOrcLLJITBuilderSetCompileFunctionCreator(
    LLVMOrcLLJITBuilderRef builder,
    LLVMOrcLLJITBuilderCompileFunctionCreatorFunction f, void *ctx) {
  unwrap(builder)->setCompileFunctionCreator(
      [=](llvm::orc::JITTargetMachineBuilder jtmb)
          -> Expected<std::unique_ptr<IRCompileLayer::IRCompiler>> {
        LLVMOrcIRCompilerRef ptr = nullptr;
        auto err = llvm::unwrap(f(&ptr, ctx, wrap(&jtmb)));
        if (!err)
          return std::unique_ptr<IRCompileLayer::IRCompiler>(unwrap(ptr));
        return std::move(err);
      });
}

typedef LLVMMemoryBufferRef (*LLVMObjectCacheGetObjectFunc)(void *,
                                                            LLVMModuleRef);
typedef void (*LLVMObjectCacheNotifyCompiledFunc)(void *, LLVMModuleRef,
                                                  const char *, size_t);

class ObjectCacheWrapper : public llvm::ObjectCache {
public:
  ObjectCacheWrapper(void *context, LLVMObjectCacheGetObjectFunc getter,
                     LLVMObjectCacheNotifyCompiledFunc notifier)
      : enabled(getter != nullptr && notifier != nullptr), context(context),
        getFunc(getter), notifyFunc(notifier) {}

  ~ObjectCacheWrapper() = default;

protected:
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *m) override;
  void notifyObjectCompiled(const llvm::Module *m,
                            llvm::MemoryBufferRef obj) override;

private:
  bool enabled;
  void *context;
  LLVMObjectCacheGetObjectFunc getFunc;
  LLVMObjectCacheNotifyCompiledFunc notifyFunc;
};

// Implements llvm::ObjectCache::notifyObjectCompiled, called from CompileLayer
void ObjectCacheWrapper::notifyObjectCompiled(const llvm::Module *m,
                                              llvm::MemoryBufferRef obj) {
  if (!enabled)
    return;

  assert(m && "Caching requires module");

  notifyFunc(context, llvm::wrap(m), obj.getBufferStart(), obj.getBufferSize());
}

// Implements llvm::ObjectCache::getObject, called from CompileLayer
std::unique_ptr<llvm::MemoryBuffer>
ObjectCacheWrapper::getObject(const llvm::Module *m) {
  if (!enabled)
    return nullptr;

  assert(m && "Lookup requires module");

  auto mb = unwrap(getFunc(context, llvm::wrap(m)));
  if (mb == nullptr)
    return nullptr;

  return std::unique_ptr<llvm::MemoryBuffer>(mb);
}

typedef struct OpaqueObjectCacheWrapper *ObjectCacheWrapperRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(ObjectCacheWrapper, ObjectCacheWrapperRef);

extern "C" ObjectCacheWrapperRef
LLVMCreateObjectCacheWrapper(void *context, LLVMObjectCacheGetObjectFunc getter,
                             LLVMObjectCacheNotifyCompiledFunc notifier) {
  auto *cache = new ObjectCacheWrapper(context, getter, notifier);
  return wrap(cache);
}

extern "C" void LLVMDisposeObjectCacheWrapper(ObjectCacheWrapperRef cache) {
  delete unwrap(cache);
}

extern "C" LLVMErrorRef
LLVMOrcCreateOwningCompiler(LLVMOrcIRCompilerRef *result,
                            LLVMOrcJITTargetMachineBuilderRef builder,
                            ObjectCacheWrapperRef wrapper) {
  auto tm = unwrap(builder)->createTargetMachine();
  if (!tm)
    return wrap(tm.takeError());
  auto compiler =
      new llvm::orc::TMOwningSimpleCompiler(std::move(*tm), unwrap(wrapper));
  *result = wrap(compiler);
  return nullptr;
}
