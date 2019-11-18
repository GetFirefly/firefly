#include "CodeGen/CodeGen.h"
#include "CodeGen/Options.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/CommandFlags.inc"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
// Colorized output support
#include "llvm/Support/WithColor.h"

using namespace lumen;

namespace cl = llvm::cl;

// The list of input file names to compile
static cl::list<std::string>
InputFilenames(cl::desc("<source files>"),
               cl::Positional,
               cl::OneOrMore);


static cl::opt<FrontendOptions::ActionType>
EmitAction("emit", cl::desc("Select the kind of output desired"),
                   cl::init(FrontendOptions::ActionType::None),
                   cl::values(clEnumValN(FrontendOptions::ActionType::EmitAST, "ast",
                              "output the raw AST")),
                   cl::values(clEnumValN(FrontendOptions::ActionType::EmitMLIR, "mlir",
                              "output high-level MLIR")),
                   cl::values(clEnumValN(FrontendOptions::ActionType::EmitMLIR_LLVM, "llvm-mlir",
                              "output MLIR after lowering to the LLVM dialect")),
                   cl::values(clEnumValN(FrontendOptions::ActionType::EmitLLVM_IR, "llvm-ir",
                              "output LLVM IR")),
                   cl::values(clEnumValN(FrontendOptions::ActionType::EmitObject, "obj",
                              "compile the code to an object file")),
                   cl::values(clEnumValN(FrontendOptions::ActionType::EmitAssembly, "obj",
                              "compile the code to an object file")),
                   cl::values(clEnumValN(FrontendOptions::ActionType::EmitExecutable, "exe",
                              "compile the code to an executable")),
                   cl::values(clEnumValN(FrontendOptions::ActionType::RunJIT, "jit",
                              "execute the code with the JIT by calling the main function")));

// The file to write the final executable to,
// in the case of multiple files, we use the basename
// of the input, and write to the same directory as the
// specified output filename, otherwise we write to the
// same directory as the input filename
static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::init("-"),
               cl::value_desc("filename"));

// When enabled, produces additional information about what the compiler is doing
static cl::opt<bool>
Verbose("v", cl::desc("Enable verbose output"));

// The optimization level to apply to generated code
static cl::opt<char>
OptLevel("O",
         cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                  "(default = '-O2')"),
         cl::Prefix,
         cl::ZeroOrMore,
         cl::init(' '));

// A user-specified target triple
static cl::opt<std::string>
TargetTriple("target",
            cl::desc("Set target triple, e.g. x86_64-apple-darwin"
                     "(default is host triple)"),
            cl::value_desc("{arch}-{vendor}-{os}[-{environment}]"));

// A list of include paths to search
static cl::list<std::string>
IncludeDirs("I", cl::desc("Add paths to include search path"));

// Initializes LLVM prerequisites
void initializeLLVM() {
    using PassRegistry = llvm::PassRegistry;

    // Initialize targets first, so that --version shows registered targets.
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllDisassemblers();

    // Initialize codegen and IR passes used by the compiler
    PassRegistry& registry = *PassRegistry::getPassRegistry();
    llvm::initializeCore(registry);
    llvm::initializeTarget(registry);
    llvm::initializeCodeGen(registry);
    llvm::initializeAnalysis(registry);
    llvm::initializeTransformUtils(registry);
    llvm::initializeScalarOpts(registry);
    llvm::initializeLoopStrengthReducePass(registry);
    llvm::initializeLowerIntrinsicsPass(registry);
    llvm::initializeEntryExitInstrumenterPass(registry);
    llvm::initializePostInlineEntryExitInstrumenterPass(registry);
    llvm::initializeUnreachableBlockElimLegacyPassPass(registry);
    llvm::initializeConstantHoistingLegacyPassPass(registry);
    llvm::initializeScalarOpts(registry);
    llvm::initializeVectorization(registry);
    llvm::initializeScalarizeMaskedMemIntrinPass(registry);
    llvm::initializeExpandReductionsPass(registry);
    llvm::initializeHardwareLoopsPass(registry);

    // Initialize debugging passes.
    llvm::initializeScavengerTestPass(registry);

    // Register our Dialects with MLIR
    mlir::registerDialect<ErlangDialect>();
    mlir::registerPassManagerCLOptions();
}

CodeGenContext CodeGenContextCreate(int argc, char **argv) {
  initializeLLVM();

  cl::ParseCommandLineOptions(argc, argv, "lumen compiler\n");

  // If -mcpu=help or -mattrs=help are given, print help information instead of compiling
  bool skip = (MCPU == "help") || (!MAttrs.empty() && MAttrs.front() == "help");

  // If we are supposed to override the target triple, do so now.
  std::string triple;
  if (!TargetTriple.empty()) {
    auto tt = Triple::normalize(StringRef(TargetTriple));
    triple = tt.getTriple();
  } else {
    triple = llvm::sys::getProcessTriple();
  }

  CodeGenOpt::Level optLevel = CodeGenOpt::Default;
  switch (OptLevel) {
    case ' ': break;
    case '0': optLevel = CodeGenOpt::None; break;
    case '1': optLevel = CodeGenOpt::Less; break;
    case '2': optLevel = CodeGenOpt::Default; break;
    case '3': optLevel = CodeGenOpt::Aggressive; break;
    default:
      WithColor::error(errs(), progname) << "invalid optimization level.\n";
        return 1;
  }

  TargetOptions targetOpts = InitTargetOptionsFromCodeGenFlags();
  targetOpts.GuaranteedTailCallOpt = true;
  targetOpts.MCOptions.AsmVerbose = true;
  targetOpts.MCOptions.PreserveAsmComments = true;
  targetOpts.MCOptions.IASSearchPaths = IncludeDirs;

  auto codeModel = getCodeModel();
  auto relocModel = getRelocModel();

  auto opts = new Options();
  opts->setVerbosity(Verbose);
  opts->setCpu(getCPUStr());
  opts->setArch(MArch);
  opts->setTriple(triple);
  opts->setTargetOptions(targetOpts);
  opts->setOptLevel(optLevel);

  if (auto result = opts->getTargetMachine()) {
      auto ctx = new Context(std::move(opts));
      return (CodeGenContext)(ctx);
  } else {
      WithColor::error(llvm::errs(), "lumen") << result.takeError();
      return nullptr;
  }
}

void CodeGenContextDipose(CodeGenContext ctx) {
    delete ctx;
}