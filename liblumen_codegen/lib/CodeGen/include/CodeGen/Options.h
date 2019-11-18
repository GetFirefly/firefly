#ifndef LUMEN_CODEGEN_OPTIONS_H
#define LUMEN_CODEGEN_OPTIONS_H

#include <string>
#include <memory>
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/CodeGen/CommandFlags.inc"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/MC/SubtargetFeature.h"

namespace lumen {

enum class ActionType {
    None,           // No specific action
    EmitLLVM_IR,    // Parse and dump the LLVM IR
    RunJIT,         // Parse and then execute via JIT
    EmitAssembly,   // Parse and compile to an object, but output textual assembly
    EmitObject,     // Parse and compile to an object file
    EmitExecutable, // Parse and compile to an executable
};

class Options {
    ActionType Action;
    std::string OutputFilename;
    bool VerboseOutput;

    llvm::CodeGenOpt::Level OptLevel = llvm::CodeGenOpt::Default;
    llvm::Reloc::Model RelocModel = llvm::Reloc::Static;
    llvm::CodeModel::Model CodeModel = llvm::CodeModel::Small;

    std::string MCPU;
    std::string MArch;
    Triple TargetTriple;
    TargetOptions TargetOpts;
    const Target *Target;
    SubtargetFeatures Features;
    std::unique_ptr<TargetMachine> Machine;

public:
    Options() : Action(ActionType::None) {}

    void setVerbosity(bool verbose) {
        Verbose = verbose;
    }
    void setAction(LumenActionType actionType) {
        switch (actionType) {
            case ActionType_None:
                Action = ActionType::None;
                break;
            case ActionType_EmitLLVM_IR:
                Action = ActionType::EmitLLVM_IR;
                break;
            case ActionType_RunJIT:
                Action = ActionType::RunJIT;
                break;
            case ActionType_EmitAssembly:
                Action = ActionType::EmitAssembly;
                break;
            case ActionType_EmitObject:
                Action = ActionType::EmitObject;
                break;
            case ActionType_EmitExecutable:
                Action = ActionType::EmitExecutable;
                break;
        }
    }
    void setAction(ActionType actionType) {
        Action = actionType;
    }
    void setCpu(std::string cpu) {
        MCPU = std::move(cpu);
    }
    void setArch(std::string arch) {
        MArch = std::move(arch);
    }
    void setTriple(std::string triple) {
        TargetTriple = Triple(triple):

        Features = SubtargetFeatures();
        Features.getDefaultSubtargetFeatures(TargetTriple);

        // On wasmXX, set up better default features
        auto arch = TargetTriple.getArch();
        switch (arch) {
            case Triple::ArchType::wasm32:;
            case Triple::ArchType::wasm64:
                Features.AddFeature("-atomics");
                Features.AddFeature("+sign-ext");
                Features.AddFeature("+bulk-memory");
                Features.AddFeature("+tail-call");
                Features.AddFeature("+exception-handling");
                Features.AddFeature("+mutable-globals");
            default:
                break;
        }
    }
    void setTargetOptions(TargetOptions options) {
        TargetOpts = options;
    }
    void setOptLevel(llvm::CodeGenOpt::Level level) {
        OptLevel = level;
    }

    Expected<TargetMachine*> getTargetMachine() {
        if (Target != nullptr) {
            return Target;
        }
        // Get the target
        std::string err;
        Target = TargetRegistry::lookupTarget(MArch, TargetTriple, err);
        if (!Target) {
            return llvm::createStringError(llvm::inconvertibleErrorCode(), err);
        } else {
            Machine = std::unique_ptr<TargetMachine>(Target->createTargetMachine(
                TargetTriple.getTriple(),
                CPU,
                Features.getString(),
                TargetOpts,
                RelocModel,
                CodeModel,
                OptLevel,
                /*jit*/false
            ));
            assert(Machine != nullptr && "Unable to allocate target machine!");
            return Machine;
        }
    }
}

} // end namespace lumen
#endif