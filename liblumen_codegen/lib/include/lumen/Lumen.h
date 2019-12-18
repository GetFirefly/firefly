#ifndef LUMEN_LUMEN_H
#define LUMEN_LUMEN_H

#include "lumen/LLVM.h"
#include "lumen/RustString.h"

#include "llvm-c/Core.h"
#include "llvm-c/Object.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

enum class LLVMLumenResult { Success, Failure };

extern "C" char *LLVMLumenGetLastError(void);
extern "C" void LLVMLumenSetLastError(const char *);

extern "C" void LLVMLumenSetLLVMOptions(int Argc, char **Argv);

#endif
