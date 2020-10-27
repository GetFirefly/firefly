#ifndef EIR_OPS_H_
#define EIR_OPS_H_

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "lumen/EIR/Builder/ModuleBuilderSupport.h"
#include "lumen/EIR/IR/EIRAttributes.h"
#include "lumen/EIR/IR/EIRTraits.h"
#include "lumen/EIR/IR/EIRTypes.h"

using ::llvm::APFloat;
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::dyn_cast;
using ::llvm::Optional;
using ::llvm::SmallString;
using ::llvm::SmallVector;
using ::llvm::SmallVectorImpl;
using ::llvm::StringRef;
using ::mlir::ArrayAttr;
using ::mlir::Block;
using ::mlir::BlockArgument;
using ::mlir::BoolAttr;
using ::mlir::Builder;
using ::mlir::CallInterfaceCallable;
using ::mlir::DiagnosticSeverity;
using ::mlir::FlatSymbolRefAttr;
using ::mlir::Identifier;
using ::mlir::IntegerAttr;
using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::MutableDictionaryAttr;
using ::mlir::MutableOperandRange;
using ::mlir::NamedAttribute;
using ::mlir::OpAsmParser;
using ::mlir::OpAsmPrinter;
using ::mlir::OpAsmSetValueNameFn;
using ::mlir::OpBuilder;
using ::mlir::OperandRange;
using ::mlir::Operation;
using ::mlir::OperationState;
using ::mlir::OpFoldResult;
using ::mlir::OpRewritePattern;
using ::mlir::OwningRewritePatternList;
using ::mlir::ParseResult;
using ::mlir::PatternRewriter;
using ::mlir::StringAttr;
using ::mlir::SymbolTable;
using ::mlir::Type;
using ::mlir::TypeAttr;
using ::mlir::TypeRange;
using ::mlir::Value;
using ::mlir::ValueRange;

namespace OpTrait = ::mlir::OpTrait;
namespace MemoryEffects = ::mlir::MemoryEffects;

#include "lumen/EIR/IR/EIROpInterface.h"

namespace lumen {
namespace eir {

class MatchBranch {
   public:
    MatchBranch(Location loc, Block *dest, ArrayRef<Value> destArgs,
                std::unique_ptr<MatchPattern> pattern)
        : loc(loc),
          dest(dest),
          destArgs(destArgs.begin(), destArgs.end()),
          pattern(std::move(pattern)) {}

    Location getLoc() const { return loc; }
    Block *getDest() const { return dest; }
    ArrayRef<Value> getDestArgs() const { return destArgs; }
    MatchPatternType getPatternType() const { return pattern->getKind(); }
    bool isCatchAll() const {
        return getPatternType() == MatchPatternType::Any;
    }

    MatchPattern *getPattern() const { return pattern.get(); }

    template <typename T>
    T *getPatternTypeOrNull() const {
        T *result = dyn_cast<T>(getPattern());
        return result;
    }

   private:
    Location loc;
    Block *dest;
    SmallVector<Value, 3> destArgs;
    std::unique_ptr<MatchPattern> pattern;
};

/// Performs lowering of a match operation
LogicalResult lowerPatternMatch(::mlir::OpBuilder &builder, Location loc,
                                Value selector, ArrayRef<MatchBranch> branches);

}  // namespace eir
}  // namespace lumen

//===----------------------------------------------------------------------===//
// TableGen
//===----------------------------------------------------------------------===//

/// All operations are declared in this auto-generated header
#define GET_OP_CLASSES
#include "lumen/EIR/IR/EIROps.h.inc"

#endif  // EIR_OPS_H_
