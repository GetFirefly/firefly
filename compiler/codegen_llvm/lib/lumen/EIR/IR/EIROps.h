#ifndef EIR_OPS_H_
#define EIR_OPS_H_

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "lumen/EIR/Builder/ModuleBuilderSupport.h"
#include "lumen/EIR/IR/EIRAttributes.h"
#include "lumen/EIR/IR/EIRTraits.h"
#include "lumen/EIR/IR/EIRTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using ::llvm::dyn_cast;
using ::llvm::APFloat;
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::Optional;
using ::llvm::SmallVector;
using ::llvm::SmallVectorImpl;
using ::llvm::SmallString;
using ::llvm::StringRef;
using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::Value;
using ::mlir::ValueRange;
using ::mlir::Type;
using ::mlir::TypeRange;
using ::mlir::Block;
using ::mlir::BlockArgument;
using ::mlir::Identifier;
using ::mlir::Builder;
using ::mlir::OpBuilder;
using ::mlir::OpAsmPrinter;
using ::mlir::OpAsmParser;
using ::mlir::OperationState;
using ::mlir::NamedAttribute;
using ::mlir::TypeAttr;
using ::mlir::StringAttr;
using ::mlir::ArrayAttr;
using ::mlir::IntegerAttr;
using ::mlir::BoolAttr;
using ::mlir::FlatSymbolRefAttr;
using ::mlir::MutableDictionaryAttr;
using ::mlir::CallInterfaceCallable;
using ::mlir::OperandRange;
using ::mlir::MutableOperandRange;
using ::mlir::OpAsmSetValueNameFn;
using ::mlir::OwningRewritePatternList;
using ::mlir::SymbolTable;
using ::mlir::OpRewritePattern;
using ::mlir::PatternRewriter;
using ::mlir::OpFoldResult;
using ::mlir::ParseResult;
using ::mlir::LogicalResult;

namespace OpTrait = ::mlir::OpTrait;
namespace MemoryEffects = ::mlir::MemoryEffects;

namespace lumen {
namespace eir {

#include "lumen/EIR/IR/EIROpInterface.h.inc"

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
  bool isCatchAll() const { return getPatternType() == MatchPatternType::Any; }

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

/// Calculates the size of the boxed type for allocation
int64_t calculateAllocSize(unsigned pointerSizeInBits, BoxType type);

/// Performs lowering of a match operation
void lowerPatternMatch(::mlir::OpBuilder &builder, Location loc, Value selector,
                       ArrayRef<MatchBranch> branches);

//===----------------------------------------------------------------------===//
// TableGen
//===----------------------------------------------------------------------===//

/// All operations are declared in this auto-generated header
#define GET_OP_CLASSES
#include "lumen/EIR/IR/EIROps.h.inc"

}  // namespace eir
}  // namespace lumen

#endif  // EIR_OPS_H_
