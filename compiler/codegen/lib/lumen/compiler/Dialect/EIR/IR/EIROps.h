#ifndef EIR_OPS_H_
#define EIR_OPS_H_

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTraits.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Translation/ModuleBuilderSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffects.h"

using namespace mlir;
using ::llvm::APFloat;
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::Optional;
using ::llvm::SmallVector;
using ::llvm::StringRef;

namespace lumen {
namespace eir {

#include "lumen/compiler/Dialect/EIR/IR/EIROpInterface.h.inc"

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
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h.inc"

}  // namespace eir
}  // namespace lumen

#endif  // EIR_OPS_H_
