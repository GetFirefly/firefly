#include "lumen/EIR/Builder/ModuleBuilderSupport.h"

namespace lumen {
namespace eir {

// Recursively searches for the Operation which defines the given value
Operation *getDefinition(Value val) {
  if (auto arg = val.dyn_cast_or_null<BlockArgument>()) {
    Block *block = arg.getOwner();
    // If this block is the entry block, then we can't get the definition
    if (block->isEntryBlock()) return nullptr;
    // If this block has no predecessors, then we can't get the definition
    if (block->hasNoPredecessors()) return nullptr;
    // If there is a single predecessor, check the value passed as argument
    // to this block.
    //
    // If this block has multiple predecessors, we need to check if the
    // argument traces back to a single value; otherwise there are different
    // values in different branches, and we can't get a single definition
    Operation *result = nullptr;
    for (Block *pred : block->getPredecessors()) {
      auto index = arg.getArgNumber();
      Operation *found = nullptr;
      pred->walk([&](BranchOpInterface branchInterface) {
        auto op = branchInterface.getOperation();
        for (auto it = pred->succ_begin(), e = pred->succ_end(); it != e;
             ++it) {
          // If the successor isn't our block, we don't care
          Block *succ = *it;
          if (block != succ) continue;
          // No operands, nothing to do
          auto maybeSuccessorOperands =
              branchInterface.getSuccessorOperands(it.getIndex());
          if (!maybeSuccessorOperands.hasValue()) continue;
          // Otherwise take a look at the value passed as the successor block
          // argument
          auto successorOperands = maybeSuccessorOperands.getValue();
          Value candidate = successorOperands[index];
          Operation *def = getDefinition(candidate);
          if (found && def != found) return WalkResult::interrupt();
          found = def;
        }
      });
      // If this result doesn't match the last, we've found a conflict
      if (result && found != result) return nullptr;
      result = found;
    }

    return nullptr;
  }
  return val.getDefiningOp();
}

}  // namespace eir
}  // namespace lumen
