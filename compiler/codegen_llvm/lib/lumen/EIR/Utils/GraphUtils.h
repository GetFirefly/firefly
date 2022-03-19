// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LUMEN_UTILS_GRAPHUTILS_H_
#define LUMEN_UTILS_GRAPHUTILS_H_

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Operation.h"

#include <vector>

namespace mlir {
namespace lumen {

// Puts all of the |unsortedOps| into |sortedOps| in an arbitrary topological
// order.
// https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
//
// Preconditions: |unsortedOps| has no cycles within the set of ops.
std::vector<Operation *> sortOpsTopologically(
    const llvm::SetVector<Operation *> &unsortedOps);
template <int N>
SmallVector<Operation *, N> sortOpsTopologically(
    const SmallVector<Operation *, N> &unsortedOps) {
    auto result = sortOpsTopologically(
        llvm::SetVector<Operation *>(unsortedOps.begin(), unsortedOps.end()));
    return SmallVector<Operation *, N>(result.begin(), result.end());
}

}  // namespace lumen
}  // namespace mlir

#endif  // LUMEN_UTILS_GRAPHUTILS_H_
