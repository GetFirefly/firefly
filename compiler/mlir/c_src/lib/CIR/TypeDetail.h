#pragma once

#include "CIR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace cir {
namespace detail {

//===----------------------------------------------------------------------===//
// CIRFunTypeStorage.
//===----------------------------------------------------------------------===//

struct CIRFunTypeStorage : public TypeStorage {
public:
  using KeyTy = std::tuple<FunctionType, ArrayRef<Type>>;

  CIRFunTypeStorage(FunctionType calleeType, ArrayRef<Type> envTypes)
      : calleeType(calleeType), isThin(envTypes.size() == 0),
        envArity(envTypes.size()), envTypes(envTypes.data()) {}

  /// Hook into the type uniquing infrastructure.
  static CIRFunTypeStorage *construct(TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    return new (allocator.allocate<CIRFunTypeStorage>()) CIRFunTypeStorage(
        std::get<0>(key), allocator.copyInto(std::get<1>(key)));
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    // LLVM doesn't like hashing bools in tuples.
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
  }

  bool operator==(const KeyTy &key) const {
    auto otherEnv = std::get<1>(key);
    if (envArity != otherEnv.size()) {
      return false;
    }
    return calleeType == std::get<0>(key) && getEnvTypes() == otherEnv;
  }

  KeyTy getAsKey() const {
    return KeyTy(calleeType, ArrayRef<Type>(envTypes, (size_t)envArity));
  }

  FunctionType getCalleeType() const { return calleeType; }

  ArrayRef<Type> getEnvTypes() const {
    return ArrayRef<Type>(envTypes, envArity);
  }

private:
  FunctionType calleeType;
  bool isThin;
  unsigned envArity;
  const Type *envTypes;
};

} // namespace detail
} // namespace cir
} // namespace mlir
