#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using llvm::cast;
using llvm::GCRegistry;
using llvm::GCStrategy;
using llvm::Optional;
using llvm::PointerType;

class LumenGC : public GCStrategy {
   public:
    LumenGC() { UseStatepoints = true; }

    Optional<bool> isGCManagedPointer(const llvm::Type *ty) const override {
        // This is only valid on pointer typed values
        const PointerType *ptrTy = cast<PointerType>(ty);
        // This is fairly conventional from what I've seen,
        // i.e. using address space 1 for managed pointers.
        // It's not reserved, or special, we could use any
        // space, but this is good enough for us.
        return ptrTy->getAddressSpace() == 1;
    }
};

static GCRegistry::Add<LumenGC> _gc("lumen", "The Lumen GC strategy");
