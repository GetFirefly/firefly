#ifndef EIR_TRAITS_H
#define EIR_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

template <typename ConcreteType>
class YieldPoint : public mlir::OpTrait::TraitBase<ConcreteType, YieldPoint> {
 public:
   static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
     return success();
   }
};

} // namespace OpTrait
} // namespace mlir

#endif
