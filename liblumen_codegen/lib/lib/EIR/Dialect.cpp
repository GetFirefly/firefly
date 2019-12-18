#include "eir/Dialect.h"
#include "eir/Ops.h"
#include "eir/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

namespace M = mlir;

using namespace eir;

/// Create an instance of the EIR dialect, owned by the context.
///
/// This is where EIR types, operations, and attributes are registered.
EirDialect::EirDialect(M::MLIRContext *ctx)
    : M::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "eir/Ops.cpp.inc"
      >();
  // addInterfaces<EirInlinerInterface>();
  addTypes<AtomType>();
  addTypes<BooleanType>();
  addTypes<FixnumType>();
  addTypes<BigIntType>();
  addTypes<FloatType>();
  addTypes<PackedFloatType>();
  addTypes<NilType>();
  addTypes<ConsType>();
  addTypes<MapType>();
  addTypes<TupleType>();
  addTypes<ClosureType>();
  addTypes<BoxType>();
}
