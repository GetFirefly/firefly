#include "eir/Dialect.h"
#include "eir/Ops.h"
#include "eir/Types.h"
#include "eir/Attributes.h"

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
  addTypes<TermType,
           AnyListType,
           AnyNumberType,
           AnyIntegerType,
           AnyFloatType,
           AnyBinaryType,
           AtomType,
           BooleanType,
           FixnumType,
           BigIntType,
           FloatType,
           PackedFloatType,
           NilType,
           ConsType,
           TupleType,
           MapType,
           ClosureType,
           BinaryType,
           HeapBinType,
           BoxType,
           RefType>();

  addAttributes<AtomAttr,
                BinaryAttr,
                SeqAttr>();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
M::Operation *EirDialect::materializeConstant(M::OpBuilder &builder,
                                              M::Attribute value, M::Type type,
                                              M::Location loc) {
  return builder.create<ConstantOp>(loc, type, value);
}
